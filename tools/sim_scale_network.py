#!/usr/bin/env python3
"""Launch N OpenThread simulation nodes, form a Thread network, and report resource use.

Default simulation builds only allow 33 nodes (OPENTHREAD_SIMULATION_MAX_NETWORK_SIZE).
This script can rebuild with a larger limit when --build is passed.

By default uses Virtual Time (event-driven) for much faster formation than real-time.

Examples:
  # Rebuild for virtual-time + 1024-node capacity, then start 1000 nodes
  ulimit -n 65536
  ./tools/sim_scale_network.py --build --nodes 1000 --max-network-size 1024

  # Real-time mode (better for wall-clock CPU sampling realism)
  ./tools/sim_scale_network.py --real-time --nodes 100

  # Keep network running for interactive inspection
  ./tools/sim_scale_network.py --nodes 100 --hold 300
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

OT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OT_CLI = OT_ROOT / "build" / "simulation" / "examples" / "apps" / "cli" / "ot-cli-ftd"
THREAD_CERT_DIR = OT_ROOT / "tests" / "scripts" / "thread-cert"

NETWORK_NAME = "OT-Scale"
NETWORK_KEY = "00112233445566778899aabbccddeeff"
PANID = 0xFACE
CHANNEL = 11

DONE_OR_ERROR = re.compile(r"^(Done|Error(?: \d+:.*)?)$")
LOG_LINE = re.compile(
    r"^("
    r"\[(NONE|CRIT|WARN|NOTE|INFO|DEBG)\]"
    r"|-.*-+: "
    r"|\[[DINWC\-]\] (?=[\w\-]{14}:)\w+-*:"
    r")"
)

# Shared by advance()/wait helpers; set in main().
_TIME_ENGINE = None


@dataclass
class ProcStats:
    rss_kb: int = 0
    vsz_kb: int = 0
    cpu_pct: float = 0.0
    threads: int = 0


@dataclass
class Node:
    node_id: int
    proc: subprocess.Popen
    lock: threading.Lock = field(default_factory=threading.Lock)
    cmd_lock: threading.Lock = field(default_factory=threading.Lock)
    reader: Optional[threading.Thread] = None
    lines: List[str] = field(default_factory=list)
    closed: bool = False

    def start_reader(self) -> None:
        self.reader = threading.Thread(target=self._read_loop, daemon=True, name=f"ot-reader-{self.node_id}")
        self.reader.start()

    def _read_loop(self) -> None:
        assert self.proc.stdout is not None
        try:
            for raw in self.proc.stdout:
                line = raw.rstrip("\r\n")
                with self.lock:
                    self.lines.append(line)
        except (ValueError, OSError):
            # stdout closed during shutdown
            return

    def write_cmd(self, command: str) -> None:
        """Send a CLI command without waiting for Done (caller pumps VT)."""
        if self.proc.poll() is not None:
            raise RuntimeError(f"node {self.node_id} exited with {self.proc.returncode}")
        assert self.proc.stdin is not None
        with self.cmd_lock:
            with self.lock:
                self.lines.clear()
            self.proc.stdin.write(command + "\n")
            self.proc.stdin.flush()

    def cmd(self, command: str, timeout: float = 10.0) -> List[str]:
        if self.proc.poll() is not None:
            raise RuntimeError(f"node {self.node_id} exited with {self.proc.returncode}")

        assert self.proc.stdin is not None
        with self.cmd_lock:
            with self.lock:
                self.lines.clear()

            self.proc.stdin.write(command + "\n")
            self.proc.stdin.flush()
            pump()

            deadline = time.monotonic() + timeout
            output: List[str] = []
            saw_echo = False

            while time.monotonic() < deadline:
                pump()
                with self.lock:
                    while self.lines:
                        line = self.lines.pop(0)
                        if not saw_echo:
                            # OT CLI echoes as "> <command>"
                            if line == command or line == f"> {command}" or line.endswith(command):
                                saw_echo = True
                            continue
                        if not line or line == ">":
                            continue
                        if LOG_LINE.match(line):
                            continue
                        if DONE_OR_ERROR.match(line):
                            if line != "Done":
                                raise RuntimeError(f"node {self.node_id}: {command!r} -> {line}")
                            return output
                        output.append(line)
                time.sleep(0.001 if (_TIME_ENGINE and _TIME_ENGINE.virtual_time) else 0.005)

            raise TimeoutError(f"node {self.node_id}: timeout on {command!r}, got {output!r}")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.proc.poll() is None:
                self.proc.send_signal(signal.SIGTERM)
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=3)
        finally:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except OSError:
                pass
            try:
                if self.proc.stdout:
                    self.proc.stdout.close()
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", type=int, default=100, help="Number of simulation nodes (default: 100)")
    p.add_argument("--ot-cli", type=Path, default=DEFAULT_OT_CLI, help="Path to ot-cli-ftd")
    p.add_argument("--ot-root", type=Path, default=OT_ROOT, help="OpenThread source root")
    p.add_argument(
        "--build",
        action="store_true",
        help="Rebuild simulation with OT_SIMULATION_MAX_NETWORK_SIZE >= nodes",
    )
    p.add_argument(
        "--max-network-size",
        type=int,
        default=0,
        help="Value for OT_SIMULATION_MAX_NETWORK_SIZE when building (default: max(nodes, 128))",
    )
    p.add_argument("--join-timeout", type=float, default=0.0,
                   help="Seconds to wait for all nodes to attach (0 = auto: max(120, nodes*0.5))")
    p.add_argument("--stabilize", type=float, default=10.0, help="Extra seconds to wait after attach before sampling")
    p.add_argument("--hold", type=float, default=0.0, help="Keep nodes running N seconds after report (0=exit)")
    p.add_argument("--sample-interval", type=float, default=2.0, help="Host resource sample interval while joining")
    p.add_argument(
        "--join-batch",
        type=int,
        default=0,
        help="Start Thread on joiners in batches (0 = auto: 50 if nodes>=200 else all-at-once)",
    )
    p.add_argument(
        "--join-batch-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between joiner batches (reduces radio storm)",
    )
    p.add_argument(
        "--router-eligible",
        type=int,
        default=32,
        help="How many nodes keep router-eligible (rest become children only; Thread max routers is 32)",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=32,
        help="Max parallel CLI operations when configuring / polling state",
    )
    p.add_argument(
        "--port-offset",
        type=int,
        default=-1,
        help="PORT_OFFSET for OT simulation radio ports (-1 = auto)",
    )
    p.add_argument(
        "--virtual-time",
        dest="virtual_time",
        action="store_true",
        default=True,
        help="Use Virtual Time simulation (default)",
    )
    p.add_argument(
        "--real-time",
        dest="virtual_time",
        action="store_false",
        help="Use real-time simulation (slower; wall-clock CPU more realistic)",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write full report JSON (default: <script_dir>/ot_scale_<N>.json)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_fd_limit(num_nodes: int) -> None:
    """Raise RLIMIT_NOFILE so parent can keep stdin/stdout pipes for every node.

    Each node costs ~2 FDs in the parent (PIPE stdin + stdout). Default soft
    limit is often 1024, which fails around ~500 nodes with Errno 24.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Parent pipes + slack for Python/runtime; children also need sockets but
    # those count against each child process, not this limit directly.
    want = num_nodes * 4 + 1024
    if hard != resource.RLIM_INFINITY:
        want = min(want, hard)
    if soft < want:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            log(f"Raised RLIMIT_NOFILE soft limit to {soft} (hard={hard})")
        except (ValueError, OSError) as exc:
            need = num_nodes * 4 + 1024
            raise SystemExit(
                f"Too many open files: soft limit is {soft}, need about {need} "
                f"for {num_nodes} nodes ({exc}).\n"
                f"Raise it in this shell first, e.g.:\n"
                f"  ulimit -n {need}\n"
                f"then re-run the script."
            ) from exc

    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    min_needed = num_nodes * 3 + 256
    if soft < min_needed:
        raise SystemExit(
            f"Open-files limit ({soft}) is still too low for {num_nodes} nodes "
            f"(need >= {min_needed}). Run: ulimit -n {num_nodes * 4 + 1024}"
        )


def build_simulation(ot_root: Path, max_network_size: int, virtual_time: bool) -> Path:
    script = ot_root / "script" / "cmake-build"
    if not script.is_file():
        raise FileNotFoundError(script)

    vt = "ON" if virtual_time else "OFF"
    log(f"Building simulation MAX_NETWORK_SIZE={max_network_size} VIRTUAL_TIME={vt} ...")
    env = os.environ.copy()
    env["OT_CMAKE_NINJA_TARGET"] = "ot-cli-ftd"
    cmd = [
        str(script),
        "simulation",
        f"-DOT_SIMULATION_MAX_NETWORK_SIZE={max_network_size}",
        f"-DOT_SIMULATION_VIRTUAL_TIME={vt}",
        "-DOT_SIMULATION_VIRTUAL_TIME_UART=OFF",
    ]
    subprocess.check_call(cmd, cwd=ot_root, env=env)
    binary = ot_root / "build" / "simulation" / "examples" / "apps" / "cli" / "ot-cli-ftd"
    if not binary.is_file():
        raise FileNotFoundError(f"build succeeded but binary missing: {binary}")
    return binary


def _cmake_cache_value(ot_cli: Path, key: str) -> Optional[str]:
    # ot-cli-ftd -> .../build/simulation/examples/apps/cli/ -> build/simulation/
    build_dir = ot_cli.resolve().parents[3]
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    prefix = f"{key}:"
    for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[-1].strip()
    return None


def cmake_virtual_time_enabled(ot_cli: Path) -> Optional[bool]:
    """Return True/False if CMakeCache knows VIRTUAL_TIME, else None."""
    val = _cmake_cache_value(ot_cli, "OT_SIMULATION_VIRTUAL_TIME")
    if val is None:
        return None
    return val.upper() in ("ON", "1", "TRUE")


def cmake_max_network_size(ot_cli: Path) -> Optional[int]:
    """Compiled OT_SIMULATION_MAX_NETWORK_SIZE; must match VirtualTime.MAX_NODES."""
    val = _cmake_cache_value(ot_cli, "OT_SIMULATION_MAX_NETWORK_SIZE")
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


class TimeEngine:
    """Real-time sleep or VirtualTime event advance."""

    def __init__(self, virtual_time: bool, max_nodes: int, port_offset: int, work_dir: Path):
        self.virtual_time = virtual_time
        self.sim_seconds = 0.0
        self.work_dir = work_dir
        self._vt = None
        self._lock = threading.Lock()
        if not virtual_time:
            return

        work_dir.mkdir(parents=True, exist_ok=True)
        # OT simulation and VirtualTime both use relative vt.<port>.sock paths.
        os.chdir(work_dir)

        if str(THREAD_CERT_DIR) not in sys.path:
            sys.path.insert(0, str(THREAD_CERT_DIR))
        os.environ["PORT_OFFSET"] = str(port_offset)
        os.environ["OT_VT_USE_UNIX_SOCKET"] = "1"
        os.environ.setdefault("TEST_NAME", "sim_scale_network")

        import simulator as ot_sim  # type: ignore

        ot_sim.VirtualTime.MAX_NODES = max_nodes
        ot_sim.VirtualTime.PORT_OFFSET = port_offset
        ot_sim.VirtualTime.USE_UNIX_SOCKET = True
        self._vt = ot_sim.VirtualTime(message_factory=None)
        log(f"VirtualTime ready cwd={work_dir} port={self._vt.port} MAX_NODES={max_nodes}")

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            return
        if self._vt is None:
            time.sleep(seconds)
        else:
            with self._lock:
                self._vt.go(seconds)
        self.sim_seconds += seconds

    def pump(self) -> None:
        """Process pending VT events without advancing time."""
        if self._vt is None:
            return
        with self._lock:
            try:
                self._vt.go(0)
            except BlockingIOError:
                # Non-blocking unix sockets can spuriously EAGAIN here.
                pass

    def device_count(self) -> int:
        if self._vt is None:
            return 0
        with self._lock:
            return len(self._vt.devices)

    def wait_new_device(self, prev_count: int, timeout: float = 2.0) -> bool:
        """Pump until a new VT device appears or timeout."""
        if self._vt is None:
            time.sleep(0.005)
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump()
            if self.device_count() > prev_count:
                return True
            time.sleep(0.001)
        return False

    def stop(self) -> None:
        if self._vt is not None:
            try:
                with self._lock:
                    self._vt.stop()
            except Exception:  # noqa: BLE001
                pass
            self._vt = None


def advance(seconds: float) -> None:
    if _TIME_ENGINE is None:
        time.sleep(seconds)
        return
    t0 = time.monotonic()
    _TIME_ENGINE.advance(seconds)
    wall = time.monotonic() - t0
    if wall >= 2.0:
        log(f"advance({seconds:.1f}s sim) took {wall:.1f}s wall")


def pump() -> None:
    if _TIME_ENGINE is not None:
        _TIME_ENGINE.pump()


def max_safe_vt_port_offset(max_nodes: int) -> int:
    """Largest PORT_OFFSET that keeps node ports in VirtualTime radio range.

    simulator.VirtualTime._is_radio() treats ports < BASE_PORT*2 (18000) as radio.
    Node port = 9000 + PORT_OFFSET*(MAX_NODES+1) + node_id, so large offsets break.
    """
    # 9000 + offset*(max_nodes+1) + max_nodes < 18000
    room = 9000 - max_nodes
    if room <= 0:
        return 0
    return max(0, (room - 1) // (max_nodes + 1))


def pick_port_offset(ot_cli: Path, preferred: int = 0) -> int:
    """Find a PORT_OFFSET where node 1 can bind radio sockets."""
    candidates = [preferred] + [n for n in range(0, 50) if n != preferred]
    for offset in candidates:
        env = os.environ.copy()
        env["PORT_OFFSET"] = str(offset)
        proc = subprocess.Popen(
            [str(ot_cli), "1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(ot_cli.parent),
        )
        try:
            out, _ = proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            if out and "address already in use" in out.lower():
                continue
            return offset
        text = (out or "").lower()
        if "address already in use" in text:
            continue
        if "invalid nodeid" in text:
            continue
        return offset
    raise RuntimeError("could not find a free PORT_OFFSET")


def probe_max_node_id(ot_cli: Path) -> int:
    """Binary-search the highest accepted node id for this build."""

    def accepts(node_id: int) -> bool:
        proc = subprocess.Popen(
            [str(ot_cli), str(node_id)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ot_cli.parent),
        )
        try:
            out, _ = proc.communicate(timeout=1.5)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            return "Invalid NodeId" not in (out or "")
        return "Invalid NodeId" not in (out or "")

    lo, hi, best = 1, 512, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if accepts(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _read_jiffies(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    return int(fields[13]) + int(fields[14])


def read_proc_stats(pid: int, prev: Optional[Tuple[int, float]] = None) -> Tuple[ProcStats, Tuple[int, float]]:
    """Return process stats and (utime+stime jiffies, monotonic time) for CPU delta."""
    stats = ProcStats()
    status_path = Path(f"/proc/{pid}/status")
    now = time.monotonic()

    if status_path.exists():
        for line in status_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                stats.rss_kb = int(line.split()[1])
            elif line.startswith("VmSize:"):
                stats.vsz_kb = int(line.split()[1])
            elif line.startswith("Threads:"):
                stats.threads = int(line.split()[1])

    try:
        jiffies = _read_jiffies(pid)
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        jiffies = 0

    if prev is not None and jiffies:
        prev_j, prev_t = prev
        dt = max(now - prev_t, 1e-6)
        clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        stats.cpu_pct = max(0.0, (jiffies - prev_j) * 100.0 / clk_tck / dt)

    return stats, (jiffies, now)


def host_snapshot(
    nodes: List[Node],
    prev_cpu: Dict[int, Tuple[int, float]],
) -> Tuple[Dict, Dict[int, Tuple[int, float]]]:
    per_node = []
    total_rss = 0
    total_vsz = 0
    total_cpu = 0.0
    total_threads = 0
    new_prev: Dict[int, Tuple[int, float]] = {}

    for node in nodes:
        pid = node.proc.pid
        st, marker = read_proc_stats(pid, prev_cpu.get(pid))
        new_prev[pid] = marker
        total_rss += st.rss_kb
        total_vsz += st.vsz_kb
        total_cpu += st.cpu_pct
        total_threads += st.threads
        per_node.append(
            {
                "node_id": node.node_id,
                "pid": pid,
                "rss_kb": st.rss_kb,
                "vsz_kb": st.vsz_kb,
                "cpu_pct": round(st.cpu_pct, 2),
                "threads": st.threads,
            }
        )

    rss_list = [x["rss_kb"] for x in per_node]
    snap = {
        "timestamp": time.time(),
        "count": len(nodes),
        "total_rss_kb": total_rss,
        "total_rss_mb": round(total_rss / 1024, 2),
        "total_vsz_kb": total_vsz,
        "total_vsz_mb": round(total_vsz / 1024, 2),
        "total_cpu_pct": round(total_cpu, 2),
        "total_threads": total_threads,
        "rss_kb_avg": round(statistics.mean(rss_list), 1) if rss_list else 0,
        "rss_kb_min": min(rss_list) if rss_list else 0,
        "rss_kb_max": max(rss_list) if rss_list else 0,
        "rss_kb_p50": int(statistics.median(rss_list)) if rss_list else 0,
        "self_maxrss_kb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "per_node": per_node,
    }
    return snap, new_prev


def spawn_node(ot_cli: Path, node_id: int, env: dict) -> Node:
    prev_devices = _TIME_ENGINE.device_count() if _TIME_ENGINE is not None else 0
    try:
        proc = subprocess.Popen(
            [str(ot_cli), str(node_id)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(ot_cli.parent),
        )
    except OSError as exc:
        if exc.errno == 24:  # EMFILE
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            raise OSError(
                exc.errno,
                f"Too many open files while starting node {node_id} "
                f"(RLIMIT_NOFILE soft={soft}, hard={hard}). "
                f"Try: ulimit -n {max(soft * 2, 8192)}",
            ) from exc
        raise
    node = Node(node_id=node_id, proc=proc)
    node.start_reader()

    if _TIME_ENGINE is not None and _TIME_ENGINE.virtual_time:
        # Wait only until VT accepts this node (was a fixed ~300ms sleep).
        if not _TIME_ENGINE.wait_new_device(prev_devices, timeout=2.0):
            if proc.poll() is not None:
                raise RuntimeError(_spawn_fail_msg(node, proc.returncode))
            # Continue anyway; CLI may still work after a couple pumps.
            pump()
    else:
        time.sleep(0.005)
        pump()

    if proc.poll() is not None:
        raise RuntimeError(_spawn_fail_msg(node, proc.returncode))
    return node


def _spawn_fail_msg(node: Node, returncode: Optional[int]) -> str:
    time.sleep(0.05)
    with node.lock:
        tail = "".join(node.lines[-20:]).strip()
    detail = f", output={tail!r}" if tail else ""
    return f"failed to start node {node.node_id}, exit={returncode}{detail}"


def configure_leader(node: Node) -> str:
    """Create a fresh dataset on the leader and return active dataset TLVs (hex)."""
    node.cmd("dataset init new")
    node.cmd(f"dataset networkname {NETWORK_NAME}")
    node.cmd(f"dataset networkkey {NETWORK_KEY}")
    node.cmd(f"dataset panid {PANID:#04x}")
    node.cmd(f"dataset channel {CHANNEL}")
    node.cmd("dataset commit active")
    node.cmd("routerselectionjitter 1")
    node.cmd("ifconfig up")
    node.cmd("thread start")
    # `dataset active -x` prints hex TLVs
    lines = node.cmd("dataset active -x")
    hex_tlvs = next((ln.strip() for ln in lines if ln.strip() and not ln.startswith("Active")), "")
    if not hex_tlvs:
        # Some builds only print the hex line.
        hex_tlvs = lines[0].strip() if lines else ""
    if not hex_tlvs or any(c not in "0123456789abcdefABCDEF" for c in hex_tlvs):
        raise RuntimeError(f"failed to read active dataset TLVs: {lines!r}")
    return hex_tlvs


def configure_joiner(node: Node, dataset_tlvs: str, router_eligible: bool) -> None:
    node.cmd(f"dataset set active {dataset_tlvs}")
    node.cmd("routerselectionjitter 1")
    node.cmd("mode rdn")
    if not router_eligible:
        # Avoid hundreds of nodes competing for the 32 router slots (MLE storm).
        node.cmd("routereligible disable")
    node.cmd("ifconfig up")
    node.cmd("thread start")


def _joiner_commands(dataset_tlvs: str, router_eligible: bool) -> List[str]:
    cmds = [
        f"dataset set active {dataset_tlvs}",
        "routerselectionjitter 1",
        "mode rdn",
    ]
    if not router_eligible:
        cmds.append("routereligible disable")
    cmds.extend(["ifconfig up", "thread start"])
    return cmds


def cmd_many_vt(
    items: List[Tuple[Node, str]],
    timeout: float = 30.0,
    chunk_size: int = 64,
) -> Dict[int, List[str]]:
    """Run one CLI command per node, interleaved under VirtualTime.

    Large sets are processed in chunks so one slow node cannot stall hundreds
    of peers, and so we keep pumping VT regularly.
    """
    if not items:
        return {}

    results: Dict[int, List[str]] = {}
    size = max(1, chunk_size)
    for start in range(0, len(items), size):
        chunk = items[start:start + size]
        # Scale timeout with chunk size; large datasets need more wall time.
        chunk_timeout = max(timeout, 15.0 + 0.25 * len(chunk))
        results.update(_cmd_many_vt_one_chunk(chunk, chunk_timeout))
    return results


def _cmd_many_vt_one_chunk(
    items: List[Tuple[Node, str]],
    timeout: float,
) -> Dict[int, List[str]]:
    pending: Dict[int, dict] = {}
    for node, command in items:
        if node.proc.poll() is not None:
            raise RuntimeError(f"node {node.node_id} exited with {node.proc.returncode}")
        assert node.proc.stdin is not None
        with node.cmd_lock:
            with node.lock:
                node.lines.clear()
            node.proc.stdin.write(command + "\n")
            node.proc.stdin.flush()
        pending[node.node_id] = {
            "node": node,
            "command": command,
            "output": [],
            "saw_echo": False,
        }

    pump()
    deadline = time.monotonic() + timeout
    results: Dict[int, List[str]] = {}

    while pending and time.monotonic() < deadline:
        pump()
        finished: List[int] = []
        for node_id, st in pending.items():
            node: Node = st["node"]
            command: str = st["command"]
            with node.lock:
                while node.lines:
                    line = node.lines.pop(0)
                    if not st["saw_echo"]:
                        if line == command or line == f"> {command}" or line.endswith(command):
                            st["saw_echo"] = True
                        continue
                    if not line or line == ">":
                        continue
                    if LOG_LINE.match(line):
                        continue
                    if DONE_OR_ERROR.match(line):
                        if line != "Done":
                            raise RuntimeError(f"node {node_id}: {command!r} -> {line}")
                        results[node_id] = st["output"]
                        finished.append(node_id)
                        break
                    st["output"].append(line)
        for node_id in finished:
            del pending[node_id]
        if pending:
            time.sleep(0.001)

    if pending:
        stuck = sorted(pending)
        raise TimeoutError(f"VT batch CLI timeout, pending nodes={stuck[:20]}")
    return results


def configure_joiners_vt(
    joiners: List[Node],
    dataset_tlvs: str,
    router_eligible_ids: set,
) -> None:
    """Configure joiners by broadcasting each CLI step across the batch."""
    # Nodes may have different command lists (routereligible). Split into two groups.
    groups: Dict[bool, List[Node]] = {True: [], False: []}
    for node in joiners:
        groups[node.node_id in router_eligible_ids].append(node)

    for eligible, nodes in groups.items():
        if not nodes:
            continue
        for command in _joiner_commands(dataset_tlvs, eligible):
            cmd_many_vt([(n, command) for n in nodes])


def configure_joiners_parallel(
    joiners: List[Node],
    dataset_tlvs: str,
    router_eligible_ids: set,
    parallel: int,
) -> None:
    def _one(node: Node) -> None:
        configure_joiner(node, dataset_tlvs, node.node_id in router_eligible_ids)

    workers = max(1, min(parallel, len(joiners)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, n) for n in joiners]
        for fut in as_completed(futs):
            fut.result()


def wait_attached(
    nodes: List[Node],
    timeout: float,
    sample_cb=None,
    parallel: int = 32,
) -> Dict[int, str]:
    """Wait until all nodes attach.

    timeout is wall-clock seconds in real-time mode, and simulated seconds in
    virtual-time mode.
    """
    states: Dict[int, str] = {}
    attached = {"leader", "router", "child"}
    use_vt = _TIME_ENGINE is not None and _TIME_ENGINE.virtual_time
    deadline_wall = time.monotonic() + timeout
    advanced = 0.0

    def _poll(node: Node) -> Tuple[int, str]:
        try:
            out = node.cmd("state", timeout=8)
            st = out[0].strip() if out else "unknown"
        except Exception:  # noqa: BLE001
            st = "unknown"
        return node.node_id, st

    def _poll_vt(pending_nodes: List[Node]) -> None:
        # Poll in chunks; a single 800-node state blast stalls VT/CLI.
        chunk = 64
        for start in range(0, len(pending_nodes), chunk):
            group = pending_nodes[start:start + chunk]
            try:
                results = cmd_many_vt([(n, "state") for n in group], timeout=45.0, chunk_size=chunk)
            except Exception as exc:  # noqa: BLE001
                log(f"batch state poll failed ({exc}); falling back serial for "
                    f"nodes {group[0].node_id}..{group[-1].node_id}")
                for node in group:
                    _, st = _poll(node)
                    states[node.node_id] = st
                continue
            for node in group:
                out = results.get(node.node_id) or []
                states[node.node_id] = out[0].strip() if out else "unknown"

    while True:
        if use_vt:
            if advanced >= timeout:
                break
        elif time.monotonic() >= deadline_wall:
            break

        pending_nodes = [n for n in nodes if states.get(n.node_id) not in attached]
        if not pending_nodes:
            return states

        if use_vt:
            _poll_vt(pending_nodes)
        else:
            workers = max(1, min(parallel, len(pending_nodes)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for node_id, st in pool.map(_poll, pending_nodes):
                    states[node_id] = st

        pending = [n.node_id for n in pending_nodes if states.get(n.node_id) not in attached]
        if sample_cb:
            sample_cb()

        done = len(nodes) - len(pending)
        log(f"attached {done}/{len(nodes)}  pending={pending[:10]}{'...' if len(pending) > 10 else ''}"
            + (f"  sim={_TIME_ENGINE.sim_seconds:.1f}s" if use_vt and _TIME_ENGINE else ""))
        if not pending:
            return states

        # Large networks: advance in smaller steps so we keep logging progress.
        step = 0.5 if use_vt and len(nodes) >= 200 else 1.0
        if use_vt and len(pending) > 200:
            log(f"advancing {step:.1f}s sim with {len(pending)} pending...")
        advance(step)
        advanced += step

    return states


def print_report(report: Dict) -> None:
    host = report["host"]
    states = report["states"]
    counts = Counter(states.values())
    print()
    print("=" * 64)
    print(" OpenThread Simulation Scale Report")
    print("=" * 64)
    print(f" nodes requested : {report['nodes_requested']}")
    print(f" nodes running   : {report['nodes_running']}")
    print(f" time mode       : {report.get('time_mode', 'unknown')}")
    print(f" join elapsed    : {report['join_seconds']:.1f}s wall"
          + (f" / {report.get('sim_seconds', 0):.1f}s sim" if report.get('time_mode') == 'virtual' else ""))
    print(f" states          : {dict(counts)}")
    print("-" * 64)
    print(" Host resources (after stabilize)")
    print(f"  total RSS      : {host['total_rss_mb']:.2f} MB  "
          f"(avg {host['rss_kb_avg']/1024:.2f} MB/node, "
          f"min {host['rss_kb_min']/1024:.2f}, max {host['rss_kb_max']/1024:.2f})")
    print(f"  total VSZ      : {host['total_vsz_mb']:.2f} MB")
    print(f"  total CPU      : {host['total_cpu_pct']:.1f}%")
    print(f"  total threads  : {host['total_threads']}")
    print(f"  children maxRSS: {host['self_maxrss_kb']/1024:.2f} MB (ru_maxrss)")
    if report.get("host_peak"):
        peak = report["host_peak"]
        print("-" * 64)
        print(" Peak during join")
        print(f"  peak RSS       : {peak['total_rss_mb']:.2f} MB")
        print(f"  peak CPU       : {peak['total_cpu_pct']:.1f}%")
    print("=" * 64)


def main() -> int:
    global _TIME_ENGINE

    args = parse_args()
    if args.nodes < 1:
        print("--nodes must be >= 1", file=sys.stderr)
        return 2

    ensure_fd_limit(args.nodes)

    ot_cli = args.ot_cli
    max_size = args.max_network_size or max(args.nodes, 128)
    virtual_time = args.virtual_time

    need_build = args.build or not ot_cli.is_file()
    if not need_build and virtual_time:
        vt_flag = cmake_virtual_time_enabled(ot_cli)
        if vt_flag is False:
            log("Binary is real-time; rebuilding with VIRTUAL_TIME=ON")
            need_build = True
        elif vt_flag is None:
            log("WARNING: cannot detect VIRTUAL_TIME from CMakeCache; "
                "if nodes hang, re-run with --build")

    if need_build:
        ot_cli = build_simulation(args.ot_root, max_size, virtual_time=virtual_time)
    else:
        # VirtualTime port = 9000 + PORT_OFFSET*(MAX_NODES+1) must use the
        # same MAX_NODES the binary compiled with (MAX_NETWORK_SIZE).
        compiled = cmake_max_network_size(ot_cli)
        if compiled is not None and compiled != max_size:
            if args.max_network_size and args.max_network_size != compiled:
                log(f"WARNING: --max-network-size={args.max_network_size} but binary "
                    f"was built with {compiled}; aligning VirtualTime to {compiled}")
            else:
                log(f"Aligning VirtualTime MAX_NODES to binary ({compiled})")
            max_size = compiled
        proc = subprocess.Popen(
            [str(ot_cli), str(args.nodes)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ot_cli.parent),
        )
        try:
            out, _ = proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            out = out or ""
        if "Invalid NodeId" in (out or ""):
            supported = probe_max_node_id(ot_cli)
            print(
                f"ERROR: {ot_cli} only supports {supported} nodes, need {args.nodes}.\n"
                f"Re-run with --build (will use OT_SIMULATION_MAX_NETWORK_SIZE={max_size}).",
                file=sys.stderr,
            )
            return 1
        log(f"Binary accepts node id {args.nodes}")

    # Clean simulation runtime state (flash files) and stale VT sockets.
    work_dir = ot_cli.parent.resolve()
    for tmp_dir in (Path("tmp"), work_dir / "tmp", Path.cwd() / "tmp"):
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
    for sock_path in list(Path.cwd().glob("vt.*.sock")) + list(work_dir.glob("vt.*.sock")):
        try:
            sock_path.unlink()
        except OSError:
            pass

    port_offset = args.port_offset
    if virtual_time:
        vt_max_off = max_safe_vt_port_offset(max_size)
        if port_offset < 0:
            # Stay in [0, vt_max_off] so radio ports remain < 18000.
            port_offset = (os.getpid() % (vt_max_off + 1)) if vt_max_off > 0 else 0
        elif port_offset > vt_max_off:
            log(f"PORT_OFFSET={port_offset} too large for MAX_NODES={max_size} "
                f"(max safe={vt_max_off}); clamping")
            port_offset = vt_max_off
    elif port_offset < 0:
        preferred = (os.getpid() % 40) + 1
        try:
            port_offset = pick_port_offset(ot_cli, preferred=preferred)
        except Exception as exc:  # noqa: BLE001
            port_offset = preferred
            log(f"PORT_OFFSET auto-detect failed ({exc}), using {port_offset}")
    log(f"PORT_OFFSET={port_offset}")

    env = os.environ.copy()
    env["PORT_OFFSET"] = str(port_offset)
    if virtual_time:
        env["OT_VT_USE_UNIX_SOCKET"] = "1"
    else:
        env.pop("OT_VT_USE_UNIX_SOCKET", None)

    _TIME_ENGINE = TimeEngine(
        virtual_time=virtual_time,
        max_nodes=max_size,
        port_offset=port_offset,
        work_dir=work_dir,
    )
    log(f"Time mode: {'virtual' if virtual_time else 'real'}")

    nodes: List[Node] = []
    samples: List[Dict] = []
    prev_cpu: Dict[int, Tuple[int, float]] = {}
    peak: Optional[Dict] = None

    def sample_cpu() -> None:
        nonlocal peak, prev_cpu
        snap, prev_cpu = host_snapshot(nodes, prev_cpu)
        slim = {k: v for k, v in snap.items() if k != "per_node"}
        samples.append(slim)
        if peak is None or slim["total_rss_kb"] > peak["total_rss_kb"]:
            peak = slim
        if args.verbose:
            log(f"rss={slim['total_rss_mb']:.1f}MB cpu={slim['total_cpu_pct']:.1f}%")

    t0 = time.monotonic()
    try:
        log(f"Using binary {ot_cli}")
        log("Starting leader (node 1)")
        nodes.append(spawn_node(ot_cli, 1, env))
        dataset_tlvs = configure_leader(nodes[0])

        for _ in range(60):
            out = nodes[0].cmd("state")
            st = out[0].strip() if out else "unknown"
            if args.verbose:
                log(f"leader state={st}")
            if st == "leader":
                break
            advance(1.0)
        else:
            raise RuntimeError(f"node 1 did not become leader, state={nodes[0].cmd('state')}")
        log("Leader ready")

        if args.nodes > 1:
            log(f"Starting nodes 2..{args.nodes}")
            for i in range(2, args.nodes + 1):
                nodes.append(spawn_node(ot_cli, i, env))
                if i % 50 == 0 or i == args.nodes:
                    log(f"spawned {i}/{args.nodes}")

            router_n = max(1, min(args.router_eligible, args.nodes))
            router_ids = set(range(1, router_n + 1))
            log(f"router-eligible nodes: 1..{router_n}  (others: child-only)")

            joiners = nodes[1:]
            batch = args.join_batch
            if batch <= 0:
                # Even under VT, starting hundreds of Thread stacks at once
                # creates an event storm that stalls advance()/attach.
                batch = 50 if args.nodes >= 200 else len(joiners)
            batch = max(1, min(batch, len(joiners)))
            # Real-time default 2s; VT settle is cheaper with a shorter sim step
            # because each advance() wakes every node (wall time grows with N).
            if virtual_time and abs(args.join_batch_delay - 2.0) < 1e-9:
                batch_delay = 0.5
            else:
                batch_delay = args.join_batch_delay

            for start in range(0, len(joiners), batch):
                chunk = joiners[start:start + batch]
                log(f"joining batch nodes {chunk[0].node_id}..{chunk[-1].node_id} "
                    f"({start + len(chunk)}/{len(joiners)})")
                if virtual_time:
                    # Interleave CLI across nodes; VirtualTime.go() stays single-threaded.
                    configure_joiners_vt(chunk, dataset_tlvs, router_ids)
                else:
                    configure_joiners_parallel(chunk, dataset_tlvs, router_ids, args.parallel)
                if start + batch < len(joiners) and batch_delay > 0:
                    log(f"batch settle advance {batch_delay:.1f}s sim...")
                    advance(batch_delay)

        for node in nodes:
            _, marker = read_proc_stats(node.proc.pid)
            prev_cpu[node.proc.pid] = marker

        last_sample = 0.0
        sample_every = args.sample_interval
        if args.nodes >= 500:
            sample_every = max(sample_every, 5.0)

        def on_wait_tick() -> None:
            nonlocal last_sample
            now = time.monotonic()
            if now - last_sample >= sample_every:
                sample_cpu()
                last_sample = now

        join_timeout = args.join_timeout if args.join_timeout > 0 else max(120.0, args.nodes * 0.5)
        wait_parallel = 1 if virtual_time else args.parallel
        log(f"Waiting for attach (timeout={join_timeout:.0f}s "
            f"{'sim' if virtual_time else 'wall'}, "
            f"{'batch-cli' if virtual_time else f'parallel={wait_parallel}'})")
        states = wait_attached(nodes, join_timeout, sample_cb=on_wait_tick, parallel=wait_parallel)
        join_seconds = time.monotonic() - t0

        attached = sum(1 for s in states.values() if s in ("leader", "router", "child"))
        log(f"Join finished: {attached}/{args.nodes} attached in {join_seconds:.1f}s wall "
            f"/ {_TIME_ENGINE.sim_seconds:.1f}s sim")

        if args.stabilize > 0:
            log(f"Stabilizing {args.stabilize:.0f}s ({'sim' if virtual_time else 'wall'}) ...")
            if virtual_time:
                # Advance in chunks so we can still sample host RSS/CPU.
                left = args.stabilize
                while left > 0:
                    step = min(sample_every, left)
                    advance(step)
                    sample_cpu()
                    left -= step
            else:
                end = time.monotonic() + args.stabilize
                while time.monotonic() < end:
                    sample_cpu()
                    time.sleep(args.sample_interval)

        if not virtual_time:
            time.sleep(1.0)
        else:
            advance(0.2)
        final_host, prev_cpu = host_snapshot(nodes, prev_cpu)

        report = {
            "nodes_requested": args.nodes,
            "nodes_running": len(nodes),
            "time_mode": "virtual" if virtual_time else "real",
            "join_seconds": round(join_seconds, 2),
            "sim_seconds": round(_TIME_ENGINE.sim_seconds, 2),
            "states": {str(k): v for k, v in states.items()},
            "state_counts": dict(Counter(states.values())),
            "host": {k: v for k, v in final_host.items() if k != "per_node"},
            "host_per_node": final_host["per_node"],
            "host_peak": peak,
            "host_samples": samples,
            "dataset": {
                "network_name": NETWORK_NAME,
                "network_key": NETWORK_KEY,
                "panid": f"{PANID:#04x}",
                "channel": CHANNEL,
                "active_tlvs": dataset_tlvs,
            },
            "port_offset": port_offset,
            "ot_cli": str(ot_cli),
        }
        print_report(report)

        json_out = args.json_out
        if json_out is None:
            json_out = SCRIPT_DIR / f"ot_scale_{args.nodes}.json"
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with json_out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log(f"Wrote {json_out}")

        if attached < args.nodes:
            log(f"WARNING: only {attached}/{args.nodes} nodes attached")
            rc = 1
        else:
            rc = 0

        if args.hold > 0:
            log(f"Holding network for {args.hold:.0f}s ({'sim' if virtual_time else 'wall'})...")
            if virtual_time:
                left = args.hold
                while left > 0:
                    advance(min(1.0, left))
                    left -= 1.0
            else:
                time.sleep(args.hold)

        return rc

    except KeyboardInterrupt:
        log("Interrupted")
        return 130
    finally:
        log("Stopping nodes...")
        for node in reversed(nodes):
            try:
                node.close()
            except Exception:  # noqa: BLE001
                pass
        if _TIME_ENGINE is not None:
            _TIME_ENGINE.stop()
            _TIME_ENGINE = None


if __name__ == "__main__":
    sys.exit(main())
