#!/usr/bin/env python3
"""Run OpenThread Nexus scale test and compare host resources with sim_scale.

Nexus runs all Thread nodes inside a single process (virtual time).
Simulation (`sim_scale_network.py`) runs one process per node (real time).

Examples:
  # Build nexus + run 800-node scale, compare with tools/ot_scale_800.json
  ./tools/nexus_scale_compare.py --nodes 800 --build

  # Use an already-built binary and an explicit sim report
  ./tools/nexus_scale_compare.py --nodes 800 \\
      --nexus-bin ./nexus_test/tests/nexus/nexus_scale_network \\
      --sim-json ./tools/ot_scale_800.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

OT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BUILD_DIR = OT_ROOT / "nexus_test"
RESULT_RE = re.compile(
    r"NEXUS_SCALE_RESULT\s+nodes=(?P<nodes>\d+)\s+attached=(?P<attached>\d+)\s+"
    r"leader=(?P<leader>\d+)\s+router=(?P<router>\d+)\s+child=(?P<child>\d+)\s+"
    r"detached=(?P<detached>\d+)\s+sim_ms=(?P<sim_ms>\d+)\s+wall_ms=(?P<wall_ms>\d+)"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", type=int, default=200, help="Number of Nexus nodes (default: 200)")
    p.add_argument("--router-eligible", type=int, default=32, help="Router-eligible node count (default: 32)")
    p.add_argument("--ot-root", type=Path, default=OT_ROOT, help="OpenThread source root")
    p.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR, help="Nexus CMake build dir")
    p.add_argument("--nexus-bin", type=Path, default=None, help="Path to nexus_scale_network binary")
    p.add_argument("--build", action="store_true", help="Build Nexus tests before running")
    p.add_argument(
        "--sim-json",
        type=Path,
        default=None,
        help="Simulation report JSON to compare (default: tools/ot_scale_<N>.json if present)",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write Nexus report JSON (default: tools/nexus_scale_<N>.json)",
    )
    p.add_argument("--sample-interval", type=float, default=0.5, help="Host sample interval while Nexus runs")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def ensure_fd_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = min(hard if hard != resource.RLIM_INFINITY else 65536, max(soft, 8192))
    if soft < want:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        except (ValueError, OSError):
            pass


def build_nexus(ot_root: Path, build_dir: Path) -> Path:
    script = ot_root / "tests" / "nexus" / "build.sh"
    if not script.is_file():
        raise FileNotFoundError(script)

    log(f"Building Nexus into {build_dir} ...")
    env = os.environ.copy()
    env["top_builddir"] = str(build_dir)
    subprocess.check_call(["bash", str(script)], cwd=str(ot_root), env=env)

    bin_path = build_dir / "tests" / "nexus" / "nexus_scale_network"
    if not bin_path.is_file():
        subprocess.check_call(["ninja", "nexus_scale_network"], cwd=str(build_dir))
    if not bin_path.is_file():
        raise FileNotFoundError(bin_path)
    return bin_path


def read_proc(pid: int, prev: Optional[Tuple[int, float]] = None) -> Tuple[Dict, Tuple[int, float]]:
    now = time.monotonic()
    rss_kb = vsz_kb = threads = 0
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
            elif line.startswith("VmSize:"):
                vsz_kb = int(line.split()[1])
            elif line.startswith("Threads:"):
                threads = int(line.split()[1])

    jiffies = 0
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        fields = stat.read_text().split()
        jiffies = int(fields[13]) + int(fields[14])

    cpu_pct = 0.0
    if prev is not None and jiffies:
        prev_j, prev_t = prev
        dt = max(now - prev_t, 1e-6)
        clk = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        cpu_pct = max(0.0, (jiffies - prev_j) * 100.0 / clk / dt)

    return (
        {
            "rss_kb": rss_kb,
            "rss_mb": round(rss_kb / 1024, 2),
            "vsz_kb": vsz_kb,
            "vsz_mb": round(vsz_kb / 1024, 2),
            "threads": threads,
            "cpu_pct": round(cpu_pct, 2),
        },
        (jiffies, now),
    )


def run_nexus_with_sampling(
    binary: Path,
    nodes: int,
    router_eligible: int,
    sample_interval: float,
    verbose: bool,
) -> Tuple[Dict, Dict]:
    proc = subprocess.Popen(
        [str(binary), str(nodes), str(router_eligible)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(binary.parent),
    )

    lines: List[str] = []
    result: Optional[Dict] = None
    samples: List[Dict] = []
    peak: Optional[Dict] = None
    prev: Optional[Tuple[int, float]] = None
    stop = threading.Event()

    def reader() -> None:
        nonlocal result
        assert proc.stdout is not None
        for line in proc.stdout:
            text = line.rstrip("\n")
            lines.append(text)
            if verbose:
                print(text, flush=True)
            m = RESULT_RE.search(text)
            if m:
                result = {k: int(v) for k, v in m.groupdict().items()}

    def sampler() -> None:
        nonlocal prev, peak
        while not stop.wait(sample_interval):
            if proc.poll() is not None:
                break
            try:
                snap, prev = read_proc(proc.pid, prev)
            except (FileNotFoundError, ProcessLookupError):
                break
            snap["timestamp"] = time.time()
            samples.append(snap)
            if peak is None or snap["rss_kb"] > peak["rss_kb"]:
                peak = dict(snap)
            if verbose:
                log(f"nexus pid={proc.pid} rss={snap['rss_mb']:.1f}MB cpu={snap['cpu_pct']:.1f}%")

    t_reader = threading.Thread(target=reader, daemon=True)
    t_sampler = threading.Thread(target=sampler, daemon=True)
    t_reader.start()

    # Nexus virtual-time runs can finish in <100ms; sample aggressively.
    sample_interval = min(sample_interval, 0.05)
    time.sleep(0.01)
    if proc.poll() is None:
        snap, prev = read_proc(proc.pid)
        snap["timestamp"] = time.time()
        samples.append(snap)
        peak = dict(snap)
        t_sampler.start()

    rc = proc.wait()
    stop.set()
    t_reader.join(timeout=2)
    t_sampler.join(timeout=2)

    if result is None:
        for text in lines:
            m = RESULT_RE.search(text)
            if m:
                result = {k: int(v) for k, v in m.groupdict().items()}
                break

    if result is None:
        raise RuntimeError(
            f"nexus_scale_network exited {rc} without NEXUS_SCALE_RESULT.\n"
            f"Last output:\n" + "\n".join(lines[-40:])
        )
    if rc != 0:
        raise RuntimeError(f"nexus_scale_network failed rc={rc}, result={result}")

    maxrss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    final = samples[-1] if samples else {
        "rss_kb": maxrss_kb,
        "rss_mb": round(maxrss_kb / 1024, 2),
        "vsz_kb": 0,
        "vsz_mb": 0.0,
        "threads": 0,
        "cpu_pct": 0.0,
    }
    peak = peak or dict(final)
    # Short runs often miss /proc samples; ru_maxrss is the reliable peak.
    if maxrss_kb > peak.get("rss_kb", 0):
        peak = dict(peak)
        peak["rss_kb"] = maxrss_kb
        peak["rss_mb"] = round(maxrss_kb / 1024, 2)

    host = {
        "final": final,
        "peak": peak,
        "samples": samples,
        "children_maxrss_kb": maxrss_kb,
        "children_maxrss_mb": round(maxrss_kb / 1024, 2),
        "process_count": 1,
    }
    return result, host


def ratio(a: float, b: float) -> str:
    if b == 0:
        return "n/a"
    return f"{a / b:.2f}x"


def print_compare(nexus: Dict, sim: Optional[Dict], host: Dict) -> None:
    n_host = host["peak"]
    print()
    print("=" * 72)
    print(" Nexus vs Simulation Host Resource Compare")
    print("=" * 72)
    print(f" Nexus nodes/attached : {nexus['nodes']} / {nexus['attached']}")
    print(
        f" Nexus roles          : leader={nexus['leader']} router={nexus['router']} "
        f"child={nexus['child']} detached={nexus['detached']}"
    )
    print(f" Nexus wall / sim     : {nexus['wall_ms']} ms / {nexus['sim_ms']} ms")
    print("-" * 72)
    print(" Nexus host (single process)")
    print(f"  peak RSS            : {n_host['rss_mb']:.2f} MB")
    print(f"  peak VSZ            : {n_host.get('vsz_mb', 0):.2f} MB")
    print(f"  peak CPU            : {n_host['cpu_pct']:.1f}%")
    print(f"  threads             : {n_host.get('threads', 0)}")
    print("  processes           : 1")
    print(f"  children maxRSS     : {host['children_maxrss_mb']:.2f} MB")

    if sim:
        s_host = sim.get("host", {})
        s_peak = sim.get("host_peak") or s_host
        sim_rss = float(s_peak.get("total_rss_mb", s_host.get("total_rss_mb", 0)))
        sim_cpu = float(s_peak.get("total_cpu_pct", s_host.get("total_cpu_pct", 0)))
        sim_procs = int(sim.get("nodes_running", sim.get("nodes_requested", 0)))
        print("-" * 72)
        print(" Simulation host (one process per node)")
        print(f"  nodes               : {sim.get('nodes_requested')} "
              f"(states: {sim.get('state_counts')})")
        print(f"  join elapsed        : {sim.get('join_seconds')} s")
        print(f"  peak/total RSS      : {sim_rss:.2f} MB")
        print(f"  peak/total CPU      : {sim_cpu:.1f}%")
        print(f"  processes           : {sim_procs}")
        print("-" * 72)
        print(" Comparison (simulation / nexus)")
        print(
            f"  RSS ratio           : {ratio(sim_rss, float(n_host['rss_mb']))}  "
            f"(sim {sim_rss:.2f} MB / nexus {n_host['rss_mb']:.2f} MB)"
        )
        print(
            f"  CPU ratio           : {ratio(sim_cpu, float(n_host['cpu_pct']))}  "
            f"(sim {sim_cpu:.1f}% / nexus {n_host['cpu_pct']:.1f}%)"
        )
        print(f"  process ratio       : {ratio(float(sim_procs), 1.0)}  ({sim_procs} / 1)")
        if nexus["nodes"] and sim_procs:
            print(f"  RSS per node sim    : {sim_rss / max(sim_procs, 1):.3f} MB")
            print(f"  RSS per node nexus  : {float(n_host['rss_mb']) / max(nexus['nodes'], 1):.3f} MB")
    else:
        print("-" * 72)
        print(" No simulation JSON found; skipped comparison.")
        print(" Run sim_scale_network.py first, or pass --sim-json.")
    print("=" * 72)


def main() -> int:
    args = parse_args()
    if args.nodes < 1:
        print("--nodes must be >= 1", file=sys.stderr)
        return 2

    ensure_fd_limit()
    nexus_bin = (args.nexus_bin or (args.build_dir / "tests" / "nexus" / "nexus_scale_network")).resolve()

    if args.build or not nexus_bin.is_file():
        nexus_bin = build_nexus(args.ot_root, args.build_dir)

    sim_json = args.sim_json
    if sim_json is None:
        candidate = SCRIPT_DIR / f"ot_scale_{args.nodes}.json"
        if candidate.is_file():
            sim_json = candidate

    json_out = args.json_out or (SCRIPT_DIR / f"nexus_scale_{args.nodes}.json")

    log(f"Running {nexus_bin} nodes={args.nodes} router_eligible={args.router_eligible}")
    result, host = run_nexus_with_sampling(
        nexus_bin, args.nodes, args.router_eligible, args.sample_interval, args.verbose
    )

    report = {
        "framework": "nexus",
        "nodes_requested": args.nodes,
        "router_eligible": args.router_eligible,
        "result": result,
        "host": {
            "peak_rss_mb": host["peak"]["rss_mb"],
            "peak_vsz_mb": host["peak"].get("vsz_mb", 0),
            "peak_cpu_pct": host["peak"]["cpu_pct"],
            "final_rss_mb": host["final"]["rss_mb"],
            "final_cpu_pct": host["final"]["cpu_pct"],
            "threads": host["final"].get("threads", 0),
            "process_count": 1,
            "children_maxrss_mb": host["children_maxrss_mb"],
        },
        "host_peak": host["peak"],
        "host_final": host["final"],
        "host_samples": host["samples"],
        "nexus_bin": str(nexus_bin),
        "sim_json": str(sim_json) if sim_json else None,
    }

    sim_report = None
    if sim_json and sim_json.is_file():
        with sim_json.open(encoding="utf-8") as f:
            sim_report = json.load(f)
        sim_rss = float((sim_report.get("host_peak") or sim_report.get("host", {})).get("total_rss_mb") or 0)
        nexus_rss = float(report["host"]["peak_rss_mb"] or 0)
        report["comparison"] = {
            "sim_total_rss_mb": sim_rss,
            "sim_total_cpu_pct": float(
                (sim_report.get("host_peak") or sim_report.get("host", {})).get("total_cpu_pct") or 0
            ),
            "sim_processes": sim_report.get("nodes_running"),
            "rss_ratio_sim_over_nexus": round(sim_rss / nexus_rss, 3) if nexus_rss else None,
        }

    print_compare(result, sim_report, host)

    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"Wrote {json_out}")
    return 0 if result["attached"] == result["nodes"] else 1


if __name__ == "__main__":
    sys.exit(main())
