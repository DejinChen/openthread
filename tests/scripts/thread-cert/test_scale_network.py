#!/usr/bin/env python3
#
#  Copyright (c) 2026, The OpenThread Authors.
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  1. Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     distribution and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the
#     names of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#

"""Large-scale Thread network formation on the simulation platform.

Default size (32) fits a stock simulation build
(`OT_SIMULATION_MAX_NETWORK_SIZE=33`). For larger nets, rebuild and set env:

  ./script/cmake-build simulation \\
      -DOT_SIMULATION_VIRTUAL_TIME=ON \\
      -DOT_SIMULATION_MAX_NETWORK_SIZE=1024

  OT_SCALE_NUM_NODES=200 OT_SIMULATION_MAX_NETWORK_SIZE=1024 \\
      ./script/test cert tests/scripts/thread-cert/test_scale_network.py

Optional env:
  OT_SCALE_NUM_NODES         total nodes (default: 32)
  OT_SCALE_ROUTER_ELIGIBLE   router-eligible count (default: min(32, NUM_NODES))
  OT_SCALE_JOIN_BATCH        nodes started per wave (default: 16)
  OT_SCALE_JOIN_BATCH_DELAY  sim seconds between waves (default: 2)
  OT_SCALE_ATTACH_TIMEOUT    max sim seconds to wait for full attach (default: max(120, NUM_NODES))
  OT_SIMULATION_MAX_NETWORK_SIZE  must match binary; also sets VirtualTime.MAX_NODES
  OT_SCALE_JSON_OUT          optional path to write resource report JSON
"""

from __future__ import annotations

import json
import logging
import os
import resource
import statistics
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Must be set before importing config/simulator (they read env at import time).
# Stock `script/test cert` builds use VIRTUAL_TIME=ON; without an external
# VirtualTime engine, ot-cli-ftd exits immediately (pexpect EOF on '> ').
os.environ.setdefault('VIRTUAL_TIME', '1')
if os.environ.get('VIRTUAL_TIME') == '1':
    os.environ.setdefault('OT_VT_USE_UNIX_SOCKET', '1')

import config
import simulator
import sniffer_transport
import thread_cert

NUM_NODES = int(os.getenv('OT_SCALE_NUM_NODES', '32'))
ROUTER_ELIGIBLE = int(os.getenv('OT_SCALE_ROUTER_ELIGIBLE', str(min(32, NUM_NODES))))
JOIN_BATCH = int(os.getenv('OT_SCALE_JOIN_BATCH', '100'))
JOIN_BATCH_DELAY = float(os.getenv('OT_SCALE_JOIN_BATCH_DELAY', '0.2'))
ATTACH_TIMEOUT = float(os.getenv('OT_SCALE_ATTACH_TIMEOUT', str(max(120, NUM_NODES))))
# Default stays at stock 33 unless the user explicitly overrides.
MAX_NETWORK_SIZE = int(os.getenv('OT_SIMULATION_MAX_NETWORK_SIZE', '33'))
JSON_OUT = os.getenv('OT_SCALE_JSON_OUT', '')

# Keep VirtualTime / sniffer port math aligned with the simulation binary.
simulator.VirtualTime.MAX_NODES = max(simulator.VirtualTime.MAX_NODES, MAX_NETWORK_SIZE)
simulator.VirtualTime.USE_UNIX_SOCKET = os.environ.get('OT_VT_USE_UNIX_SOCKET', '0') == '1'
sniffer_transport.SnifferSocketTransport.MAX_NETWORK_SIZE = max(
    sniffer_transport.SnifferSocketTransport.MAX_NETWORK_SIZE, MAX_NETWORK_SIZE)

if not config.VIRTUAL_TIME:
    raise SystemExit('test_scale_network.py requires VIRTUAL_TIME=1 (VT-built ot-cli-ftd needs '
                     'an external VirtualTime engine). Run via: ./script/test cert '
                     'tests/scripts/thread-cert/test_scale_network.py')

LEADER = 1
ATTACHED = {'leader', 'router', 'child'}


def _node_pid(node) -> Optional[int]:
    pexpect_obj = getattr(node, 'pexpect', None)
    proc = getattr(pexpect_obj, 'proc', None) if pexpect_obj is not None else None
    if proc is None:
        return None
    return proc.pid


def _read_jiffies(pid: int) -> int:
    fields = Path(f'/proc/{pid}/stat').read_text().split()
    return int(fields[13]) + int(fields[14])


def _read_proc(pid: int, prev: Optional[Tuple[int, float]] = None) -> Tuple[dict, Tuple[int, float]]:
    now = time.monotonic()
    out = {'rss_kb': 0, 'vsz_kb': 0, 'threads': 0, 'cpu_pct': 0.0}
    status = Path(f'/proc/{pid}/status')
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                out['rss_kb'] = int(line.split()[1])
            elif line.startswith('VmSize:'):
                out['vsz_kb'] = int(line.split()[1])
            elif line.startswith('Threads:'):
                out['threads'] = int(line.split()[1])
    try:
        jiffies = _read_jiffies(pid)
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        jiffies = 0
    if prev is not None and jiffies:
        prev_j, prev_t = prev
        dt = max(now - prev_t, 1e-6)
        clk = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
        out['cpu_pct'] = max(0.0, (jiffies - prev_j) * 100.0 / clk / dt)
    return out, (jiffies, now)


def snapshot_nodes(nodes: dict, prev_cpu: Dict[int, Tuple[int, float]]) -> Tuple[dict, Dict[int, Tuple[int, float]]]:
    """Sum RSS/VSZ/CPU across all simulation node processes (+ this Python controller)."""
    per_node = []
    total_rss = total_vsz = total_threads = 0
    total_cpu = 0.0
    new_prev: Dict[int, Tuple[int, float]] = {}

    for node_id, node in sorted(nodes.items()):
        pid = _node_pid(node)
        if pid is None:
            continue
        st, marker = _read_proc(pid, prev_cpu.get(pid))
        new_prev[pid] = marker
        total_rss += st['rss_kb']
        total_vsz += st['vsz_kb']
        total_threads += st['threads']
        total_cpu += st['cpu_pct']
        per_node.append({'node_id': node_id, 'pid': pid, **st})

    # Include the test/VirtualTime controller process itself.
    self_pid = os.getpid()
    self_st, self_marker = _read_proc(self_pid, prev_cpu.get(self_pid))
    new_prev[self_pid] = self_marker

    rss_list = [x['rss_kb'] for x in per_node]
    snap = {
        'timestamp': time.time(),
        'node_count': len(per_node),
        'nodes_rss_kb': total_rss,
        'nodes_rss_mb': round(total_rss / 1024, 2),
        'nodes_vsz_kb': total_vsz,
        'nodes_vsz_mb': round(total_vsz / 1024, 2),
        'nodes_cpu_pct': round(total_cpu, 2),
        'nodes_threads': total_threads,
        'rss_kb_avg': round(statistics.mean(rss_list), 1) if rss_list else 0,
        'rss_kb_min': min(rss_list) if rss_list else 0,
        'rss_kb_max': max(rss_list) if rss_list else 0,
        'controller_rss_kb': self_st['rss_kb'],
        'controller_vsz_kb': self_st['vsz_kb'],
        'controller_cpu_pct': round(self_st['cpu_pct'], 2),
        'total_rss_kb': total_rss + self_st['rss_kb'],
        'total_rss_mb': round((total_rss + self_st['rss_kb']) / 1024, 2),
        'children_maxrss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    }
    return snap, new_prev


def _build_topology(num_nodes: int, router_eligible: int) -> dict:
    topology = {}
    for node_id in range(1, num_nodes + 1):
        params = {
            'mode': 'rdn',
            'name': f'NODE_{node_id}',
        }
        # Do not set max_children above OPENTHREAD_CONFIG_MLE_MAX_CHILDREN
        # (stock default is 10); invalid `childmax` returns Error and times out.
        if node_id <= router_eligible:
            params['router_eligible'] = True
        else:
            params['router_eligible'] = False
        topology[node_id] = params
    return topology


class TestScaleNetwork(thread_cert.TestCase):
    """Form a Thread network with many simulation nodes and wait until all attach."""

    USE_MESSAGE_FACTORY = False
    SUPPORT_NCP = False
    PACKET_VERIFICATION = config.PACKET_VERIFICATION_NONE

    TOPOLOGY = _build_topology(NUM_NODES, ROUTER_ELIGIBLE)

    def test(self):
        if NUM_NODES < 2:
            self.skipTest('OT_SCALE_NUM_NODES must be >= 2')
        if ROUTER_ELIGIBLE < 1 or ROUTER_ELIGIBLE > NUM_NODES:
            self.skipTest('OT_SCALE_ROUTER_ELIGIBLE out of range')
        if NUM_NODES > MAX_NETWORK_SIZE:
            self.skipTest(f'NUM_NODES={NUM_NODES} > MAX_NETWORK_SIZE={MAX_NETWORK_SIZE}; '
                          f'rebuild simulation with -DOT_SIMULATION_MAX_NETWORK_SIZE>={NUM_NODES} '
                          f'and export OT_SIMULATION_MAX_NETWORK_SIZE')

        logging.info('scale network: nodes=%d router_eligible=%d batch=%d delay=%.1fs timeout=%.0fs', NUM_NODES,
                     ROUTER_ELIGIBLE, JOIN_BATCH, JOIN_BATCH_DELAY, ATTACH_TIMEOUT)

        self._prev_cpu: Dict[int, Tuple[int, float]] = {}
        self._samples: List[dict] = []
        self._peak: Optional[dict] = None

        wall0 = time.monotonic()
        leader = self.nodes[LEADER]
        leader.start()
        self.simulator.go(config.LEADER_STARTUP_DELAY)
        self.assertEqual(leader.get_state(), 'leader')
        self._sample_resources('leader_ready')

        joiners = [self.nodes[i] for i in range(2, NUM_NODES + 1)]
        batch = max(1, min(JOIN_BATCH, len(joiners)))
        for start in range(0, len(joiners), batch):
            chunk = joiners[start:start + batch]
            for node in chunk:
                node.start()
            logging.info('started nodes %d..%d (%d/%d)', chunk[0].nodeid, chunk[-1].nodeid, start + len(chunk),
                         len(joiners))
            self._sample_resources(f'after_start_{start + len(chunk)}')
            if start + batch < len(joiners) and JOIN_BATCH_DELAY > 0:
                self.simulator.go(JOIN_BATCH_DELAY)

        attached = self._wait_all_attached(ATTACH_TIMEOUT)
        wall = time.monotonic() - wall0
        final = self._sample_resources('attached')

        counts = {}
        for node_id, state in attached.items():
            counts[state] = counts.get(state, 0) + 1

        report = {
            'nodes': NUM_NODES,
            'router_eligible': ROUTER_ELIGIBLE,
            'wall_seconds': round(wall, 2),
            'sim_seconds': round(self.simulator.now(), 2),
            'states': counts,
            'host_final': {k: v for k, v in final.items() if k != 'timestamp'},
            'host_peak': {k: v for k, v in (self._peak or {}).items() if k != 'timestamp'},
            'host_samples': self._samples,
        }
        self._log_resource_report(report)
        if JSON_OUT:
            path = Path(JSON_OUT)
            path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            logging.info('wrote resource report %s', path)

        self.assertEqual(counts.get('leader', 0), 1)
        self.assertEqual(sum(1 for s in attached.values() if s in ATTACHED), NUM_NODES)
        self.assertEqual(len(attached), NUM_NODES)

    def _sample_resources(self, label: str) -> dict:
        snap, self._prev_cpu = snapshot_nodes(self.nodes, self._prev_cpu)
        snap['label'] = label
        snap['sim_seconds'] = round(self.simulator.now(), 2)
        self._samples.append(snap)
        if self._peak is None or snap['total_rss_kb'] > self._peak['total_rss_kb']:
            self._peak = snap
        logging.info(
            'resources[%s]: nodes_rss=%.1fMB ctrl=%.1fMB total=%.1fMB cpu=%.1f%% threads=%d sim=%.1fs',
            label,
            snap['nodes_rss_mb'],
            snap['controller_rss_kb'] / 1024,
            snap['total_rss_mb'],
            snap['nodes_cpu_pct'],
            snap['nodes_threads'],
            snap['sim_seconds'],
        )
        return snap

    def _log_resource_report(self, report: dict) -> None:
        host = report['host_final']
        peak = report.get('host_peak') or {}
        logging.info('scale network done: wall=%.1fs sim=%.1fs states=%s', report['wall_seconds'],
                     report['sim_seconds'], report['states'])
        logging.info(
            'host final: nodes_rss=%.2fMB (avg %.2fMB/node) ctrl=%.2fMB total=%.2fMB '
            'cpu=%.1f%% threads=%d children_maxRSS=%.2fMB', host['nodes_rss_mb'], host['rss_kb_avg'] / 1024,
            host['controller_rss_kb'] / 1024, host['total_rss_mb'], host['nodes_cpu_pct'], host['nodes_threads'],
            host['children_maxrss_kb'] / 1024)
        if peak:
            logging.info('host peak: total_rss=%.2fMB nodes_rss=%.2fMB cpu=%.1f%% @ %s', peak['total_rss_mb'],
                         peak['nodes_rss_mb'], peak['nodes_cpu_pct'], peak.get('label', '?'))

    def _wait_all_attached(self, timeout: float) -> dict:
        states = {}
        deadline = self.simulator.now() + timeout
        step = 1.0

        while self.simulator.now() < deadline:
            pending = []
            for node_id, node in self.nodes.items():
                state = node.get_state()
                states[node_id] = state
                if state not in ATTACHED:
                    pending.append(node_id)

            done = NUM_NODES - len(pending)
            if (done == NUM_NODES) or (int(self.simulator.now()) % 5 == 0):
                logging.info('attached %d/%d pending=%s sim=%.1fs', done, NUM_NODES, pending[:10],
                             self.simulator.now())
                self._sample_resources(f'attach_{done}')

            if not pending:
                return states

            self.simulator.go(step)

        # Final sample for assertion message.
        for node_id, node in self.nodes.items():
            states[node_id] = node.get_state()
        pending = [i for i, s in states.items() if s not in ATTACHED]
        self.fail(f'attach timeout after {timeout:.0f}s sim; attached={NUM_NODES - len(pending)}/{NUM_NODES} '
                  f'pending={pending[:20]}')


if __name__ == '__main__':
    unittest.main()
