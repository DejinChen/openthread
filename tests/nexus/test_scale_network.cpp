/*
 *  Copyright (c) 2026, The OpenThread Authors.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *  1. Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *  3. Neither the name of the copyright holder nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 *   Nexus scale network test used for host-resource comparison against
 *   multi-process OT simulation (`tools/sim_scale_network.py`).
 *
 * Usage:
 *   nexus_scale_network [num_nodes] [router_eligible]
 *
 * Defaults: num_nodes=200, router_eligible=32
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "platform/nexus_core.hpp"
#include "platform/nexus_node.hpp"

namespace ot {
namespace Nexus {

static constexpr uint16_t kNumberOfRoles = (Mle::kRoleLeader + 1);
typedef uint16_t          RoleStats[kNumberOfRoles];

static uint64_t WallTimeMs(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (static_cast<uint64_t>(ts.tv_sec) * 1000u) + static_cast<uint64_t>(ts.tv_nsec / 1000000u);
}

static void CalculateRoleStats(Core &aNexus, RoleStats &aRoleStats)
{
    ClearAllBytes(aRoleStats);

    for (Node &node : aNexus.GetNodes())
    {
        aRoleStats[node.Get<Mle::Mle>().GetRole()]++;
    }
}

static uint16_t CountAttached(const RoleStats &aRoleStats)
{
    return static_cast<uint16_t>(aRoleStats[Mle::kRoleLeader] + aRoleStats[Mle::kRoleRouter] +
                                 aRoleStats[Mle::kRoleChild]);
}

static bool StartedNodesAttached(const RoleStats &aRoleStats, uint16_t aStarted)
{
    return (aRoleStats[Mle::kRoleLeader] == 1) && (aRoleStats[Mle::kRoleDetached] == 0) &&
           (CountAttached(aRoleStats) == aStarted);
}

static void WaitStartedAttached(Core &aNexus, uint16_t aStarted, uint32_t aMaxWaitMs, uint32_t aStepMs)
{
    RoleStats roleStats;

    for (uint32_t step = 0; step < aMaxWaitMs / aStepMs; step++)
    {
        CalculateRoleStats(aNexus, roleStats);

        if ((step % 20) == 0)
        {
            Log("started=%u roles leader=%u router=%u child=%u detached=%u disabled=%u", aStarted,
                roleStats[Mle::kRoleLeader], roleStats[Mle::kRoleRouter], roleStats[Mle::kRoleChild],
                roleStats[Mle::kRoleDetached], roleStats[Mle::kRoleDisabled]);
        }

        if (StartedNodesAttached(roleStats, aStarted))
        {
            return;
        }

        aNexus.AdvanceTime(aStepMs);
    }

    CalculateRoleStats(aNexus, roleStats);
    Log("TIMEOUT started=%u roles leader=%u router=%u child=%u detached=%u disabled=%u", aStarted,
        roleStats[Mle::kRoleLeader], roleStats[Mle::kRoleRouter], roleStats[Mle::kRoleChild],
        roleStats[Mle::kRoleDetached], roleStats[Mle::kRoleDisabled]);
    VerifyOrQuit(StartedNodesAttached(roleStats, aStarted));
}

void TestScaleNetwork(uint16_t aNumNodes, uint16_t aRouterEligible)
{
    static constexpr uint32_t kMaxWaitTimeMs = 30 * Time::kOneMinuteInMsec;
    static constexpr uint32_t kStepMs        = 500;
    static constexpr uint32_t kStabilizeMs   = 10 * Time::kOneSecondInMsec;
    static constexpr uint16_t kJoinBatch     = 50;

    Core      nexus;
    Node     *leader;
    Node    **nodes;
    RoleStats roleStats;
    uint64_t  wallStart;
    uint64_t  wallMs;
    uint16_t  attached;

    VerifyOrQuit(aNumNodes >= 1);
    VerifyOrQuit(aRouterEligible >= 1);
    VerifyOrQuit(aRouterEligible <= aNumNodes);

    Log("Nexus scale: nodes=%u router_eligible=%u", aNumNodes, aRouterEligible);

    nodes = static_cast<Node **>(malloc(sizeof(Node *) * aNumNodes));
    VerifyOrQuit(nodes != nullptr);

    wallStart = WallTimeMs();

    for (uint16_t i = 0; i < aNumNodes; i++)
    {
        nodes[i] = &nexus.CreateNode();
    }

    nexus.AdvanceTime(0);
    SuccessOrQuit(Instance::SetGlobalLogLevel(kLogLevelCrit));

    leader = nodes[0];
    leader->Form();
    nexus.AdvanceTime(15 * Time::kOneSecondInMsec);
    VerifyOrQuit(leader->Get<Mle::Mle>().IsLeader());
    Log("Leader ready");

    for (uint16_t i = 1; i < aRouterEligible; i++)
    {
        nodes[i]->Join(*leader, Node::kAsFtd);
        nexus.AdvanceTime(Time::kOneSecondInMsec);
    }

    WaitStartedAttached(nexus, aRouterEligible, 5 * Time::kOneMinuteInMsec, kStepMs);
    CalculateRoleStats(nexus, roleStats);
    Log("Routers formed: router=%u", roleStats[Mle::kRoleRouter]);

    for (uint16_t i = aRouterEligible; i < aNumNodes; i++)
    {
        nodes[i]->Join(*leader, Node::kAsFed);

        if (((i - aRouterEligible + 1) % kJoinBatch) == 0 || (i + 1) == aNumNodes)
        {
            Log("Joined FEDs through node index %u", i);
            nexus.AdvanceTime(2 * Time::kOneSecondInMsec);
        }
    }

    WaitStartedAttached(nexus, aNumNodes, kMaxWaitTimeMs, kStepMs);

    Log("Stabilizing %u sec (sim time)", kStabilizeMs / Time::kOneSecondInMsec);
    nexus.AdvanceTime(kStabilizeMs);

    CalculateRoleStats(nexus, roleStats);
    wallMs   = WallTimeMs() - wallStart;
    attached = CountAttached(roleStats);

    printf("NEXUS_SCALE_RESULT nodes=%u attached=%u leader=%u router=%u child=%u detached=%u "
           "sim_ms=%u wall_ms=%llu\n",
           aNumNodes, attached, roleStats[Mle::kRoleLeader], roleStats[Mle::kRoleRouter], roleStats[Mle::kRoleChild],
           roleStats[Mle::kRoleDetached], nexus.GetNow().GetValue(), static_cast<unsigned long long>(wallMs));

    free(nodes);
    VerifyOrQuit(attached == aNumNodes);
}

} // namespace Nexus
} // namespace ot

int main(int argc, char *argv[])
{
    uint16_t numNodes       = 200;
    uint16_t routerEligible = 32;

    if (argc >= 2)
    {
        numNodes = static_cast<uint16_t>(atoi(argv[1]));
    }

    if (argc >= 3)
    {
        routerEligible = static_cast<uint16_t>(atoi(argv[2]));
    }

    ot::Nexus::TestScaleNetwork(numNodes, routerEligible);
    printf("All tests passed\n");
    return 0;
}
