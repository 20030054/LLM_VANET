//
// Common base of the vehicle and RSU applications: message helpers, the CPU
// queue used to execute tasks, reassembly of task-input fragments, and the
// per-task CSV log.
//
#pragma once

#include <fstream>
#include <map>
#include <utility>

#include "veins/modules/application/ieee80211p/DemoBaseApplLayer.h"
#include "kvrsu/KvMessage_m.h"

namespace kvrsu {

const long TARGET_LOCAL = -1;
const long TARGET_EDGE = -2;

class KvBaseApp : public veins::DemoBaseApplLayer {
public:
    void initialize(int stage) override;

protected:
    enum SelfKind { KIND_EXEC_DONE = 100, KIND_LAST_BASE };

    double cpuHz = 1e9;          // cycles per second of this node
    int fragBytes = 1300;        // payload bytes per data fragment
    simtime_t busyUntil;         // time at which the CPU queue becomes empty

    struct Rx {
        int got = 0;
        double cycles = 0;
        simtime_t deadline;
    };
    std::map<std::pair<long, long>, Rx> rx; // (source, task) -> reassembly state

    // one CSV with a row per finished task, shared by all nodes of the run
    static std::ofstream taskLog;

    double backlog() const;
    simtime_t enqueueCpu(double cycles);
    void scheduleExecDone(simtime_t at, long src, long taskId);

    KvMessage* makeMsg(int type, long dst, int payloadBytes);
    void sendTaskData(long dst, long taskId, double cycles, long bytes, simtime_t created, simtime_t deadline);

    void onWSM(veins::BaseFrame1609_4* frame) override;
    void handleSelfMsg(cMessage* msg) override;

    // hooks for the two applications
    virtual void onKv(KvMessage* msg) = 0;
    virtual void onLocalTaskDone(long taskId) {}
    virtual void onOtherSelfMsg(cMessage* msg);
};

} // namespace kvrsu
