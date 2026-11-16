//
// RSU application: keeps the resource table, collects requests into decision
// windows, asks the decision server (engine/kvrsu/server.py) and delivers each
// decision after the time the server needed to compute it. The RSU also acts
// as the edge server that executes tasks offloaded to it.
//
#pragma once

#include <vector>

#include "kvrsu/EngineClient.h"
#include "kvrsu/KvBaseApp.h"

namespace kvrsu {

class RsuApp : public KvBaseApp {
public:
    ~RsuApp() override;
    void initialize(int stage) override;
    void finish() override;

protected:
    enum SelfKind { KIND_WINDOW = KIND_LAST_BASE, KIND_BEACON, KIND_SEND };

    struct Entry {
        std::string sumoId;
        double x, y, speed, heading, cpuHz, backlog;
        simtime_t seen;
    };
    struct Req {
        long id, addr;
        double cycles;
        long bytes;
        simtime_t created, deadline;
    };

    std::map<long, Entry> table;
    std::vector<Req> pending;
    EngineClient engine;
    std::ofstream winLog;
    cMessage* windowTimer = nullptr;
    cMessage* beaconTimer = nullptr;
    simtime_t engineFreeAt;
    long windowIdx = 0;
    long nRequests = 0, nPolicy = 0, nFallback = 0, nExpired = 0;
    double posX = 0, posY = 0;

    void updateEntry(KvMessage* m);
    void runWindow();

    void onKv(KvMessage* msg) override;
    void onOtherSelfMsg(cMessage* msg) override;
};

} // namespace kvrsu
