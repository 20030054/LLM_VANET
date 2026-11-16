//
// Vehicle application: reports its state to the RSU, generates tasks, asks
// the RSU where to run each task, carries out the decision and executes
// tasks that other vehicles offload to it.
//
#pragma once

#include "kvrsu/KvBaseApp.h"

namespace kvrsu {

class VehicleApp : public KvBaseApp {
public:
    ~VehicleApp() override;
    void initialize(int stage) override;
    void finish() override;

protected:
    enum SelfKind { KIND_TASK = KIND_LAST_BASE, KIND_STATE, KIND_DEADLINE, KIND_DECISION_TIMEOUT };
    enum Mode { MODE_POLICY = 0, MODE_FALLBACK = 1, MODE_TIMEOUT = 2, MODE_NOCOVERAGE = 3 };

    struct Task {
        double cycles = 0;
        long bytes = 0;
        simtime_t created, deadline;
        double tDecision = -1; // seconds from creation to decision, -1 = none
        int mode = -1;
        long target = TARGET_LOCAL;
        bool decided = false;
        cMessage* deadlineMsg = nullptr;
        cMessage* timeoutMsg = nullptr;
    };

    static long nextTaskId;
    std::map<long, Task> tasks;

    long rsuAddr = -1;
    simtime_t rsuSeen = -1;
    double lastHeading = 0;
    cMessage* taskTimer = nullptr;
    cMessage* stateTimer = nullptr;
    long nGenerated = 0, nOk = 0, nMiss = 0;

    void fillState(KvMessage* m);
    void generateTask();
    void startLocal(long id);
    void closeTask(long id, const char* status);
    void logTask(long id, const Task& t, const char* status);

    void onKv(KvMessage* msg) override;
    void onLocalTaskDone(long taskId) override;
    void onOtherSelfMsg(cMessage* msg) override;
};

} // namespace kvrsu
