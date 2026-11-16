#include "kvrsu/VehicleApp.h"

#include <cmath>

using namespace veins;

namespace kvrsu {

Define_Module(kvrsu::VehicleApp);

long VehicleApp::nextTaskId = 1;

VehicleApp::~VehicleApp()
{
    cancelAndDelete(taskTimer);
    cancelAndDelete(stateTimer);
    for (auto& kv : tasks) {
        cancelAndDelete(kv.second.deadlineMsg);
        cancelAndDelete(kv.second.timeoutMsg);
    }
}

void VehicleApp::initialize(int stage)
{
    KvBaseApp::initialize(stage);
    if (stage == 0) {
        taskTimer = new cMessage("task", KIND_TASK);
        stateTimer = new cMessage("state", KIND_STATE);
        simtime_t firstTask = std::max(simTime(), SimTime(par("warmup").doubleValue())) + exponential(1.0 / par("taskRate").doubleValue());
        scheduleAt(firstTask, taskTimer);
        scheduleAt(simTime() + uniform(0, par("stateInterval").doubleValue()), stateTimer);
    }
}

void VehicleApp::fillState(KvMessage* m)
{
    // Speed and direction of travel as reported by SUMO, in OMNeT++ coordinates.
    double speed = mobility ? mobility->getSpeed() : 0.0;
    if (mobility && speed > 0.1) {
        Coord dir = mobility->getHeading().toCoord();
        lastHeading = std::atan2(dir.y, dir.x);
    }
    m->setSumoId(mobility ? mobility->getExternalId().c_str() : "");
    m->setX(curPosition.x);
    m->setY(curPosition.y);
    m->setSpeed(speed);
    m->setHeading(lastHeading);
    m->setCpuHz(cpuHz);
    m->setBacklog(backlog());
}

void VehicleApp::generateTask()
{
    long id = nextTaskId++;
    Task& t = tasks[id];
    t.cycles = par("taskCycles").doubleValue();
    t.bytes = par("taskBytes").intValue();
    t.created = simTime();
    t.deadline = simTime() + par("taskDeadline").doubleValue();
    t.deadlineMsg = new cMessage("deadline", KIND_DEADLINE);
    t.deadlineMsg->addPar("task").setLongValue(id);
    scheduleAt(t.deadline, t.deadlineMsg);
    nGenerated++;

    bool covered = rsuAddr >= 0 && simTime() - rsuSeen <= par("coverageTimeout").doubleValue();
    if (!covered) { // no RSU in reach: nobody to ask
        t.decided = true;
        t.mode = MODE_NOCOVERAGE;
        startLocal(id);
        return;
    }
    KvMessage* req = makeMsg(KV_REQUEST, rsuAddr, par("requestBytes").intValue());
    fillState(req);
    req->setTaskId(id);
    req->setCycles(t.cycles);
    req->setBytes(t.bytes);
    req->setCreated(t.created);
    req->setDeadline(t.deadline);
    sendDown(req);

    t.timeoutMsg = new cMessage("decisionTimeout", KIND_DECISION_TIMEOUT);
    t.timeoutMsg->addPar("task").setLongValue(id);
    scheduleAt(t.created + par("decisionTimeoutFrac").doubleValue() * (t.deadline - t.created), t.timeoutMsg);
}

void VehicleApp::startLocal(long id)
{
    scheduleExecDone(enqueueCpu(tasks[id].cycles), myId, id);
}

void VehicleApp::onLocalTaskDone(long taskId)
{
    closeTask(taskId, "ok");
}

void VehicleApp::closeTask(long id, const char* status)
{
    auto it = tasks.find(id);
    if (it == tasks.end()) return; // already closed (e.g. result after the deadline)
    logTask(id, it->second, status);
    if (std::string(status) == "ok") nOk++;
    else nMiss++;
    cancelAndDelete(it->second.deadlineMsg);
    cancelAndDelete(it->second.timeoutMsg);
    tasks.erase(it);
}

void VehicleApp::logTask(long id, const Task& t, const char* status)
{
    if (!taskLog.is_open()) return;
    static const char* modes[] = {"policy", "fallback", "timeout", "nocoverage"};
    const char* target = t.target == TARGET_LOCAL ? "local" : (t.target == TARGET_EDGE ? "edge" : "v2v");
    bool ok = std::string(status) == "ok";
    taskLog << id << ',' << myId << ',' << (mobility ? mobility->getExternalId() : "") << ',' << t.created.dbl() << ','
            << t.cycles << ',' << t.bytes << ',' << (t.deadline - t.created).dbl() << ',' << t.tDecision << ','
            << (t.mode >= 0 ? modes[t.mode] : "none") << ',' << target << ','
            << (ok ? (simTime() - t.created).dbl() : -1.0) << ',' << status << '\n';
}

void VehicleApp::onKv(KvMessage* m)
{
    if (m->getSrc() == rsuAddr) rsuSeen = simTime(); // any frame from the RSU proves coverage
    switch (m->getKvType()) {
    case KV_RSU_BEACON:
        rsuAddr = m->getSrc();
        rsuSeen = simTime();
        break;
    case KV_DECISION: {
        auto it = tasks.find(m->getTaskId());
        if (it == tasks.end() || it->second.decided) break; // too late: task closed or already handled
        Task& t = it->second;
        t.decided = true;
        t.tDecision = (simTime() - t.created).dbl();
        t.mode = m->getMode() == 0 ? MODE_POLICY : MODE_FALLBACK;
        t.target = m->getTarget();
        cancelAndDelete(t.timeoutMsg);
        t.timeoutMsg = nullptr;
        if (t.target == TARGET_LOCAL) startLocal(it->first);
        else sendTaskData(t.target == TARGET_EDGE ? rsuAddr : t.target, it->first, t.cycles, t.bytes, t.created, t.deadline);
        break;
    }
    case KV_RESULT:
        closeTask(m->getTaskId(), "ok");
        break;
    default:
        break;
    }
}

void VehicleApp::onOtherSelfMsg(cMessage* msg)
{
    switch (msg->getKind()) {
    case KIND_TASK:
        generateTask();
        scheduleAt(simTime() + exponential(1.0 / par("taskRate").doubleValue()), taskTimer);
        break;
    case KIND_STATE: {
        KvMessage* s = makeMsg(KV_STATE, -1, par("stateBytes").intValue());
        fillState(s);
        sendDown(s);
        scheduleAt(simTime() + par("stateInterval").doubleValue(), stateTimer);
        break;
    }
    case KIND_DEADLINE: {
        long id = msg->par("task").longValue();
        auto it = tasks.find(id);
        if (it != tasks.end()) {
            it->second.deadlineMsg = nullptr;
            closeTask(id, "miss");
        }
        delete msg;
        break;
    }
    case KIND_DECISION_TIMEOUT: {
        long id = msg->par("task").longValue();
        auto it = tasks.find(id);
        if (it != tasks.end()) {
            it->second.timeoutMsg = nullptr;
            if (!it->second.decided) { // the RSU did not answer in time: run locally
                it->second.decided = true;
                it->second.mode = MODE_TIMEOUT;
                startLocal(id);
            }
        }
        delete msg;
        break;
    }
    default:
        KvBaseApp::onOtherSelfMsg(msg);
    }
}

void VehicleApp::finish()
{
    // tasks whose deadline lies after the end of the run (or of this vehicle's trip) are not counted
    for (auto& kv : tasks) logTask(kv.first, kv.second, "censored");
    taskLog.flush();
    recordScalar("tasksGenerated", nGenerated);
    recordScalar("tasksOk", nOk);
    recordScalar("tasksMissed", nMiss);
    KvBaseApp::finish();
}

} // namespace kvrsu
