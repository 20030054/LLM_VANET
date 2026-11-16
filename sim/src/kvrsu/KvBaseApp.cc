#include "kvrsu/KvBaseApp.h"

#include <algorithm>
#include <cmath>

using namespace veins;

namespace kvrsu {

std::ofstream KvBaseApp::taskLog;

void KvBaseApp::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        cpuHz = par("cpuHz").doubleValue();
        fragBytes = par("fragBytes");
        busyUntil = 0;
    }
}

double KvBaseApp::backlog() const
{
    return std::max(0.0, (busyUntil - simTime()).dbl());
}

// First-come-first-served CPU: returns the time at which the new job finishes.
simtime_t KvBaseApp::enqueueCpu(double cycles)
{
    simtime_t start = std::max(simTime(), busyUntil);
    busyUntil = start + cycles / cpuHz;
    return busyUntil;
}

void KvBaseApp::scheduleExecDone(simtime_t at, long src, long taskId)
{
    cMessage* m = new cMessage("execDone", KIND_EXEC_DONE);
    m->addPar("src").setLongValue(src);
    m->addPar("task").setLongValue(taskId);
    scheduleAt(at, m);
}

KvMessage* KvBaseApp::makeMsg(int type, long dst, int payloadBytes)
{
    KvMessage* m = new KvMessage("kv");
    // unicast frames are acknowledged and retransmitted by the MAC (useAcks)
    populateWSM(m, dst < 0 ? LAddress::L2BROADCAST() : dst);
    m->addByteLength(payloadBytes);
    m->setKvType(type);
    m->setSrc(myId);
    m->setDst(dst);
    return m;
}

// The task input is sent as a train of MAC frames. The receiver starts the
// task when it has seen all of them.
void KvBaseApp::sendTaskData(long dst, long taskId, double cycles, long bytes, simtime_t created, simtime_t deadline)
{
    int count = std::max(1L, (bytes + fragBytes - 1) / fragBytes);
    for (int i = 0; i < count; ++i) {
        int len = (i == count - 1) ? (int) (bytes - (long) fragBytes * (count - 1)) : fragBytes;
        KvMessage* m = makeMsg(KV_DATA, dst, std::max(len, 1));
        m->setTaskId(taskId);
        m->setCycles(cycles);
        m->setBytes(bytes);
        m->setCreated(created);
        m->setDeadline(deadline);
        m->setFragCount(count);
        sendDown(m);
    }
}

void KvBaseApp::onWSM(BaseFrame1609_4* frame)
{
    KvMessage* m = dynamic_cast<KvMessage*>(frame);
    if (!m) return;
    if (m->getDst() >= 0 && m->getDst() != myId) return; // not for this node

    if (m->getKvType() == KV_DATA) {
        if (simTime() > m->getDeadline()) return; // too late to be useful
        auto key = std::make_pair(m->getSrc(), m->getTaskId());
        Rx& r = rx[key];
        r.cycles = m->getCycles();
        r.deadline = m->getDeadline();
        if (++r.got == m->getFragCount()) {
            scheduleExecDone(enqueueCpu(r.cycles), m->getSrc(), m->getTaskId());
            rx.erase(key);
        }
        if (rx.size() > 64) { // drop reassembly state of tasks that can no longer finish
            for (auto it = rx.begin(); it != rx.end();) it = (simTime() > it->second.deadline) ? rx.erase(it) : std::next(it);
        }
        return;
    }
    onKv(m);
}

void KvBaseApp::handleSelfMsg(cMessage* msg)
{
    if (msg->getKind() == KIND_EXEC_DONE) {
        long src = msg->par("src").longValue();
        long task = msg->par("task").longValue();
        delete msg;
        if (src == myId) {
            onLocalTaskDone(task);
        }
        else {
            KvMessage* r = makeMsg(KV_RESULT, src, par("resultBytes").intValue());
            r->setTaskId(task);
            sendDown(r);
        }
        return;
    }
    onOtherSelfMsg(msg);
}

void KvBaseApp::onOtherSelfMsg(cMessage* msg)
{
    DemoBaseApplLayer::handleSelfMsg(msg);
}

} // namespace kvrsu
