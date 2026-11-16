#include "kvrsu/RsuApp.h"

#include <iomanip>
#include <sstream>

#include "veins/base/modules/BaseMobility.h"

using namespace veins;

namespace kvrsu {

Define_Module(kvrsu::RsuApp);

RsuApp::~RsuApp()
{
    cancelAndDelete(windowTimer);
    cancelAndDelete(beaconTimer);
}

void RsuApp::initialize(int stage)
{
    KvBaseApp::initialize(stage);
    if (stage == 0) {
        BaseMobility* mob = FindModule<BaseMobility*>::findSubModule(getParentModule());
        Coord pos = mob->getPositionAt(simTime());
        posX = pos.x;
        posY = pos.y;

        std::string prefix = par("logPrefix").stdstringValue();
        if (taskLog.is_open()) taskLog.close();
        taskLog.open(prefix + "-tasks.csv");
        taskLog << "task,veh,sumo_id,created,cycles,bytes,deadline,t_decision,mode,target,t_done,status\n";
        winLog.open(prefix + "-windows.csv");
        winLog << "t,window,n_veh,n_req,n_policy,p,u,m,ghat,t_service,t_pred\n";

        windowTimer = new cMessage("window", KIND_WINDOW);
        beaconTimer = new cMessage("rsuBeacon", KIND_BEACON);
        engineFreeAt = 0;
        scheduleAt(simTime() + par("window").doubleValue(), windowTimer);
        scheduleAt(simTime() + uniform(0, 0.1), beaconTimer);
    }
    else if (stage == 1) {
        engine.connectTo(par("serverHost").stdstringValue(), par("serverPort").intValue(), par("connectTimeout").doubleValue());
    }
}

void RsuApp::updateEntry(KvMessage* m)
{
    Entry& e = table[m->getSrc()];
    e.sumoId = m->getSumoId();
    e.x = m->getX();
    e.y = m->getY();
    e.speed = m->getSpeed();
    e.heading = m->getHeading();
    e.cpuHz = m->getCpuHz();
    e.backlog = m->getBacklog();
    e.seen = simTime();
}

void RsuApp::onKv(KvMessage* m)
{
    if (m->getKvType() == KV_STATE) {
        updateEntry(m);
    }
    else if (m->getKvType() == KV_REQUEST) {
        updateEntry(m);
        pending.push_back({m->getTaskId(), m->getSrc(), m->getCycles(), m->getBytes(), m->getCreated(), m->getDeadline()});
        nRequests++;
    }
}

void RsuApp::runWindow()
{
    simtime_t now = simTime();

    // forget vehicles that have been silent for too long and requests that can no longer be met
    double timeout = par("tableTimeout").doubleValue();
    for (auto it = table.begin(); it != table.end();) it = (now - it->second.seen > timeout) ? table.erase(it) : std::next(it);
    std::vector<Req> live;
    for (const Req& r : pending) {
        if (r.deadline > now) live.push_back(r);
        else nExpired++;
    }
    pending.clear();
    if (live.empty()) return;

    std::ostringstream os;
    os << std::setprecision(9);
    os << "WINDOW " << ++windowIdx << ' ' << now.dbl() << ' ' << posX << ' ' << posY << ' ' << cpuHz << ' ' << backlog() << ' '
       << table.size() << ' ' << live.size() << '\n';
    for (const auto& kv : table) {
        const Entry& e = kv.second;
        os << "V " << kv.first << ' ' << (e.sumoId.empty() ? "-" : e.sumoId) << ' ' << e.x << ' ' << e.y << ' ' << e.speed << ' '
           << e.heading << ' ' << e.cpuHz << ' ' << e.backlog << '\n';
    }
    for (const Req& r : live) {
        os << "R " << r.id << ' ' << r.addr << ' ' << r.cycles << ' ' << r.bytes << ' ' << (r.deadline - r.created).dbl() << ' '
           << (now - r.created).dbl() << '\n';
    }
    os << "END\n";

    std::istringstream reply(engine.exchange(os.str()));
    std::string tag;
    long n, nPol, p, u, m, ghat;
    double tService, tPred;
    reply >> tag >> n >> tService >> nPol >> p >> u >> m >> ghat >> tPred;
    if (tag != "RESULT" || n != (long) live.size()) throw cRuntimeError("RsuApp: malformed reply from decision server");

    double fixed = par("fixedServiceTime").doubleValue();
    double scale = par("computeScale").doubleValue();
    if (fixed >= 0) tService = fixed;
    tService *= scale;

    std::map<long, long> addrOf;
    for (const Req& r : live) addrOf[r.id] = r.addr;
    for (long i = 0; i < n; ++i) {
        long id, target;
        int mode;
        double delay;
        reply >> tag >> id >> target >> mode >> delay;
        if (tag != "D") throw cRuntimeError("RsuApp: malformed decision line");
        KvMessage* d = makeMsg(KV_DECISION, addrOf[id], par("decisionBytes").intValue());
        d->setTaskId(id);
        d->setTarget(target);
        d->setMode(mode);
        cMessage* carrier = new cMessage("sendDecision", KIND_SEND);
        carrier->setContextPointer(d);
        // decisions of the policy are ready when the engine has finished; fallback decisions at once
        scheduleAt(now + (delay > 0 ? tService : 0.0) + par("procDelay").doubleValue(), carrier);
        if (mode == 0) nPolicy++;
        else nFallback++;
    }
    engineFreeAt = now + tService;
    winLog << now.dbl() << ',' << windowIdx << ',' << table.size() << ',' << live.size() << ',' << nPol << ',' << p << ',' << u << ','
           << m << ',' << ghat << ',' << tService << ',' << tPred * scale << '\n';
}

void RsuApp::onOtherSelfMsg(cMessage* msg)
{
    switch (msg->getKind()) {
    case KIND_WINDOW:
        // a window opens only when the engine is idle: slow inference stretches the windows
        if (simTime() >= engineFreeAt) runWindow();
        scheduleAt(std::max(simTime() + par("window").doubleValue(), engineFreeAt), windowTimer);
        break;
    case KIND_BEACON:
        sendDown(makeMsg(KV_RSU_BEACON, -1, par("beaconBytes").intValue()));
        scheduleAt(simTime() + par("rsuBeaconInterval").doubleValue(), beaconTimer);
        break;
    case KIND_SEND:
        sendDown(static_cast<KvMessage*>(msg->getContextPointer()));
        delete msg;
        break;
    default:
        KvBaseApp::onOtherSelfMsg(msg);
    }
}

void RsuApp::finish()
{
    recordScalar("requests", nRequests);
    recordScalar("decisionsPolicy", nPolicy);
    recordScalar("decisionsFallback", nFallback);
    recordScalar("requestsExpired", nExpired);
    recordScalar("windows", windowIdx);
    engine.close();
    winLog.close();
    taskLog.flush();
    KvBaseApp::finish();
}

} // namespace kvrsu
