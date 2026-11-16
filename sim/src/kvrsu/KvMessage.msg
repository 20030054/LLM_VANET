//
// One packet type for all application messages of the kvrsu scenario.
//
import veins.modules.messages.BaseFrame1609_4;

namespace kvrsu;

enum KvType {
    KV_STATE = 0;       // vehicle -> RSU   periodic state report (broadcast)
    KV_RSU_BEACON = 1;  // RSU -> all       announces the RSU and its address
    KV_REQUEST = 2;     // vehicle -> RSU   asks where to run a task
    KV_DECISION = 3;    // RSU -> vehicle   answer to a request
    KV_DATA = 4;        // vehicle -> executor  one fragment of the task input
    KV_RESULT = 5;      // executor -> vehicle  task result
}

packet KvMessage extends veins::BaseFrame1609_4 {
    int kvType;
    long src = -1;          // sender address (MAC address of the node)
    long dst = -1;          // intended receiver, -1 = everybody
    long taskId = -1;

    // sender state (KV_STATE, KV_REQUEST)
    string sumoId;
    double x;
    double y;
    double speed;           // m/s
    double heading;         // rad, direction of travel
    double cpuHz;           // CPU capacity in cycles/s
    double backlog;         // seconds of queued work

    // task description (KV_REQUEST, KV_DATA)
    double cycles;
    long bytes;
    simtime_t created;
    simtime_t deadline;     // absolute time

    // KV_DECISION
    long target = -1;       // vehicle address, -1 = local, -2 = RSU
    int mode = 0;           // 0 = policy under test, 1 = heuristic fallback

    // KV_DATA
    int fragCount = 1;
}
