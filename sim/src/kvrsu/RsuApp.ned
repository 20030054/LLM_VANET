package kvrsu;

import org.car2x.veins.modules.application.ieee80211p.DemoBaseApplLayer;

//
// RSU side of the offloading scenario. See RsuApp.h.
//
simple RsuApp extends DemoBaseApplLayer
{
    parameters:
        @class(kvrsu::RsuApp);
        double cpuHz = default(2e10);                    // edge-server CPU capacity, cycles/s
        int fragBytes = default(1300);
        int resultBytes = default(100);
        int beaconBytes = default(40);
        int decisionBytes = default(60);
        double window @unit(s) = default(0.1s);          // decision window length (Delta)
        double tableTimeout @unit(s) = default(2.5s);    // drop vehicles silent for this long
        double rsuBeaconInterval @unit(s) = default(1s);
        double procDelay @unit(s) = default(0.1ms);      // fixed handling delay per decision
        string serverHost = default("127.0.0.1");
        int serverPort;                                  // port of the decision server
        double connectTimeout @unit(s) = default(900s);  // how long to wait for the server to come up
        double computeScale = default(1.0);              // multiplies measured inference time (emulates slower hardware)
        double fixedServiceTime @unit(s) = default(-1s); // >= 0: ignore measured time (debugging only)
        string logPrefix;                                // path prefix of the -tasks.csv and -windows.csv files
}
