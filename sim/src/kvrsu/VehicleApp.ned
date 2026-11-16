package kvrsu;

import org.car2x.veins.modules.application.ieee80211p.DemoBaseApplLayer;

//
// Vehicle side of the offloading scenario. See VehicleApp.h.
//
simple VehicleApp extends DemoBaseApplLayer
{
    parameters:
        @class(kvrsu::VehicleApp);
        double cpuHz = default(uniform(0.5e9, 2e9));            // CPU capacity, cycles/s (drawn once per vehicle)
        int fragBytes = default(1300);                           // payload per data frame
        int resultBytes = default(100);
        int stateBytes = default(60);
        int requestBytes = default(80);
        double warmup @unit(s) = default(30s);                   // no tasks before this time
        double stateInterval @unit(s) = default(0.5s);           // period of state reports to the RSU
        double coverageTimeout @unit(s) = default(2.5s);         // RSU considered out of reach after this silence
        double taskRate = default(0.2);                          // tasks per second per vehicle (Poisson)
        volatile double taskCycles = default(uniform(1e8, 8e8)); // CPU cycles per task
        volatile int taskBytes = default(intuniform(2000, 8000));// input size per task
        volatile double taskDeadline @unit(s) = default(uniform(0.5s, 1.5s));
        double decisionTimeoutFrac = default(0.5);               // wait this share of the deadline for a decision
}
