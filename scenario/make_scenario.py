#!/usr/bin/env python3
"""Generate a SUMO grid scenario with a constant number of vehicles.

    make_scenario.py --out DIR --vehicles 150 --speed 15 [--seed 1]

The network is a square grid (default 5x5 junctions, 250 m blocks = 1 km^2),
so the number of vehicles equals the density in vehicles/km^2.  Rerouters
keep every vehicle driving for the whole run, which holds the density
constant.  ``--speed`` is the speed limit of all roads in m/s.

Files written to DIR:
    net.net.xml, vtypes.add.xml, trips.trips.xml, rerouter.add.xml,
    sim.sumo.cfg   and
    scenario.env   (playground size and RSU position for the OMNeT++ run)

Needs SUMO (netgenerate and $SUMO_HOME/tools).
"""
import argparse
import os
import subprocess
import sys

VEINS_MARGIN = 25  # TraCIScenarioManager.margin (m), added by Veins to all coordinates


def run(cmd, cwd):
    # run inside the scenario directory: some SUMO tools drop helper files into the working directory
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, cwd=cwd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vehicles", type=int, required=True)
    ap.add_argument("--speed", type=float, required=True, help="speed limit in m/s")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grid", type=int, default=5, help="junctions per side")
    ap.add_argument("--block", type=float, default=250.0, help="block length in m")
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--junction", default="traffic_light", choices=["traffic_light", "priority"])
    ap.add_argument("--insert-time", type=float, default=20.0, help="all vehicles depart within this time (s)")
    a = ap.parse_args()

    tools = os.path.join(os.environ.get("SUMO_HOME", ""), "tools")
    if not os.path.isfile(os.path.join(tools, "randomTrips.py")):
        sys.exit("SUMO_HOME is not set or has no tools/randomTrips.py")
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    p = lambda name: os.path.join(a.out, name)

    run(["netgenerate", "--grid", "--grid.number", str(a.grid), "--grid.length", str(a.block),
         "--default.lanenumber", str(a.lanes), "--default.speed", str(a.speed),
         "--default-junction-type", a.junction, "--output-file", p("net.net.xml")], a.out)

    with open(p("vtypes.add.xml"), "w") as f:
        f.write('<additional>\n  <vType id="car" vClass="passenger" maxSpeed="%.1f" '
                'speedFactor="normc(1,0.1,0.6,1.2)" sigma="0.5"/>\n</additional>\n' % max(a.speed * 1.2, 5))

    run([sys.executable, os.path.join(tools, "randomTrips.py"), "-n", p("net.net.xml"), "-o", p("trips.trips.xml"),
         "-b", "0", "-e", str(a.insert_time), "--period", str(a.insert_time / a.vehicles),
         "--seed", str(a.seed), "--fringe-factor", "1",
         "--trip-attributes", 'type="car" departLane="best" departSpeed="max" departPos="random_free"'], a.out)

    run([sys.executable, os.path.join(tools, "generateContinuousRerouters.py"), "-n", p("net.net.xml"),
         "-o", p("rerouter.add.xml")], a.out)

    with open(p("sim.sumo.cfg"), "w") as f:
        f.write("""<configuration>
  <input>
    <net-file value="net.net.xml"/>
    <route-files value="trips.trips.xml"/>
    <additional-files value="vtypes.add.xml,rerouter.add.xml"/>
  </input>
  <time><begin value="0"/><step-length value="0.1"/></time>
  <processing><time-to-teleport value="-1"/><collision.action value="none"/></processing>
  <report><no-step-log value="true"/><no-warnings value="true"/></report>
</configuration>
""")

    extent = (a.grid - 1) * a.block
    with open(p("scenario.env"), "w") as f:
        f.write("PLAYGROUND=%d\nRSU_X=%.1f\nRSU_Y=%.1f\nAREA_KM2=%.4f\n" % (
            extent + 2 * VEINS_MARGIN + 50, extent / 2 + VEINS_MARGIN, extent / 2 + VEINS_MARGIN,
            (extent / 1000.0) ** 2))
    print("scenario written to", a.out)


if __name__ == "__main__":
    main()
