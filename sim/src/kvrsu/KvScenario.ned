package kvrsu;

import org.car2x.veins.base.connectionManager.ConnectionManager;
import org.car2x.veins.base.modules.BaseWorldUtility;
import org.car2x.veins.modules.mobility.traci.TraCIScenarioManagerForker;
import org.car2x.veins.modules.obstacle.ObstacleControl;
import org.car2x.veins.modules.world.annotations.AnnotationManager;
import org.car2x.veins.nodes.Car;
import org.car2x.veins.nodes.RSU;

//
// One RSU and SUMO-driven vehicles. SUMO is started by the simulation itself
// (TraCIScenarioManagerForker), so no launch daemon is needed on the cluster.
//
network KvScenario
{
    parameters:
        double playgroundSizeX @unit(m);
        double playgroundSizeY @unit(m);
        double playgroundSizeZ @unit(m);
    submodules:
        obstacles: ObstacleControl;
        annotations: AnnotationManager;
        connectionManager: ConnectionManager;
        world: BaseWorldUtility {
            playgroundSizeX = parent.playgroundSizeX;
            playgroundSizeY = parent.playgroundSizeY;
            playgroundSizeZ = parent.playgroundSizeZ;
        }
        manager: TraCIScenarioManagerForker;
        rsu[1]: RSU;
        node[0]: Car;
    connections allowunconnected:
}
