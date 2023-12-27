import os
import sys
import time

from pydrake.all import (
    StartMeshcat,
    Simulator,
    Diagram
)
from pydrake.multibody.plant import MultibodyPlant

from generate_models import Playground
from utils import min_distance_collision_checker

sys.path.append(os.path.dirname(__file__))

meshcat = StartMeshcat()


def simulate_env():
    playground = Playground(meshcat, time_step=0.001)
    diagram: Diagram = playground.env.diagram
    point_clouds = playground.env.point_clouds
    plant: MultibodyPlant = playground.env.plant

    simulator = Simulator(diagram)
    simulator.AdvanceTo(0.01)

    for body_idx, pcd in point_clouds.items():
        context = plant.GetMyContextFromRoot(simulator.get_context())
        X_WO = plant.EvalBodyPoseInWorld(context=context, body=plant.get_body(body_idx))
        meshcat.SetLineSegments(
            'body_' + str(int(body_idx)) + "_pcd",
            X_WO @ pcd.xyzs(),
            (X_WO @ pcd.xyzs()) + 0.01 * (X_WO.rotation() @ pcd.normals()),
        )

    # meshcat.StartRecording()
    # simulator.AdvanceTo(5.0)
    # meshcat.StopRecording()
    # meshcat.PublishRecording()

    # plant: MultibodyPlant = playground['plant']
    # print(min_distance_collision_checker(plant, plant.GetMyContextFromRoot(simulator.get_context()), 0))

    while True:  # to make simulation meshcat run
        time.sleep(1)


if __name__ == "__main__":
    simulate_env()
