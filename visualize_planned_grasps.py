import os

from time import sleep

from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.visualization import ModelVisualizer
from pydrake.all import *

import utils
from generate_models.create_environment import CustomConfigureParser, Playground
from settings import GRIPPER_MODEL_URL, PR2_MODEL_URL, GRIPPER_BASE_LINK
from grasping import *

meshcat: Meshcat = StartMeshcat()


def visualize():
    playground = Playground(meshcat, reset_cache=True)
    env = playground.construct_welded_sim_with_gripper(
        continuous_state=playground.default_continuous_state(),
        gripper_model_url=GRIPPER_MODEL_URL,
        meshcat=meshcat,
    )

    diagram_context = env.get_fresh_diagram_context()
    plant_context = env.plant.GetMyContextFromRoot(diagram_context)
    scene_graph_context = env.scene_graph.GetMyContextFromRoot(diagram_context)

    for mb in playground.env.movable_bodies:
        pc = mb.point_cloud
        body_pose = mb.get_pose(env.plant, plant_context)
        pc = pc.transformed(body_pose)  # change this later

        pc.visualize(name="point cloud", meshcat=meshcat)

        gripper_body = env.plant.GetBodyIndices(env.model_id)[0]
        gripper_body = env.plant.get_body(gripper_body)

        for i, grasp in enumerate(mb.X_OF_grasp_candidates):
            env.plant.SetFreeBodyPose(plant_context, gripper_body, body_pose @ grasp)
            env.diagram.ForcedPublish(diagram_context)
            query_object: QueryObject = env.scene_graph.get_query_output_port().Eval(scene_graph_context)
            print("distances less than 1cm")
            utils.min_distance_collision_checker(env.plant, plant_context, 0.01)
            input(f"({i})  press enter for next:")


if __name__ == "__main__":
    visualize()
