import os

from time import sleep

from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.visualization import ModelVisualizer
from pydrake.all import *
from generate_models.create_environment import CustomConfigureParser, Playground
from settings import GRIPPER_MODEL_URL, PR2_MODEL_URL, GRIPPER_BASE_LINK
from grasping import *

meshcat: Meshcat = StartMeshcat()


def visualize():
    playground = Playground(meshcat)
    env = playground.construct_welded_sim_with_gripper(
        continuous_state=playground.default_continuous_state(),
        gripper_model_url=GRIPPER_MODEL_URL,
        meshcat=meshcat,
    )

    body_idx, pc = list(playground.env.point_clouds.items())[0]
    body_pose = playground.env.plant.get_body(body_idx).EvalPoseInWorld(playground.env.get_fresh_plant_context())
    pc = pc.transformed(body_pose)  # change this later

    pc.visualize(name="point cloud", meshcat=meshcat)

    gripper = PR2Gripper(
        plant=env.plant,
        base_link_name=GRIPPER_BASE_LINK,
    )
    grasp_recommender = GraspRecommender(env, gripper)

    context = env.diagram.CreateDefaultContext()

    env.diagram.ForcedPublish(context)

    diagram_context = env.get_fresh_diagram_context()
    plant_context = env.plant.GetMyContextFromRoot(diagram_context)
    
    for i, candidate in enumerate(grasp_recommender.generate_random_candidates(
            cloud=pc,
            context=context,
            approaching_vector=None)):
        print(f'{i}th candidate')
        print(candidate.fails)
        print(candidate.fail_reason)
        print(candidate.X_frame)
        if candidate.X_frame:
            env.plant.SetFreeBodyPose(plant_context, gripper.body, candidate.X_frame)
            env.diagram.ForcedPublish(diagram_context)

        if not candidate.fails:
            print(candidate)
            break
        input("press enter for next:")


    while True:
        sleep(1)


if __name__ == "__main__":
    visualize()
