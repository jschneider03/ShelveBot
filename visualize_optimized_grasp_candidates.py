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

    diagram_context = env.get_fresh_diagram_context()
    plant_context = env.plant.GetMyContextFromRoot(diagram_context)
    for i, candidate in enumerate(grasp_recommender.generate_random_candidates(
            cloud=pc,
            context=diagram_context)):
        print(f'{i}th candidate')
        print(candidate.fails)
        print(candidate.fail_reason)
        # grasp_recommender.draw_grasp_candidate(X_frame=candidate.X_frame, model_url=GRIPPER_MODEL_URL, meshcat=meshcat, prefix="init grasp")
        env.plant.SetFreeBodyPose(plant_context, gripper.body, candidate.X_frame)
        env.diagram.ForcedPublish(diagram_context)

        optimized_candidate = grasp_recommender.optimize_grasp_with_ik(X_G=candidate.X_frame @ gripper.X_FG(), cloud=pc, context=diagram_context, approaching_vector=[1, 0, 0])
        print(f'{i}th optimized candidate')
        print(optimized_candidate.fails)
        print(optimized_candidate.fail_reason)
        # grasp_recommender.draw_grasp_candidate(X_frame=optimized_candidate.X_frame, model_url=GRIPPER_MODEL_URL, meshcat=meshcat, prefix="optimized grasp")

        # input("press enter to see the optimized version")
        if not candidate.fails:
            print("found a good candidate!")
            break
        if not optimized_candidate.fails:
            print("IK found answer")
            env.plant.SetFreeBodyPose(plant_context, gripper.body, optimized_candidate.X_frame)
            env.diagram.ForcedPublish(diagram_context)
            break
        # input("press enter for next:")
    # candidates = grasp_recommender.generate_random_candidates_one_random_point(cloud=pc, context=context, meshcat=meshcat)
    # for candidate in candidates:
    #     print(candidate.fails)
    #     print(candidate.fail_reason)
    #     grasp_recommender.draw_grasp_candidate(X_frame=candidate.X_frame, model_url=GRIPPER_MODEL_URL, meshcat=meshcat)


    while True:
        sleep(1)


if __name__ == "__main__":
    visualize()
