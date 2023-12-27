import os

from time import sleep

from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.visualization import ModelVisualizer
from pydrake.all import *
from generate_models.create_environment import CustomConfigureParser, Playground
from settings import GRIPPER_MODEL_URL, PR2_MODEL_URL, GRIPPER_BASE_LINK

meshcat: Meshcat = StartMeshcat()


def visualize():
    playground = Playground(meshcat)
    env = playground.construct_welded_sim_with_gripper(
        continuous_state=playground.default_continuous_state(),
        gripper_model_url=GRIPPER_MODEL_URL,
        meshcat=meshcat
    )
    context = env.diagram.CreateDefaultContext()
    plant_context = env.plant.GetMyContextFromRoot(context)
    base_gripper_body: Body = env.plant.GetBodyByName(GRIPPER_BASE_LINK)
    gripper_frame: Frame = base_gripper_body.body_frame()

    AddMeshcatTriad(
        meshcat,
        path="base_frame",
        X_PT=gripper_frame.CalcPoseInWorld(plant_context)
    )

    AddMeshcatTriad(
        meshcat,
        path="gripper_frame",
        X_PT=gripper_frame.CalcPoseInWorld(plant_context) @ RigidTransform(R=RotationMatrix.MakeZRotation(0.16)) @ RigidTransform(p=[0.18, 0, 0]),
        length=0.025,
        radius=0.005
    )

    env.diagram.ForcedPublish(context)
    while True:
        sleep(1)


if __name__ == "__main__":
    visualize()
