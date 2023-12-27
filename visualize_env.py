import os

from pydrake.visualization import ModelVisualizer
from pydrake.all import StartMeshcat
from generate_models.create_environment import CustomConfigureParser
from settings import GRIPPER_MODEL_URL, PR2_MODEL_URL

meshcat = StartMeshcat()


def visualize_env():
    vis = ModelVisualizer(meshcat=meshcat)
    parser = CustomConfigureParser(vis.parser())
    parser.AddModelsFromUrl(PR2_MODEL_URL)
    vis.Run()


if __name__ == "__main__":
    visualize_env()
