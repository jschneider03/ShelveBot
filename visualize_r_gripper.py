from pydrake.all import *

import sys

sys.path.append('..')

from generate_models.create_environment import CustomConfigureParser

meshcat = StartMeshcat()

vis = ModelVisualizer(meshcat=meshcat)
parser = vis.parser()
CustomConfigureParser(parser)
parser.AddModelsFromUrl(
    "package://Shelve_Bot/models/r_gripper_model.urdf"
)
vis.Run()
