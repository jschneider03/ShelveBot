from dataclasses import dataclass

from pydrake.all import RigidTransform, Context
from pydrake.multibody.plant import MultibodyPlant

from perception import CustomPointCloud
from typing import List

import numpy as np


@dataclass
class MovableBody:
    body_name: str
    model_name: str
    # point cloud is in object frame
    point_cloud: CustomPointCloud
    X_WO_init: RigidTransform
    X_WO_end: RigidTransform
    X_OF_grasp_candidates: List[RigidTransform]

    def get_body(self, plant: MultibodyPlant):
        model_instance = plant.GetModelInstanceByName(self.model_name)
        body = plant.GetBodyByName(self.body_name, model_instance)
        return body

    def get_pose(self, plant: MultibodyPlant, context: Context):
        return self.get_body(plant).EvalPoseInWorld(context)

    def shift_pose(self, plant: MultibodyPlant, context: Context, x=0., y=0., z=0.):
        X_BF = self.get_body(plant).body_frame().CalcPoseInBodyFrame(context)
        X_BF.set_translation(X_BF.translation() + np.array([x, y, z]))
        return self.get_body(plant).EvalPoseInWorld(context) @ X_BF

    def is_same(self, other: "MovableBody"):
        return self.body_name == other.body_name and self.model_name == other.model_name
