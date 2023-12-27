from dataclasses import dataclass

from pydrake.all import RigidTransform, Context, RotationMatrix
from pydrake.multibody.plant import MultibodyPlant

from perception import CustomPointCloud
from typing import List

import numpy as np


@dataclass
class FixedBody:
    body_name: str
    model_name: str

    def get_body(self, plant: MultibodyPlant):
        model_instance = plant.GetModelInstanceByName(self.model_name)
        body = plant.GetBodyByName(self.body_name, model_instance)
        return body

    def get_pose(self, plant: MultibodyPlant, context: Context):
        return self.get_body(plant).EvalPoseInWorld(context)


@dataclass
class Shelve(FixedBody):
    body_name: str
    model_name: str
    depth: float
    height: float
    floors: int
    thickness: float
    normal_direction: np.ndarray
    center_pos: np.ndarray

    def __init__(self, body_name, model_name, depth, width, height, floors, thickness, normal_direction: np.ndarray, center_pos: np.ndarray):
        super(Shelve, self).__init__(body_name=body_name, model_name=model_name)
        self.depth = depth
        self.width = width
        self.height = height
        self.floors = floors
        self.thickness = thickness
        self.normal_direction = normal_direction
        self.center_pos = center_pos

        self.up_axis = np.array([0, 0, 1])
        self.right_axis = np.cross(self.up_axis, self.normal_direction)

        self.floor_height = self.height / self.floors

    def is_point_in_shelf(self, xyz):
        d = xyz - self.center_pos
        return np.all((
            np.abs(self.normal_direction @ d) <= self.depth/2,
            np.abs(self.right_axis @ d) <= self.width / 2,
            np.abs(self.up_axis @ d) <= self.height / 2,
        ), axis=0)
    
    def get_pre_grasp_pose(self, X_WG, x=0.15, y=0., z=0.):
        r_X_WP = RotationMatrix(np.hstack((self.normal_direction.reshape(-1, 1), 
                                           self.right_axis.reshape(-1, 1), 
                                           self.up_axis.reshape(-1, 1))))
        p_X_WP = X_WG.translation()
        X_WP = RigidTransform(r_X_WP, p_X_WP)
        X_GP = RigidTransform(RotationMatrix(), 
                              np.array([x + self.depth/2, y, z]))
        X_WP.set_translation((X_WG @ X_GP).translation())
        return X_WP