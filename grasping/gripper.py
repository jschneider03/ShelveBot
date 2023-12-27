from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from pydrake.all import Frame, RigidTransform, RotationMatrix, Body
from pydrake.multibody.plant import MultibodyPlant


@dataclass
class Gripper(ABC):
    plant: MultibodyPlant
    base_link_name: str

    @property
    def body(self):
        return self.plant.GetBodyByName(self.base_link_name)

    @property
    def frame(self):
        return self.body.body_frame()

    @abstractmethod
    def is_in_grasp_zone(self, x, y, z):
        """
            coordinates are in gripper's frame
        """
        pass

    @abstractmethod
    def X_WG(self, plant_context) -> RigidTransform:
        pass

    @abstractmethod
    def X_FG(self):
        pass

    @abstractmethod
    def SetPose(self, plant_context, X_G):
        pass

    @abstractmethod
    def p_GS_G(self):
        pass

    @abstractmethod
    def X_GH_centralize_wrt_point_cloud(self, x, y, z):
        pass

    def X_frame(self, plant_context) -> RigidTransform:
        return self.frame.CalcPoseInWorld(plant_context)


class PR2Gripper(Gripper):
    def is_in_grasp_zone(self, x, y, z):
        return np.all([
            y <= 0.026,
            -0.026 <= y,
            x <= 0.02,
            -0.02 <= x,
            z <= 0.01,
            -0.01 <= z
            ],
            axis=0
        )

    def X_GH_centralize_wrt_point_cloud(self, x, y, z):
        delta_y = (np.max(y) + np.min(y))/2
        return RigidTransform(p=[0, delta_y, 0])

    def X_FG(self):
        return RigidTransform(R=RotationMatrix.MakeZRotation(0.16)) @ RigidTransform(p=[0.18, 0, 0])

    def X_WG(self, plant_context):
        return self.X_frame(plant_context) @ self.X_FG()

    def SetPose(self, plant_context, X_G):
        self.plant.SetFreeBodyPose(plant_context, self.body, X_G @ self.X_FG().inverse())

    def p_GS_G(self):
        return [0, 0.028, 0]
