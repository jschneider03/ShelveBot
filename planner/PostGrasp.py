from pydrake.multibody.tree import Frame

import utils
from environment import MovableBody
from generate_models import Playground
from grasping import PR2Gripper
from planner.Action import Action
from planner.Command import Command, Hand
from robots import PR2
from settings import GRIPPER_BASE_LINK

import numpy as np

from pydrake.all import (
    Solve,
    SolverOptions,
    MinimumDistanceLowerBoundConstraint
)
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import SnoptSolver

from utils import *


class PostGrasp(Action):
    """
    Moves PR2 into a post-grasp pose where the block is no longer in contact with the shelf floor,
    avoiding issues stemming from the resulting friction.
    """

    def __init__(self, playground: Playground, hand: Hand, movable_body: MovableBody):
        super().__init__(playground)
        self.hand = hand
        self.movable_body = movable_body

        self.start_time = 0
        self.my_traj = None
        self.total_time = 0
        self.pr2 = None
        self.gripper_env = None
        self.gripper = None


    def state_init(self):
        self.start_time = self.time
        # since gripper is too close to the object it cause problem for MinDistanceConstraint?
        # we weld the object to the gripper
        print("trying to construct welded env with object")
        self.pr2 = PR2(self.playground.construct_welded_sim_with_object_welded(
            continuous_state=self.continuous_state,
            frame_name_to_weld=self.hand.connecting_frame_name(),
            mb=self.movable_body,
        ))
        self.gripper_env = self.playground.construct_welded_sim_with_gripper(self.continuous_state)
        self.gripper = PR2Gripper(plant=self.gripper_env.plant, base_link_name=GRIPPER_BASE_LINK)

        context = self.pr2.env.get_fresh_plant_context(self.continuous_state)

        mb = self.movable_body
        gripper_frame_name = self.hand.connecting_frame_name()
        gripper_frame: Frame = self.pr2.plant.GetFrameByName(gripper_frame_name)
        object_frame: Frame = mb.get_body(self.pr2.plant).body_frame()
        X_init_WF = gripper_frame.CalcPoseInWorld(context)
        X_init_WO = object_frame.CalcPoseInWorld(context)

        X_WG =  mb.get_pose(self.pr2.plant, context)
        for shelf in self.playground.env.get_all_shelves():
            if shelf.is_point_in_shelf(X_WG.translation()):
                current_shelf = shelf
                break
        X_final_WO = current_shelf.get_pre_grasp_pose(X_WG, z=0.1, x=0.0)
        X_FO = X_init_WF.inverse() @ X_init_WO
        X_final_WF = X_final_WO @ X_FO.inverse()


        print('starting traj opt')
        utils.min_distance_collision_checker(self.pr2.plant, self.pr2.get_fresh_plant_context(), 0.01)
        self.try_only_to(gripper_frame_name, X_final_WF)
        print('traj opt done')

    def run(self, prev_command: Command):
        t = self.time - self.start_time
        done = t > self.total_time
        return prev_command.new_command_ignore_grippers(self.my_traj.value(t)), done

    def try_only_to(self, gripper_frame_name, X_WF):
        distance_lower_bound = 0.03
        self.my_traj = self.pr2.kinematic_trajectory_optimization_ungrasping(
            gripper_frame_name, X_WF,
            distance_lower_bound=distance_lower_bound,
            num_control_points=10,
            max_duration=10,
            min_duration=1
        )
        self.total_time = self.my_traj.end_time() - self.my_traj.start_time()
