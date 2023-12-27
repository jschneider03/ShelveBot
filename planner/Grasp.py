from pydrake.multibody.plant import MultibodyPlant

import utils
from environment import MovableBody
from grasping import PR2Gripper, GraspRecommender
from robots import PR2
from settings import GRIPPER_BASE_LINK
from .Action import Action
from generate_models import Playground
from .Command import Hand, Command


class Grasp(Action):
    """
    Moves PR2 from pre-grasp pose into into grasping the object of interest.
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
        self.pr2 = PR2(self.playground.construct_welded_sim(self.continuous_state))

        self.gripper_env = self.playground.construct_welded_sim_with_gripper(self.continuous_state)
        self.gripper = PR2Gripper(plant=self.gripper_env.plant, base_link_name=GRIPPER_BASE_LINK)

        mb = self.movable_body
        gripper_frame_name = self.hand.connecting_frame_name()
        for X_WF in self.get_best_grasps_sorted(gripper_frame_name, mb):
            try:
                utils.min_distance_collision_checker(self.pr2.plant, self.pr2.get_fresh_plant_context(), 0.01)
                self.try_only_to(gripper_frame_name, X_WF)
                break
            except Exception as e:
                print(e)

    def run(self, prev_command: Command):
        t = self.time - self.start_time
        done = t > self.total_time
        return prev_command.new_command_ignore_grippers(self.my_traj.value(t)), done


    def try_only_to(self, gripper_frame_name, X_WF):
        distance_lower_bound = 0.01
        print("distance lower bound is ", distance_lower_bound)
        self.my_traj = self.pr2.kinematic_trajectory_optimization_gripper(
            gripper_frame_name, X_WF,
            gripper=self.gripper,
            distance_lower_bound=distance_lower_bound,
            num_control_points=10,
            max_duration=10,
            min_duration=1
        )
        self.total_time = self.my_traj.end_time() - self.my_traj.start_time()

    def get_best_grasps_sorted(self, gripper_frame, mb: MovableBody):
        pr2_context = self.pr2.env.get_fresh_plant_context(self.continuous_state)
        pr2_plant: MultibodyPlant = self.pr2.plant
        cur_X_WF = pr2_plant.GetFrameByName(gripper_frame).CalcPoseInWorld(pr2_context)

        gripper_env = self.playground.construct_welded_sim_with_gripper(self.continuous_state,
                                                                        meshcat=self.playground.meshcat)
        gripper = PR2Gripper(plant=gripper_env.plant, base_link_name=GRIPPER_BASE_LINK)
        recommender = GraspRecommender(
            gripper_env, gripper
        )

        gripper_diagram_context = gripper_env.get_fresh_diagram_context()
        gripper_context = gripper_env.plant.GetMyContextFromRoot(gripper_diagram_context)
        candidates = []
        for X_OF in mb.X_OF_grasp_candidates:
            X_WC = mb.get_pose(gripper_env.plant, gripper_context)
            X_G = X_WC @ X_OF @ recommender.gripper.X_FG()
            approaching_vector = X_WC.translation() - cur_X_WF.translation()
            candidate = recommender.eval_candidate(X_G,
                                                   cloud=mb.point_cloud.transformed(X_WC),
                                                   context=gripper_diagram_context,
                                                   approaching_vector=approaching_vector
                                                   )
            if candidate.fails:
                continue
            candidates.append(candidate)
        candidates.sort(key=lambda item: item.cost)
        for item in candidates:
            # todo this is for debugging only
            gripper_env.plant.SetFreeBodyPose(context=gripper_context,
                                              body=gripper_env.plant.get_body(
                                                  gripper_env.plant.GetBodyIndices(gripper_env.model_id)[0]),
                                              X_WB=item.X_frame)
            #####
            # this is for the debug
            gripper_env.ForcedPublish(gripper_diagram_context)
            #####

            yield item.X_frame
