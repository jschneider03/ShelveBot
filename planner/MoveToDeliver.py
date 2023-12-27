from environment import MovableBody
from generate_models import Playground
from planner.Action import Action
from planner.Command import Command

from random import random

import numpy as np
from pydrake.all import (
    InverseKinematics,
    Solve,
    SolverOptions,
    MinimumDistanceLowerBoundConstraint,
    CalcGridPointsOptions
)
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.multibody.optimization import Toppra


from robot import ConfigurationSpace, Range

from pydrake.solvers import SnoptSolver

from manipulation.exercises.trajectories.rrt_planner.geometry import AABB, Point

from generate_models import Playground
from rrt_planning import Problem
from settings import FLOOR_FOR_BRICKS, PRESEEDED_IK
from utils import *

from robots import PR2


class MoveToDeliver(Action):
    """
    Navigates from post-grasp pose to delivery
    """

    def __init__(self, playground: Playground, movable_body: MovableBody, gripper_name, max_iterations=1000, prob_sample_goal=0.05):
        super().__init__(playground)
        self.movable_body = movable_body

        self.max_iterations = max_iterations
        self.prob_sample_goal = prob_sample_goal
        self.rrt_ix = 0
        self.start_time = 0
        self.traj_opt = None
        self.toppra_path = None
        self.regenerate_path = False
        self.pr2 = None
        self.gripper_name = gripper_name
        self.gripper_base_link_name = "l_gripper_palm_link" if "l_gripper" in gripper_name else "r_gripper_palm_link"

    def state_init(self):
        self.start_time = self.time
        self.pr2 = PR2(self.playground.construct_welded_sim_with_object_welded(
                continuous_state=self.continuous_state,
                frame_name_to_weld=self.gripper_base_link_name,
                mb=self.movable_body,
            ))
        plant_context = self.pr2.get_fresh_plant_context()
        current_shelf = None

        X_WG = self.movable_body.X_WO_end

        for shelf in self.playground.env.get_all_shelves():
            if shelf.is_point_in_shelf(X_WG.translation()):
                current_shelf = shelf
                break
        self.X_WS = current_shelf.get_pre_grasp_pose(X_WG, x=-0.05)
        
        query_object = self.pr2.scene_graph.get_query_output_port().Eval(self.pr2.get_fresh_scene_graph_context())
        self.q_init = theta_adjusted_pos(self.pr2.plant.GetPositions(plant_context))
        if PRESEEDED_IK and FLOOR_FOR_BRICKS == 2:
            self.q_goal = np.array([ 8.80022565e-02,  2.73673938e+00,  3.20028421e+00,  0.00000000e+00,
            4.84073507e-07,  2.37793203e-07, -1.22000040e+00,  5.56829241e-01,
            2.59461962e-01, -1.18160196e+00,  2.66229031e+00, -5.89841123e-01,
            3.16182516e+00,  5.48000000e-01,  1.22000017e+00,  2.00000029e-01,
            2.51999981e+00, -1.38000008e+00,  8.55125873e-07,  0.00000000e+00,
            -3.72547766e-04,  1.22174095e-01])
        else:
            self.q_goal = theta_adjusted_pos(self.solve_ik(self.X_WS, self.gripper_name))
        pr2_problem = PR2_Problem(self.pr2, query_object, plant_context, self.q_init, self.q_goal)
        self.path = self.rrt_planning(pr2_problem, self.max_iterations, self.prob_sample_goal)
        self.path = self.path[1:] if self.path[0] == self.path[1] else self.path
        
        self.rrt_ix = 0

    def run(self, prev_command: Command):
        t = self.time - self.start_time
        done = False
        
        if self.rrt_ix + 1 < len(self.path) and matching_q_holding_obj(self.continuous_state[:self.pr2.num_joints()], 
                                                                       self.path[self.rrt_ix], 
                                                                       self.gripper_name, atol=8e-2):
            self.rrt_ix += 1
            self.pr2 = PR2(self.playground.construct_welded_sim_with_object_welded(
                continuous_state=self.continuous_state,
                frame_name_to_weld=self.gripper_base_link_name,
                mb=self.movable_body,
            ))
            plant_context = self.pr2.get_fresh_plant_context()
            q_start = self.pr2.plant.GetPositions(plant_context)
            q_end = np.array(self.path[self.rrt_ix])
            self.traj_opt = self.init_traj_opt(q_start, q_end, # max_t=scaled_max_t,
                                               start=True if self.rrt_ix - 1 == 0 else False,
                                               end=True if self.rrt_ix + 1 == len(self.path) else False)
            
            gridpts = Toppra.CalcGridPoints(self.traj_opt, CalcGridPointsOptions()).reshape(-1, 1)
            self.toppra_path = Toppra(self.traj_opt, self.pr2.plant, gridpts).SolvePathParameterization()

        if matching_q_holding_obj(self.continuous_state[:self.pr2.num_joints()], self.path[-1], 
                                  self.gripper_name, atol=8e-2):
            done = True
        
        return prev_command.new_command_ignore_grippers(self.traj_opt.value(self.toppra_path.value(t))), done

    def init_traj_opt(self, q_start, q_end, min_t=0.5, max_t=50.0, n_ctrl_pts=10, eps=5e-2, avoid_collisions=True, start=False, end=False):
        plant_context = self.pr2.get_fresh_plant_context()
        solver = SnoptSolver()
        num_q = self.pr2.num_joints()
        current_robot_pos = self.pr2.plant.GetPositions(plant_context)[:num_q]
        
        traj_opt = KinematicTrajectoryOptimization(num_q, num_control_points=n_ctrl_pts)
        prog = traj_opt.get_mutable_prog()
        traj_opt.AddDurationCost(10.0)
        traj_opt.AddPositionBounds(
            self.pr2.plant.GetPositionLowerLimits()[:num_q], 
            self.pr2.get_bounded_pos_upper_lim_wrist_adj()
        )
        traj_opt.AddVelocityBounds(
            self.pr2.get_bounded_velocity_lower_limit(),
            self.pr2.get_bounded_velocity_upper_limit()
        )
        
        traj_opt.AddDurationConstraint(min_t, max_t)

        traj_opt.AddPathPositionConstraint(q_start-eps, q_start+eps, 0)
        prog.AddQuadraticErrorCost(
            np.eye(num_q), current_robot_pos, traj_opt.control_points()[:, 0]
        )

        traj_opt.AddPathPositionConstraint(q_end-eps, q_end+eps, 1)
        prog.AddQuadraticErrorCost(
            np.eye(num_q), current_robot_pos, traj_opt.control_points()[:, -1]
        )

        # Solve once without the collisions and set that as the initial guess for
        # the version with collisions.
        opts = SolverOptions()
        opts.SetOption(solver.id(), "minor feasibility tolerance", 1e-6)
        result = solver.Solve(prog, solver_options=opts)
        if not result.is_success():
            print("traj opt failed: no collision checking!")
            print("infeasible constraints", result.GetInfeasibleConstraintNames(prog))
            print("traj_opt solver name", result.get_solver_id().name())
            print('q_start', q_start)
            print("q_end", q_end)
        traj_opt.SetInitialGuess(traj_opt.ReconstructTrajectory(result))

        if avoid_collisions:
            # collision constraints
            collision_constraint = MinimumDistanceLowerBoundConstraint(
                self.pr2.plant, 0.01, plant_context, None, 0.1
            )
            evaluate_at_s = np.linspace(0, 1, 25)
            for s in evaluate_at_s:
                traj_opt.AddPathPositionConstraint(collision_constraint, s)

            result = Solve(prog, solver_options=opts)
            if not result.is_success():
                print("traj opt failed: with collision checking!")
                print("infeasible constraints", result.GetInfeasibleConstraintNames(prog))
                print("collisions", min_distance_collision_checker(self.pr2.plant, plant_context, 0.01))
                print("traj opt solver name", result.get_solver_id().name())
       
        return traj_opt.ReconstructTrajectory(result)

    def solve_ik(self, X_WG, gripper_name, max_tries=10):
        ik_context = self.pr2.get_fresh_plant_context()
        ik = InverseKinematics(self.pr2.plant, ik_context, with_joint_limits=True)
        q_variables = ik.q()  # Get variables for MathematicalProgram
        prog = ik.prog()  # Get MathematicalProgram
        solver = SnoptSolver()
        
        nominal_error = q_variables[3:] - self.pr2.get_robot_nominal_position()[3:]
        prog.AddCost(nominal_error @ nominal_error)
        
        # goal frames
        p_X_WG = X_WG.translation()
        pos_offset = 0.08*np.ones_like(p_X_WG)
        R_WG = X_WG.rotation()
        gripper_offset = np.array([0.0, 0., 0.])
        ee_pos = ik.AddPositionConstraint(
                frameA=self.pr2.plant.world_frame(),
                frameB=self.pr2.plant.GetFrameByName(gripper_name),
                p_BQ=gripper_offset,
                p_AQ_lower=p_X_WG - pos_offset,
                p_AQ_upper=p_X_WG + pos_offset,
            )
        ee_rot = ik.AddOrientationConstraint(
                frameAbar=self.pr2.plant.world_frame(),
                R_AbarA=R_WG,
                frameBbar=self.pr2.plant.GetFrameByName(gripper_name),
                R_BbarB=R_WG.MakeZRotation(np.pi),
                theta_bound=np.pi/20.0,
            )
        ee_pos.evaluator().set_description("EE Position Constraint")
        ee_rot.evaluator().set_description("EE Rotation Constraint")

        min_bound = ik.AddMinimumDistanceLowerBoundConstraint(0.03)
        min_bound.evaluator().set_description("Minimum Distance Lower Bound Constraint")
        
        for count in range(max_tries):
            # Compute a random initial guesses
            ub = np.array(get_pr2_non_base_upper_bounds(self.pr2.plant, self.pr2.model_id))
            lb = np.array(get_pr2_non_base_lower_bounds(self.pr2.plant, self.pr2.model_id))
            upper_lim = np.where(ub == np.inf, np.pi, ub)
            lower_lim = np.where(lb == -np.inf, -np.pi, lb)
            
            rands = (upper_lim - lower_lim)*np.random.uniform() + lower_lim
            current_robot_pos = theta_adjusted_pos(self.pr2.plant.GetPositions(ik_context))
            for i, ix in enumerate(get_ik_joints_ix(self.pr2.plant, self.pr2.model_id)):
                prog.SetInitialGuess(q_variables[ix], (current_robot_pos[ix] + rands[i]))

            # solve the optimization and keep the first successful one
            opts = SolverOptions()
            opts.SetOption(solver.id(), "minor feasibility tolerance", 1e-3)
            result = solver.Solve(prog, solver_options=opts)
            if result.is_success():
                print("IK succeeded in %d tries!" % (count + 1))
                print("ik solution", result.GetSolution(q_variables))
                return result.GetSolution(q_variables)

        assert result.is_success(), "IK failed!"

    def rrt_planning(self, problem, max_iterations, prob_sample_q_goal):
        """
        Input:
            problem: instance of a utility class
            max_iterations: the maximum number of samples to be collected
            prob_sample_q_goal: the probability of sampling q_goal

        Output:
            path (list): [q_start, ...., q_goal].
                        Note q's are configurations, not RRT nodes
        """
        rrt_tools = RRT_tools(problem)
        q_goal = problem.goal
        q_start = problem.start

        for k in range(max_iterations):
            q_sample = rrt_tools.sample_node_in_configuration_space()
            random_num = random()
            if random_num < prob_sample_q_goal:
                q_sample = q_goal
                n_near = rrt_tools.find_nearest_node_in_RRT_graph(q_sample)
                intermediate_q = rrt_tools.calc_intermediate_qs_wo_collision(n_near.value, q_sample)
                last_node = n_near
                for n in range(len(intermediate_q)):
                    last_node = rrt_tools.grow_rrt_tree(last_node, intermediate_q[n])
                    if rrt_tools.node_reaches_goal(last_node):
                        path = rrt_tools.backup_path_from_node(last_node)
                        return path

        return None

class PR2_Problem(Problem):
    def __init__(self, pr2, query_object, plant_context, q_start: np.array, q_goal: np.array):
        self.pr2 = pr2
        self.plant_context = plant_context
        num_q = pr2.num_actuated_joints()

        lb = pr2.plant.GetPositionLowerLimits()[:22]
        ub = pr2.plant.GetPositionUpperLimits()[:22]
        q_start = np.clip(q_start.tolist(), lb, ub)
        q_goal = np.clip(q_goal.tolist(), lb, ub)

        
        range_list = []
        for i in range(lb.shape[0]):
            if i == 2:
                range_list.append(Range(0, 2*np.pi))
            else:
                range_list.append(Range(lb[i], ub[i]))
        max_steps = num_q * [np.pi / 180 * 2]  # three degrees
        cspace_pr2 = ConfigurationSpace(range_list, l2_distance, max_steps)

        # override 
        Problem.__init__(self,
                           x=100,
                           y=100,
                           robot=None,
                           obstacles=None,
                           start=tuple(q_start),
                           goal=tuple(q_goal),
                           region=AABB(Point(-100, -100), Point(100, 100)),
                           cspace=cspace_pr2,
                           display_tree=False)
        
    
    def collide(self, configuration):
        q = np.array(configuration)
        return self.exists_collision()
    
    def exists_collision(self):
        query_object = self.pr2.plant.get_geometry_query_input_port().Eval(self.plant_context)
        inspector = query_object.inspector()
        collision_pairs = inspector.GetCollisionCandidates()
        for pair in collision_pairs:
            val = query_object.ComputeSignedDistancePairClosestPoints(pair[0], pair[1]).distance, inspector.GetName(inspector.GetFrameId(pair[0])), inspector.GetName(inspector.GetFrameId(pair[1]))
            if val[0] <= -1e-4: # slack to account for shelve and brick collision
                return True
        return False
