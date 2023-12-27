import random

import numpy as np
from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.geometry import Rgba, MeshcatVisualizerParams, MeshcatVisualizer
from pydrake.math import RotationMatrix
from pydrake.multibody import inverse_kinematics
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.solvers import Solve
from pydrake.systems.framework import DiagramBuilder
from pydrake.all import Parser
from pydrake.visualization import ApplyVisualizationConfig, VisualizationConfig

from environment import Environment
from pydrake.all import Frame, RigidTransform, PointCloud, Context
from dataclasses import dataclass

from package_utils import CustomConfigureParser
from .gripper import Gripper


@dataclass
class GraspCandidate:
    X_frame: RigidTransform
    fails: bool
    fail_reason: str
    cost: float

    @staticmethod
    def Fail(reason, suggest_X_frame=None, suggest_cost=np.inf):
        return GraspCandidate(
            X_frame=suggest_X_frame,
            fail_reason=reason,
            fails=True,
            cost=suggest_cost
        )

    @staticmethod
    def Success(X_frame, cost):
        return GraspCandidate(
            X_frame=X_frame,
            fail_reason=None,
            fails=False,
            cost=cost
        )


class GraspRecommender:
    def __init__(self, env: Environment, gripper: Gripper):
        """
        environment is necessary to perform the collision checks
        this should be the environment without robot
        model_id is the gripper's id
        every other object is welded

        gripper is treated as a rigid body object with no actuation

        gripper_base_link is used to move gripper
        the assumption is that:
        -> X is in the object that is being grasped
        -> Y is in the direction of gripper's opening
        """

        self.env = env
        self.gripper = gripper

    # todo make this gripper specific to make the code clean later...
    def optimize_grasp_with_ik(self, X_G, cloud, context, approaching_vector=None, meshcat=None) -> GraspCandidate:
        # context = context.Clone()
        # no need to clone as long as everyone is just changing the pose if the gripper

        plant = self.env.plant
        plant_context: Context = plant.GetMyContextFromRoot(context)
        self.gripper.SetPose(plant_context, X_G)
        X_frame = self.gripper.X_frame(plant_context)

        ik = inverse_kinematics.InverseKinematics(self.env.plant, plant_context, with_joint_limits=True)
        q_variables = ik.q()  # Get variables for MathematicalProgram
        init_q = plant.GetPositions(plant_context)

        prog = ik.prog()  # Get MathematicalProgram

        pos_tol = 1e-2  # 1 cm error. depending on the gripper
        rot_tol = 1e-3  # np.pi * 2/3 * 0.15
        # todo for future allow roll to change but now pitch and yaw
        eps = 1e-3
        lim_x, lim_y, lim_z = eps, pos_tol, eps
        distance_lower_bound = 1e-3  # at least 1mm far from objects
        # ik.AddPositionConstraint(
        #     frameA=self.gripper.frame,
        #     frameB=plant.world_frame(),
        #     p_BQ=(X_frame @ self.gripper.X_FG()).translation(),
        #     p_AQ_lower=(self.gripper.X_FG() @ [-lim_x, -lim_y, -lim_z]),
        #     p_AQ_upper=(self.gripper.X_FG() @ [lim_x, lim_y, lim_z]),
        # )
        # todo this is the hacky way. it assumes y axis of frame is the same as y axis of the gripper
        # which is almost true in this case. The good solution is to somehow make a meaningful frame in the gripper!
        ik.AddPositionConstraint(
            frameA=self.gripper.frame,
            frameB=plant.world_frame(),
            p_BQ=X_frame.translation(),
            p_AQ_lower=[-lim_x, -lim_y, -lim_z],
            p_AQ_upper=[lim_x, lim_y, lim_z]
        )
        ik.AddOrientationConstraint(
            frameAbar=plant.world_frame(),
            R_AbarA=X_frame.rotation(),
            frameBbar=self.gripper.frame,
            R_BbarB=RotationMatrix(),
            theta_bound=rot_tol,
        )
        ik.AddMinimumDistanceLowerBoundConstraint(distance_lower_bound)

        if approaching_vector:
            ik.AddAngleBetweenVectorsCost(
                frameA=plant.world_frame(),
                na_A=approaching_vector,
                frameB=self.gripper.frame,
                nb_B=self.gripper.X_FG().rotation() @ [1, 0, 0],
                c=20
            )

        prog.SetInitialGuess(q_variables, init_q)
        result = Solve(prog)
        if result.is_success():
            solution_q = result.GetSolution(q_variables)
            plant.SetPositions(q=solution_q, context=plant_context)
            return self.eval_candidate(X_G=self.gripper.X_WG(plant_context),
                                       cloud=cloud,
                                       context=context,
                                       approaching_vector=approaching_vector,
                                       meshcat=meshcat)
        else:
            solution_q = result.GetSolution(q_variables)
            plant.SetPositions(q=solution_q, context=plant_context)
            return GraspCandidate.Fail(
                suggest_X_frame=self.gripper.X_frame(plant_context),
                reason="IK failed to find a good solution",
                suggest_cost=result.get_optimal_cost()
            )

    def eval_candidate(self, X_G, cloud, context, approaching_vector=None, meshcat=None) -> GraspCandidate:
        """
            if approaching vector is not None then the model would penalize the grasp
            based on deviation of gripper's X axis andd approaching vector
        """
        plant = self.env.plant
        scene_graph = self.env.scene_graph
        plant_context = plant.GetMyMutableContextFromRoot(context)
        scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)

        self.gripper.SetPose(plant_context, X_G)
        X_frame = self.gripper.X_frame(plant_context=plant_context)

        # Transform cloud into gripper frame
        X_GW = X_G.inverse()
        p_GC = X_GW @ cloud.xyzs()

        # Crop to a region inside of the finger box.
        indices = self.gripper.is_in_grasp_zone(x=p_GC[0, :], y=p_GC[1, :], z=p_GC[2, :])
        if meshcat:
            pc = PointCloud(np.sum(indices))
            pc.mutable_xyzs()[:] = cloud.xyzs()[:, indices]
            meshcat.SetObject(
                "planning/points", pc, rgba=Rgba(1.0, 0, 0), point_size=0.01
            )

        query_object = scene_graph.get_query_output_port().Eval(
            scene_graph_context
        )
        # Check collisions between the gripper and the sink
        if query_object.HasCollisions():
            return GraspCandidate.Fail(
                suggest_X_frame=X_frame,
                reason="Gripper is colliding with an object in the environment",
                suggest_cost=np.inf
            )

        # Check collisions between the gripper and the point cloud
        # must be smaller than the margin used in the point cloud preprocessing.
        # todo I'm not sure if we should check with point cloud as since collision with object in the environment implies collision with the point cloud as well...
        # margin = 0.0
        # for i in range(cloud.size()):
        #     distances = query_object.ComputeSignedDistanceToPoint(
        #         cloud.xyz(i), threshold=margin
        #     )
        #     if distances:
        #         return GraspCandidate(
        #             X_frame=X_frame,
        #             fails=True,
        #             fail_reason=f"Gripper is colliding with point cloud (point {cloud.xyz(i)})",
        #             cost=np.inf
        #         )

        n_GC = X_GW.rotation().multiply(cloud.normals()[:, indices])

        # Penalize deviation of the gripper from vertical.
        # weight * -dot([0, 0, -1], R_G * [0, 1, 0]) = weight * R_G[2,1]
        # cost = 20.0 * X_G.rotation().matrix()[2, 1]
        approaching_axis = [1, 0, 0]  # x axis
        cost = 0
        if approaching_vector is not None:
            cost -= 20.0 * (X_G.rotation().matrix() @ approaching_axis) @ approaching_vector

        # Reward sum |dot product of normals with gripper x|^2
        # cost -= np.sum(n_GC[0, :] ** 2)
        cost -= np.sum(n_GC[1, :] ** 2)
        return GraspCandidate.Success(
            X_frame=X_frame,
            cost=cost
        )

    def centeralize_grasp(self, X_G, cloud, context):
        plant = self.env.plant
        scene_graph = self.env.scene_graph
        plant_context = plant.GetMyMutableContextFromRoot(context)
        scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)

        self.gripper.SetPose(plant_context, X_G)

        # Transform cloud into gripper frame
        X_GW = X_G.inverse()
        p_GC = X_GW @ cloud.xyzs()

        return X_G @ self.gripper.X_GH_centralize_wrt_point_cloud(x=p_GC[0, :], y=p_GC[1, :], z=p_GC[2, :])


    def generate_random_candidates_one_random_point(self, cloud, context: Context, approaching_vector=None,
                                                    meshcat=None):
        """
        Picks a random point in the cloud, and aligns the robot finger with the normal of that pixel.
        The rotation around the normal axis is drawn from a uniform distribution over [min_roll, max_roll].

        approaching vector is the vector along which we try to approach the object

        Returns:
            GraspCandidate object
        """
        context = context.Clone()  # defensive copying

        plant = self.env.plant
        plant_context = plant.GetMyMutableContextFromRoot(context)
        scene_graph = self.env.scene_graph
        scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)

        index = random.randint(0, cloud.size() - 1)

        # Use S for sample point/frame.
        p_WS = cloud.xyz(index)
        n_WS = cloud.normal(index)

        assert np.isclose(
            np.linalg.norm(n_WS), 1.0
        ), f"Normal has magnitude: {np.linalg.norm(n_WS)}"

        Gy = n_WS  # gripper y axis aligns with normal

        # make orthonormal x axis, aligned with world down
        # suggested that it is in the direction of approaching vector
        approaching = np.random.uniform(-1, 1, 3) if (approaching_vector is None) else approaching_vector
        Gx = approaching[:]
        Gx /= np.linalg.norm(Gx)

        Gx = Gx - (Gx @ Gy) * Gy
        if np.abs(np.linalg.norm(Gx)) < 1e-6:
            yield GraspCandidate.Fail(reason=f"the normal of the point index={index} was in the direction of approaching vector={approaching}", suggest_X_frame=RigidTransform())
            return
        Gx /= np.linalg.norm(Gx)

        Gz = np.cross(Gx, Gy)

        R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)

        if meshcat:
            AddMeshcatTriad(
                meshcat,
                path="grasping_debug",
                X_PT=RigidTransform(p=p_WS, R=R_WG)
            )

        # Try orientations from the center out
        min_roll = -np.pi / 3.0
        max_roll = np.pi / 3.0
        alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
        for theta in min_roll + (max_roll - min_roll) * alpha:
            R_WG2 = R_WG.multiply(RotationMatrix.MakeYRotation(theta))

            p_GS_G = self.gripper.p_GS_G()

            # Use G for gripper frame.
            p_SG_W = -R_WG2.multiply(p_GS_G)
            p_WG = p_WS + p_SG_W
            X_G = RigidTransform(R_WG2, p_WG)
            # final improvement round. centralize the points...
            X_G = self.centeralize_grasp(X_G=X_G, cloud=cloud, context=context)
            yield self.eval_candidate(cloud=cloud, X_G=X_G, context=context, approaching_vector=approaching_vector,
                                      meshcat=meshcat)

    def generate_random_candidates(self, cloud, context: Context, approaching_vector=None, meshcat=None):
        while True:
            for res in self.generate_random_candidates_one_random_point(cloud=cloud, context=context,
                                                                        approaching_vector=approaching_vector,
                                                                        meshcat=meshcat):
                yield res

    def generate_successful_candidates(self, max_tries, cloud, context: Context, approaching_vector=None, meshcat=None):
        gen = self.generate_random_candidates(cloud=cloud, context=context, approaching_vector=approaching_vector,
                                              meshcat=meshcat)
        return [candid
                for candid, _ in zip(gen, range(max_tries))
                if (not candid.fails)
                ]

    def draw_grasp_candidate(self, X_frame: RigidTransform, model_url, meshcat, prefix="gripper", draw_frames=True):
        # for visualizing canidate grasps
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        parser: Parser = CustomConfigureParser(parser)
        parser.AddModelsFromUrl(model_url)
        plant.WeldFrames(plant.world_frame(),
                         plant.GetBodyByName(self.gripper.base_link_name).body_frame(), X_frame)
        plant.Finalize()

        # frames_to_draw = {"gripper": {"body"}} if draw_frames else {}
        visualizer = MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph.get_query_output_port(),
            meshcat,
            MeshcatVisualizerParams(delete_on_initialization_event=False, delete_prefix_on_initialization_event=False,
                                    prefix=prefix)
        )
        ApplyVisualizationConfig(VisualizationConfig(), builder, meshcat=meshcat)

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        diagram.ForcedPublish(context)
