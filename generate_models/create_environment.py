import warnings
from typing import List

import numpy as np

from pydrake.geometry import MeshcatVisualizer, MeshcatVisualizerParams, GeometrySet, Role, CollisionFilterDeclaration
from pydrake.all import AddMultibodyPlantSceneGraph, Parser
from manipulation.station import load_scenario
from pydrake.multibody.tree import Body
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import DiagramBuilder, Diagram, Context
from pydrake.systems.primitives import PassThrough, Demultiplexer, Adder, \
    StateInterpolatorWithDiscreteDerivative, LogVectorOutput, Multiplexer, ZeroOrderHold
from pydrake.visualization import ApplyVisualizationConfig, VisualizationConfig

from environment.fixed_body import FixedBody, Shelve
from settings import DEFAULT_ENV_URL, PR2_MODEL_URL, PR2_MAIN_LINK, GRIPPER_MODEL_URL
from pydrake.all import MultibodyPlant

from perception.compute_point_cloud import GetCachedModelPointCloud
from .create_environment_files import create_environment_files
from environment.movable_body import MovableBody
from grasping.compute_grasp_candidates import GetCachedGraspCandidates
from package_utils import CustomConfigureParser, GetPathFromUrl
from utils import filterPR2CollsionGeometry
from environment import Environment
from .parameters import SHELVE_BODY_NAME, SHELF_FLOORS, SHELF_THICKNESS, SHELF_DEPTH, SHELF_WIDTH, SHELF_HEIGHT


def CustomParser(plant):
    parser = Parser(plant)
    return CustomConfigureParser(parser)


def AddPR2Plant(plant):
    pr2 = CustomParser(plant).AddModelsFromUrl(PR2_MODEL_URL)[0]
    plant.WeldFrames(plant.world_frame(), plant.GetBodyByName(PR2_MAIN_LINK).body_frame())
    return pr2


def AddPR2(builder, plant: MultibodyPlant, scene_graph):
    pr2 = AddPR2Plant(plant)
    filterPR2CollsionGeometry(scene_graph)
    return pr2


def fixPR2GripperCollisionWithObjectInGripper(scene_graph, body_name):
    # todo later maybe generalize this instead of copying it from utils?
    filter_manager = scene_graph.collision_filter_manager()
    inspector = scene_graph.model_inspector()
    pr2 = {}
    exclude_bodies = []
    for gid in inspector.GetGeometryIds(
            GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity
    ):
        gid_name = inspector.GetName(inspector.GetFrameId(gid))
        if "pr2" in gid_name:
            link_name = gid_name.split("::")[1]
            pr2[link_name] = [gid]
        if body_name in gid_name:
            exclude_bodies.append(gid)

    # print('exclude bodies:')
    # print(exclude_bodies)

    def add_exclusion(set1, set2=None):
        if set2 is None:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeWithin(GeometrySet(set1))
            )
        else:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeBetween(
                    GeometrySet(set1), GeometrySet(set2)
                )
            )

    # todo this excludes both hands. what if we want to exclude only one gripper?
    add_exclusion(exclude_bodies,
                  pr2["l_gripper_palm_link"] +
                  pr2["r_gripper_palm_link"] +
                  pr2["l_gripper_r_finger_link"] +
                  pr2["l_gripper_l_finger_link"] +
                  pr2["r_gripper_r_finger_link"] +
                  pr2["r_gripper_l_finger_link"] +
                  pr2["l_gripper_l_finger_tip_link"] +
                  pr2["l_gripper_r_finger_tip_link"] +
                  pr2["r_gripper_l_finger_tip_link"] +
                  pr2["r_gripper_r_finger_tip_link"]
    )


def GetPR2NominalPosition(plant, pr2_model_id):
    default_positions = []
    for val, name in zip(plant.GetDefaultPositions(pr2_model_id), plant.GetPositionNames(pr2_model_id)):
        if name == 'l_shoulder_pan_joint_q':
            val = 1.22
        if name == 'r_shoulder_pan_joint_q':
            val = -1.22
        if name == 'l_gripper_l_finger_joint_q':
            val = 0.55
        if name == 'r_gripper_l_finger_joint_q':
            val = 0.55
        if name == 'l_shoulder_lift_joint_q':
            val = 0.2
        if name == 'r_shoulder_lift_joint_q':
            val = 0.2
        if name == 'l_upper_arm_roll_joint_q':
            val = 2.52
        if name == 'r_upper_arm_roll_joint_q':
            val = -2.52
        if name == 'l_elbow_flex_joint_q':
            val = -1.38
        if name == 'r_elbow_flex_joint_q':
            val = -1.38
        default_positions.append(val)
    return np.array(default_positions).astype(np.float64)


def SetDefaultPR2NominalPosition(plant, pr2_model_id):
    default_context = plant.CreateDefaultContext()
    default_positions = GetPR2NominalPosition(plant, pr2_model_id)
    plant.SetPositions(default_context, pr2_model_id, default_positions)
    plant.SetDefaultPositions(pr2_model_id, default_positions)


def fix_order(builder, port, cur_order, final_order):
    n = len(cur_order)
    assert (n == len(final_order))
    assert (n == port.size())

    def core_name(name: str):
        name = name.removeprefix("control_")
        name = name.removeprefix("pr2_")
        name = name.removesuffix("_q")
        name = name.removesuffix("_x")
        name = name.removesuffix("_y")
        name = name.removesuffix("_joint")
        name = name.removesuffix("_motor")
        return name

    final_order = [core_name(name) for name in final_order]
    cur_order = [core_name(name) for name in cur_order]

    final_order_map = {name: i for i, name in enumerate(final_order)}
    assert (len(final_order_map) == n)
    for name in cur_order:
        assert (name in final_order_map)

    demux = builder.AddSystem(Demultiplexer(n, 1))
    mux = builder.AddSystem(Multiplexer(n))
    builder.Connect(port, demux.get_input_port())
    for i, name in enumerate(cur_order):
        builder.Connect(
            demux.get_output_port(i),
            mux.get_input_port(final_order_map[name])
        )
    return mux.get_output_port()


def AddPr2Controller(builder, plant, model, model_name, state_port, position_command_port, feed_forward_torque_port):
    num_positions = plant.num_positions(model)

    controller_plant: MultibodyPlant = MultibodyPlant(time_step=plant.time_step())
    controller_model = AddPR2Plant(controller_plant)
    controller_plant.RenameModelInstance(controller_model, "control_pr2")
    controller_plant.Finalize()

    kp = [(600 if 'finger_joint' in name else 100)  # gripper will explode with ki=1 when we want to close it
          for name in controller_plant.GetPositionNames()]
    ki = [(0 if 'finger_joint' in name else 1)  # gripper will explode with ki=1 when we want to close it
          for name in controller_plant.GetPositionNames()]
    kd = [20] * num_positions

    controller: InverseDynamicsController = builder.AddSystem(
        InverseDynamicsController(  # todo use another controller + tune the gains?
            controller_plant,
            kp=kp,
            ki=ki,
            kd=kd,
            has_reference_acceleration=False,
        )
    )
    controller.set_name(model_name + ".controller")
    builder.Connect(
        state_port,
        controller.get_input_port_estimated_state(),
    )

    # Add in the feed-forward torque
    adder = builder.AddSystem(Adder(2, num_positions))
    builder.Connect(
        controller.get_output_port_control(),
        adder.get_input_port(0),
    )

    # Add discrete derivative to command velocities.
    # this calculated velocity_desired based on commanded positions
    # if we decide to pass velocity as well as position we don't need this
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_positions,
            plant.time_step(),
            suppress_initial_transient=True,
        )
    )
    desired_state_from_position.set_name(
        model_name + ".desired_state_from_position"
    )
    builder.Connect(
        feed_forward_torque_port, adder.get_input_port(1)
    )
    builder.Connect(
        desired_state_from_position.get_output_port(),
        controller.get_input_port_desired_state(),
    )
    builder.Connect(
        position_command_port,
        desired_state_from_position.get_input_port(),
    )

    # todo we can remove the loggers if not needed
    logger_controller = LogVectorOutput(builder=builder, src=controller.get_output_port_control(), publish_period=0.001)
    logger_controller.set_name("logger_controller")

    logger_state = LogVectorOutput(builder=builder, src=state_port, publish_period=0.001)
    logger_state.set_name("logger_state")

    logger_des_state = LogVectorOutput(builder=builder, src=desired_state_from_position.get_output_port(),
                                       publish_period=0.001)
    logger_des_state.set_name("logger_des_state")

    # the output order is based on position, we want to convert it to torque
    # basically applying tau = B u
    output = fix_order(
        builder=builder,
        port=adder.get_output_port(),
        cur_order=controller_plant.GetPositionNames(),
        final_order=controller_plant.GetActuatorNames())
    return output


def AddPr2Driver(builder, plant: MultibodyPlant, scene_graph, model):
    num_positions = plant.num_positions(model)
    model_name = "pr2"

    positions = builder.AddSystem(PassThrough(GetPR2NominalPosition(plant, model)))
    builder.ExportInput(
        positions.get_input_port(),
        model_name + ".position",
    )
    builder.ExportOutput(
        positions.get_output_port(),
        model_name + ".position_commanded",
    )

    demux = builder.AddSystem(
        Demultiplexer(2 * num_positions, num_positions)
    )

    builder.Connect(
        plant.get_state_output_port(model),
        demux.get_input_port(),
    )
    builder.ExportOutput(
        demux.get_output_port(0),
        model_name + ".position_measured",
    )
    builder.ExportOutput(
        demux.get_output_port(1),
        model_name + ".velocity_estimated",
    )
    builder.ExportOutput(
        plant.get_state_output_port(model),
        model_name + ".state_estimated",
    )

    # Use a PassThrough to make the port optional (it will provide zero
    # values if not connected).
    torque_passthrough = builder.AddSystem(
        PassThrough([0] * num_positions)
    )
    builder.ExportInput(
        torque_passthrough.get_input_port(),
        model_name + ".feedforward_torque",
    )

    controller_port = AddPr2Controller(
        builder,
        plant,
        model,
        model_name,
        state_port=plant.get_state_output_port(model),
        position_command_port=positions.get_output_port(),
        feed_forward_torque_port=torque_passthrough.get_output_port()
    )

    builder.Connect(
        controller_port,
        plant.get_actuation_input_port(model),
    )

    # Export commanded torques.
    builder.ExportOutput(
        torque_passthrough.get_output_port(),
        model_name + ".feed_forward_torque_commanded",
    )
    builder.ExportOutput(
        controller_port,
        model_name + ".torque_commanded",
    )

    # we have to add this ZeroOrderHold in order to break the algebraic loop between controller and plant
    torque_passthrough = builder.AddSystem(
        ZeroOrderHold(vector_size=plant.get_generalized_contact_forces_output_port(model).size(), period_sec=0.0001))
    builder.Connect(
        plant.get_generalized_contact_forces_output_port(model),
        torque_passthrough.get_input_port(),
    )
    builder.ExportOutput(
        torque_passthrough.get_output_port(),
        model_name + ".torque_external",
    )


def AddThingsDefaultPos(plant, movable_bodies: List[MovableBody]):
    for mb in movable_bodies:
        plant.SetDefaultFreeBodyPose(mb.get_body(plant), mb.X_WO_init)


def GetAllMovableBodies(plant, yaml_file, reset_cache):
    default_context = plant.CreateDefaultContext()

    directives = load_scenario(filename=yaml_file).directives
    model_name_to_url = dict()
    for d in directives:
        if d.add_model:
            model_name_to_url[d.add_model.name] = d.add_model.file
    cached_pc = dict()
    cached_grasp_candidates = dict()
    movable_bodies = []
    for body_index in plant.GetFloatingBaseBodies():
        body = plant.get_body(body_index)
        model_name = plant.GetModelInstanceName(body.model_instance())
        model_url = model_name_to_url[model_name]
        if model_url not in cached_pc:
            cached_pc[model_url] = GetCachedModelPointCloud(model_url, reset_cache)
        pc = cached_pc[model_url]
        if model_url not in cached_grasp_candidates:
            cached_grasp_candidates[model_url] = GetCachedGraspCandidates(model_url, reset_cache)
        grasp_candidates = cached_grasp_candidates[model_url]

        model_name = plant.GetModelInstanceName(body.model_instance())
        origin_frame = plant.GetFrameByName(model_name + "_origin")
        final_frame = plant.GetFrameByName(model_name + "_destination")

        movable_bodies.append(MovableBody(
            body_name=body.name(),
            model_name=model_name,
            point_cloud=pc,
            X_WO_init=origin_frame.CalcPoseInWorld(default_context),
            X_WO_end=final_frame.CalcPoseInWorld(default_context),
            X_OF_grasp_candidates=grasp_candidates
        ))
    return movable_bodies


def GetAllFixedBodies(plant: MultibodyPlant) -> List[FixedBody]:
    default_context = plant.CreateDefaultContext()

    # todo this is the hacky way. just take all the objects we want by name...
    fixed_bodies = []
    for body in plant.GetBodiesWeldedTo(plant.world_body()):
        # todo using SHELVE_DEPTH, ... is bad because then we have to regenerate the models all the time if we change those parameters...
        # technical debt
        if body.name() != SHELVE_BODY_NAME:
            continue
        body: Body = body
        fixed_bodies.append(Shelve(
            body_name=body.name(),
            model_name=plant.GetModelInstanceName(body.model_instance()),
            depth=SHELF_DEPTH,
            width=SHELF_WIDTH,
            height=SHELF_HEIGHT,
            floors=SHELF_FLOORS,
            thickness=SHELF_THICKNESS,
            normal_direction=body.body_frame().CalcPoseInWorld(default_context).rotation() @ [1, 0, 0],
            center_pos=body.body_frame().CalcPoseInWorld(default_context).translation()
        ))
    return fixed_bodies


class Playground:
    def __init__(
            self,
            meshcat,
            env_yaml_file_url=DEFAULT_ENV_URL,
            time_step=0.001,
            regenerate_sdf_files=True,
            reset_cache=False,
            visualization_config=VisualizationConfig()
    ):
        if regenerate_sdf_files:
            create_environment_files()

        builder = DiagramBuilder()

        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
        plant: MultibodyPlant = plant # what's the point of this line?

        parser = CustomParser(plant)
        parser.AddModelsFromUrl(env_yaml_file_url)

        plant.set_name("plant")

        pr2_model_id = AddPR2(builder, plant, scene_graph)
        plant.Finalize()

        SetDefaultPR2NominalPosition(plant, pr2_model_id)

        env_yaml_file_path = GetPathFromUrl(env_yaml_file_url, parser)
        movable_bodies = GetAllMovableBodies(plant, env_yaml_file_path, reset_cache=reset_cache)
        point_clouds = {mb.get_body(plant).index(): mb.point_cloud for mb in movable_bodies}
        fixed_bodies = GetAllFixedBodies(plant)

        AddThingsDefaultPos(plant, movable_bodies)

        AddPr2Driver(builder, plant, scene_graph, pr2_model_id)

        self._add_visuals(builder, scene_graph, meshcat, visualization_config)

        # # Export "cheat" ports.
        builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
        builder.ExportOutput(
            plant.get_contact_results_output_port(), "contact_results"
        )
        builder.ExportOutput(
            plant.get_state_output_port(), "plant_continuous_state"
        )
        builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

        diagram = builder.Build()
        diagram.set_name("environment")

        self.meshcat = meshcat

        def context_update_function(env: Environment, sim_context: Context, continuous_state: np.ndarray) -> Context:
            sim_plant_context = env.plant.GetMyContextFromRoot(sim_context)
            env.plant.SetPositionsAndVelocities(sim_plant_context, continuous_state)
            return sim_context

        self.env = Environment(diagram=diagram,
                               plant=plant,
                               scene_graph=scene_graph,
                               model_id=pr2_model_id,
                               point_clouds=point_clouds,
                               floating_bodies=[plant.get_body(body_index) for body_index in plant.GetFloatingBaseBodies()],
                               context_update_function=context_update_function,
                               movable_bodies=movable_bodies,
                               fixed_bodies=fixed_bodies
        )
        self.env_yaml_file_url = env_yaml_file_url
        self.time_step = time_step

    def default_continuous_state(self):
        default_plant_context = self.env.plant.CreateDefaultContext()
        return self.env.plant.GetPositionsAndVelocities(default_plant_context)

    def construct_welded_sim(self, continuous_state, modify_default_context=True, meshcat=None):
        """
        constructs a plant and scene_graph that will be used by robot's controller
        the idea is that the plant that simulates the world should be different from
        the plant that robot uses to solve trajectory optimization...

        In this new plant we include objects but weld all of them so that
        dof plant == dof robot

        pitfall. This does not work if we have non single rigid body objects (like a chain)
        if you want to support that we have to somehow fix all the joints possible
        """

        sim_builder = DiagramBuilder()
        sim_plant, sim_scene_graph = AddMultibodyPlantSceneGraph(sim_builder, time_step=self.time_step)
        sim_parser = CustomParser(sim_plant)
        sim_parser.AddModelsFromUrl(self.env_yaml_file_url)
        sim_plant.set_name("plant")
        sim_pr2_model_id = AddPR2(sim_builder, sim_plant, sim_scene_graph)

        # weld everything instead of adding default pos
        """
            todo for the future have the option to weld the object that robot is
            holding to the robot's arm
        """

        sim_plant: MultibodyPlant = sim_plant
        plant_context = self.env.plant.CreateDefaultContext()
        self.env.plant.SetPositionsAndVelocities(plant_context, continuous_state)

        # weld everything to the world
        floating_bodies = self._weld_all_floating_bodies(sim_plant, plant_context)

        sim_plant.Finalize()

        SetDefaultPR2NominalPosition(sim_plant, sim_pr2_model_id)

        if modify_default_context:
            # set default pos of robot to current position
            # sim_default_context = sim_plant.CreateDefaultContext()
            positions = self.env.plant.GetPositionsFromArray(
                model_instance=self.env.model_id,
                q=self.env.plant.GetPositions(plant_context))
            # velocities = self.env.plant.GetVelocitiesFromArray(
            #     model_instance=self.env.model_id,
            #     v=self.env.plant.GetVelocities(plant_context))
            # todo how to set default velocity as well?
            sim_plant.SetDefaultPositions(positions)

        AddPr2Driver(sim_builder, sim_plant, sim_scene_graph, sim_pr2_model_id)

        self._add_visuals(sim_builder, sim_scene_graph, meshcat)

        sim_diagram = sim_builder.Build()
        sim_diagram.set_name("environment_welded")

        def context_update_function(env: Environment, sim_context: Context, continuous_state: np.ndarray) -> Context:
            self.env.plant.SetPositionsAndVelocities(plant_context, continuous_state)
            pr2_position_and_velocity = self.env.plant.GetPositionsAndVelocities(plant_context, model_instance=self.env.model_id)
            sim_plant_context = env.plant.GetMyContextFromRoot(sim_context)
            env.plant.SetPositionsAndVelocities(sim_plant_context, pr2_position_and_velocity)
            return sim_context

        return Environment(
            diagram=sim_diagram,
            plant=sim_plant,
            scene_graph=sim_scene_graph,
            model_id=sim_pr2_model_id,
            point_clouds=self._move_point_clouds_to_default(sim_diagram, sim_plant, floating_bodies),
            floating_bodies=floating_bodies,
            context_update_function=context_update_function,
            movable_bodies=self.env.movable_bodies,
            fixed_bodies=self.env.fixed_bodies
        )

    def construct_welded_sim_with_object_welded(self, continuous_state, frame_name_to_weld: str, mb: MovableBody, modify_default_context=True, meshcat=None):
        sim_builder = DiagramBuilder()
        sim_plant, sim_scene_graph = AddMultibodyPlantSceneGraph(sim_builder, time_step=self.time_step)
        sim_parser = CustomParser(sim_plant)
        sim_parser.AddModelsFromUrl(self.env_yaml_file_url)
        sim_plant.set_name("plant")
        sim_pr2_model_id = AddPR2(sim_builder, sim_plant, sim_scene_graph)

        sim_plant: MultibodyPlant = sim_plant
        plant_context = self.env.plant.CreateDefaultContext()
        self.env.plant.SetPositionsAndVelocities(plant_context, continuous_state)

        # weld everything to the world except for mb that we weld to frame
        floating_bodies = []
        for other_mb in self.env.movable_bodies:
            if other_mb.is_same(mb):
                X_WF = self.env.plant.GetFrameByName(frame_name_to_weld).CalcPoseInWorld(plant_context)
                X_WO = other_mb.get_pose(self.env.plant, plant_context)
                X_FO = X_WF.inverse() @ X_WO
                sim_plant.WeldFrames(sim_plant.GetFrameByName(frame_name_to_weld),
                                     other_mb.get_body(sim_plant).body_frame(),
                                     X_FO)
            else:
                X_WO = other_mb.get_pose(self.env.plant, plant_context)
                sim_plant.WeldFrames(sim_plant.world_frame(),
                                     other_mb.get_body(sim_plant).body_frame(),
                                     X_WO)
                floating_bodies.append(other_mb.get_body(sim_plant))

        # ignore collisions with current object
        fixPR2GripperCollisionWithObjectInGripper(sim_scene_graph, mb.body_name)

        sim_plant.Finalize()

        SetDefaultPR2NominalPosition(sim_plant, sim_pr2_model_id)

        if modify_default_context:
            # set default pos of robot to current position
            # sim_default_context = sim_plant.CreateDefaultContext()
            positions = self.env.plant.GetPositionsFromArray(
                model_instance=self.env.model_id,
                q=self.env.plant.GetPositions(plant_context))
            # velocities = self.env.plant.GetVelocitiesFromArray(
            #     model_instance=self.env.model_id,
            #     v=self.env.plant.GetVelocities(plant_context))
            # todo how to set default velocity as well?
            sim_plant.SetDefaultPositions(positions)

        AddPr2Driver(sim_builder, sim_plant, sim_scene_graph, sim_pr2_model_id)

        self._add_visuals(sim_builder, sim_scene_graph, meshcat)

        sim_diagram = sim_builder.Build()
        sim_diagram.set_name("environment_welded")

        def context_update_function(env: Environment, sim_context: Context, continuous_state: np.ndarray) -> Context:
            self.env.plant.SetPositionsAndVelocities(plant_context, continuous_state)
            pr2_position_and_velocity = self.env.plant.GetPositionsAndVelocities(plant_context, model_instance=self.env.model_id)
            sim_plant_context = env.plant.GetMyContextFromRoot(sim_context)
            env.plant.SetPositionsAndVelocities(sim_plant_context, pr2_position_and_velocity)
            return sim_context

        return Environment(
            diagram=sim_diagram,
            plant=sim_plant,
            scene_graph=sim_scene_graph,
            model_id=sim_pr2_model_id,
            point_clouds=self._move_point_clouds_to_default(sim_diagram, sim_plant, floating_bodies),
            floating_bodies=floating_bodies,
            context_update_function=context_update_function,
            movable_bodies=self.env.movable_bodies,
            fixed_bodies=self.env.fixed_bodies
        )

    def construct_welded_sim_with_gripper(self, continuous_state, gripper_model_url=GRIPPER_MODEL_URL, meshcat=None):
        sim_builder = DiagramBuilder()
        sim_plant, sim_scene_graph = AddMultibodyPlantSceneGraph(sim_builder, time_step=self.time_step)
        sim_parser = CustomParser(sim_plant)
        sim_parser.AddModelsFromUrl(self.env_yaml_file_url)
        sim_plant.set_name("plant")

        plant_context = self.env.plant.CreateDefaultContext()
        self.env.plant.SetPositionsAndVelocities(plant_context, continuous_state)
        floating_bodies = self._weld_all_floating_bodies(sim_plant, plant_context)

        # add gripper
        gripper_model = CustomParser(sim_plant).AddModelsFromUrl(gripper_model_url)[0]

        sim_plant.Finalize()

        self._add_visuals(sim_builder, sim_scene_graph, meshcat)

        sim_diagram = sim_builder.Build()
        sim_diagram.set_name("environment_welded_with_gripper")

        def context_update_function(env: Environment, sim_context: Context, continuous_state: np.ndarray) -> Context:
            warnings.warn("you should not be calling this function. This is trying to set context of environment "
                          "'construct_welded_sim_with_gripper' in which all objects are welded")
            return sim_context

        return Environment(
            diagram=sim_diagram,
            plant=sim_plant,
            scene_graph=sim_scene_graph,
            model_id=gripper_model,
            point_clouds=self._move_point_clouds_to_default(sim_diagram, sim_plant, floating_bodies),
            floating_bodies=floating_bodies,
            context_update_function=context_update_function,
            movable_bodies=self.env.movable_bodies,
            fixed_bodies=self.env.fixed_bodies
        )

    def construct_welded_sim_wo_robot(self, continuous_state, meshcat=None):
        sim_builder = DiagramBuilder()
        sim_plant, sim_scene_graph = AddMultibodyPlantSceneGraph(sim_builder, time_step=self.time_step)
        sim_parser = CustomParser(sim_plant)
        sim_parser.AddModelsFromUrl(self.env_yaml_file_url)
        sim_plant.set_name("plant")

        plant_context = self.env.plant.CreateDefaultContext()
        self.env.plant.SetPositionsAndVelocities(plant_context, continuous_state)
        floating_bodies = self._weld_all_floating_bodies(sim_plant, plant_context)

        sim_plant.Finalize()

        self._add_visuals(sim_builder, sim_scene_graph, meshcat)

        sim_diagram = sim_builder.Build()
        sim_diagram.set_name("environment_welded_wo_robot")

        def context_update_function(env: Environment, sim_context: Context, continuous_state: np.ndarray) -> Context:
            warnings.warn("you should not be calling this function. This is trying to set context of environment "
                          "'construct_welded_sim_wo_robot' in which all objects are welded")
            return sim_context

        return Environment(
            diagram=sim_diagram,
            plant=sim_plant,
            scene_graph=sim_scene_graph,
            model_id=None,
            point_clouds=self._move_point_clouds_to_default(sim_diagram, sim_plant, floating_bodies),
            floating_bodies=floating_bodies,
            context_update_function=context_update_function,
            movable_bodies=self.env.movable_bodies,
            fixed_bodies=self.env.fixed_bodies
        )

    def construct_pr2_alone_sim(self, continuous_state, meshcat=None):
        sim_builder = DiagramBuilder()
        sim_plant, sim_scene_graph = AddMultibodyPlantSceneGraph(sim_builder, time_step=self.time_step)
        sim_plant.set_name("plant")
        sim_pr2_model_id = AddPR2(sim_builder, sim_plant, sim_scene_graph)

        plant_context = self.env.plant.CreateDefaultContext()
        self.env.plant.SetPositionsAndVelocities(plant_context, continuous_state)

        sim_plant.Finalize()

        SetDefaultPR2NominalPosition(sim_plant, sim_pr2_model_id)

        positions = self.env.plant.GetPositionsFromArray(
            model_instance=self.env.model_id,
            q=self.env.plant.GetPositions(plant_context))
        sim_plant.SetDefaultPositions(positions)
        AddPr2Driver(sim_builder, sim_plant, sim_scene_graph, sim_pr2_model_id)

        self._add_visuals(sim_builder, sim_scene_graph, meshcat)

        sim_diagram = sim_builder.Build()
        sim_diagram.set_name("environment_pr2_alone")

        def context_update_function(env: Environment, sim_context: Context, continuous_state: np.ndarray) -> Context:
            self.env.plant.SetPositionsAndVelocities(plant_context, continuous_state)
            pr2_position_and_velocity = self.env.plant.GetPositionsAndVelocities(plant_context, model_instance=self.env.model_id)
            sim_plant_context = env.plant.GetMyContextFromRoot(sim_context)
            env.plant.SetPositionsAndVelocities(sim_plant_context, pr2_position_and_velocity)
            return sim_context

        return Environment(
            diagram=sim_diagram,
            plant=sim_plant,
            scene_graph=sim_scene_graph,
            model_id=sim_pr2_model_id,
            point_clouds=dict(),
            floating_bodies=[],
            context_update_function=context_update_function,
            movable_bodies=[],
            fixed_bodies=[]
        )

    def _weld_all_floating_bodies(self, sim_plant: MultibodyPlant, plant_context):
        floating_bodies = []
        for body_index in self.env.plant.GetFloatingBaseBodies():
            # plant ids
            body: Body = self.env.plant.get_body(body_index)
            model_instance = body.model_instance()
            model_instance_name = self.env.plant.GetModelInstanceName(model_instance)
            X_WO = self.env.plant.GetFreeBodyPose(plant_context, body)

            # sim indices
            model_instance = sim_plant.GetModelInstanceByName(model_instance_name)
            body = sim_plant.GetBodyByName(body.name(), model_instance)
            floating_bodies.append(body)
            sim_plant.WeldFrames(sim_plant.world_frame(), body.body_frame(), X_WO)
        return floating_bodies

    def _add_visuals(self, sim_builder, sim_scene_graph, meshcat=None, visualization_config=VisualizationConfig()):
        if meshcat:
            visualizer = MeshcatVisualizer.AddToBuilder(
                sim_builder,
                sim_scene_graph.get_query_output_port(),
                meshcat,
                MeshcatVisualizerParams(delete_on_initialization_event=False)
            )
            ApplyVisualizationConfig(visualization_config, sim_builder, meshcat=meshcat)
            return visualizer

    def _move_point_clouds_to_default(self, sim_diagram: Diagram, sim_plant: MultibodyPlant, floating_bodies):
        sim_diagram_context = sim_diagram.CreateDefaultContext()
        sim_plant_context = sim_plant.GetMyContextFromRoot(sim_diagram_context)
        return {body.index(): self.env.point_clouds[body.index()].transformed(body.EvalPoseInWorld(sim_plant_context))
                for body in floating_bodies}

