import os
import pickle

from manipulation.scenarios import AddRgbdSensors
from package_utils import CustomConfigureParser, GetModelNameFromUrl
from pydrake.all import DiagramBuilder, AddMultibodyPlantSceneGraph, Parser, Concatenate
from settings import CACHE_PATH
from .perception import CustomPointCloud


def GetCachedModelPointCloud(model_url, reset_cache):
    name = GetModelNameFromUrl(model_url) + '.pkl'
    path = os.path.join(CACHE_PATH, name)
    if not os.path.exists(path):
        reset_cache = True
    if reset_cache:
        pc = GetModelPointCloud(model_url)
        with open(path, 'wb') as f:
            pickle.dump(pc, f)
        return pc
    else:
        with open(path, 'rb') as f:
            pc = pickle.load(f)
        return pc


def MakeSystemForPointCloud(model_url):
    # example system for the object we want to get a point cloud of and the cameras used
    builder = DiagramBuilder()

    # Create the physics engine + scene graph.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    CustomConfigureParser(parser)
    object_name = 'object'
    yaml_string = generate_6_camera_scene_yaml(model_url=model_url, object_name=object_name)
    parser.AddModelsFromString(yaml_string, '.dmd.yaml')
    plant.Finalize()

    AddRgbdSensors(builder, plant, scene_graph)

    diagram = builder.Build()
    diagram.set_name("depth_camera_demo_system")
    return diagram


def GetModelPointCloud(model_url):
    print('calculating point cloud for ', model_url)

    # sets up our object for grasping with the cameras around it
    system = MakeSystemForPointCloud(model_url=model_url)

    plant = system.GetSubsystemByName("plant")

    # Evaluate the camera output ports to get the images.
    context = system.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    pcd = []
    # iterate through 6 cameras
    for i in range(6):
        cloud = system.GetOutputPort(f"camera{i}_point_cloud").Eval(context)

        # Crop to region of interest.
        # pcd.append(
        #     cloud.Crop(lower_xyz=[-0.3, -0.3, -0.3], upper_xyz=[0.3, 0.3, 0.3])
        # )
        # todo since the cameras are invisible, we don't need cropping

        pcd.append(cloud)
        # normal estimation
        pcd[i].EstimateNormals(radius=0.1, num_closest=30)

        camera = plant.GetModelInstanceByName(f"camera{i}")
        body = plant.GetBodyByName("base", camera)
        X_C = plant.EvalBodyPoseInWorld(plant_context, body)
        pcd[i].FlipNormalsTowardPoint(X_C.translation())

    # Merge point clouds
    merged_pcd = Concatenate(pcd)

    # Voxelize down-sample.  (Note that the normals still look reasonable)
    down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
    return CustomPointCloud(xyzs=down_sampled_pcd.xyzs(), normals=down_sampled_pcd.normals())


def generate_6_camera_scene_yaml(model_url, object_name='object'):
    # todo make sure that distances are ok with respect to the size of the object...
    base_frame = get_one_floating_body_base_frame(model_url)
    return f"""
directives:

- add_model:
    name: {object_name}
    file: {model_url}

- add_weld:
    parent: world
    child: {object_name}::{base_frame}

- add_frame:
    name: camera0_staging
    X_PF:
        base_frame: world
        rotation: !Rpy {{ deg: [0, 0, 0]}}

- add_frame:
    name: camera1_staging
    X_PF:
        base_frame: world
        rotation: !Rpy {{ deg: [0, 0, 90.0]}}

- add_frame:
    name: camera2_staging
    X_PF:
        base_frame: world
        rotation: !Rpy {{ deg: [0, 0, 180.0]}}

- add_frame:
    name: camera3_staging
    X_PF:
        base_frame: world
        rotation: !Rpy {{ deg: [0, 0, 270.0]}}

- add_frame:
    name: camera4_staging
    X_PF:
        base_frame: world
        rotation: !Rpy {{ deg: [0, -90.0, 0]}}

- add_frame:
    name: camera5_staging
    X_PF:
        base_frame: world
        rotation: !Rpy {{ deg: [0, 90.0, 0]}}

- add_frame:
    name: camera0_origin
    X_PF:
        base_frame: camera0_staging
        rotation: !Rpy {{ deg: [-90.0, 0, 90.0]}}
        translation: [.5, 0, .1]

- add_model:
    name: camera0
    file: package://Shelve_Bot/models/camera_box_invisible.sdf

- add_weld:
    parent: camera0_origin
    child: camera0::base

- add_frame:
    name: camera1_origin
    X_PF:
        base_frame: camera1_staging
        rotation: !Rpy {{ deg: [-90.0, 0, 90.0]}}
        translation: [.5, 0, .1]

- add_model:
    name: camera1
    file: package://Shelve_Bot/models/camera_box_invisible.sdf

- add_weld:
    parent: camera1_origin
    child: camera1::base

- add_frame:
    name: camera2_origin
    X_PF:
        base_frame: camera2_staging
        rotation: !Rpy {{ deg: [-90.0, 0, 90.0]}}
        translation: [.5, 0, .1]

- add_model:
    name: camera2
    file: package://Shelve_Bot/models/camera_box_invisible.sdf

- add_weld:
    parent: camera2_origin
    child: camera2::base

- add_frame:
    name: camera3_origin
    X_PF:
        base_frame: camera3_staging
        rotation: !Rpy {{ deg: [-90.0, 0, 90.0]}}
        translation: [.5, 0, .1]

- add_model:
    name: camera3
    file: package://Shelve_Bot/models/camera_box_invisible.sdf

- add_weld:
    parent: camera3_origin
    child: camera3::base

- add_frame:
    name: camera4_origin
    X_PF:
        base_frame: camera4_staging
        rotation: !Rpy {{ deg: [-90.0, 0, 90.0]}}
        translation: [.5, 0, 0]

- add_model:
    name: camera4
    file: package://Shelve_Bot/models/camera_box_invisible.sdf

- add_weld:
    parent: camera4_origin
    child: camera4::base

- add_frame:
    name: camera5_origin
    X_PF:
        base_frame: camera5_staging
        rotation: !Rpy {{ deg: [-90.0, 0, 90.0]}}
        translation: [.5, 0, 0]

- add_model:
    name: camera5
    file: package://Shelve_Bot/models/camera_box_invisible.sdf

- add_weld:
    parent: camera5_origin
    child: camera5::base
"""


def get_one_floating_body_base_frame(model_url):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    CustomConfigureParser(parser)
    parser.AddModelsFromUrl(model_url)
    plant.Finalize()

    floating_bodies = list(plant.GetFloatingBaseBodies())
    if len(floating_bodies) > 1:
        raise Exception("expecting a model with only one free body")
    body = plant.get_body(floating_bodies[0])
    base_name = body.name()
    return base_name
