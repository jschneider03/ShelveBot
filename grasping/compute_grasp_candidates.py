import os
import pickle

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.framework import DiagramBuilder
from tqdm import tqdm

from package_utils import GetModelNameFromUrl, CustomConfigureParser
from grasping import PR2Gripper, GraspRecommender
from perception import GetCachedModelPointCloud
from settings import CACHE_PATH, GRIPPER_MODEL_URL, GRIPPER_BASE_LINK, MAX_GRASP_CANDIDATE_GENERATION
from environment import Environment


def GetCachedGraspCandidates(model_url, reset_cache):
    name = GetModelNameFromUrl(model_url) + "_PR2Gripper_" + '.pkl'
    path = os.path.join(CACHE_PATH, name)
    if not os.path.exists(path):
        reset_cache = True
    if reset_cache:
        grasp_candidates = GetGraspCandidates(model_url)
        with open(path, 'wb') as f:
            pickle.dump(grasp_candidates, f)
        return grasp_candidates
    else:
        with open(path, 'rb') as f:
            grasp_candidates = pickle.load(f)
        return grasp_candidates


def GetGraspCandidates(model_url, grasp_candidate_count=MAX_GRASP_CANDIDATE_GENERATION):
    """
    returns a list of X_OF transformations that result in a successful grasp
    size of array is grasp_candidate_count
    """
    env = MakeEnvironmentForGraspCandidates(model_url)

    assert len(env.point_clouds) == 1
    body_idx, pc = list(env.point_clouds.items())[0]

    gripper = PR2Gripper(
        plant=env.plant,
        base_link_name=GRIPPER_BASE_LINK,
    )
    grasp_recommender = GraspRecommender(env, gripper)

    diagram_context = env.get_fresh_diagram_context()
    plant_context = env.plant.GetMyContextFromRoot(diagram_context)
    successful_grasps = []
    for candidate in tqdm(grasp_recommender.generate_random_candidates(
            cloud=pc,
            context=diagram_context), f"trying grasp candidates for {model_url}"):
        optimized_candidate = grasp_recommender.optimize_grasp_with_ik(X_G=candidate.X_frame @ gripper.X_FG(),
                                                                       cloud=pc,
                                                                       context=diagram_context)
        if optimized_candidate.fails:
            continue
        successful_grasps.append(optimized_candidate)
        if len(successful_grasps) >= grasp_candidate_count:
            break

    successful_grasps.sort(key=lambda item: item.cost)
    return [my_grasp.X_frame for my_grasp in successful_grasps]


def MakeEnvironmentForGraspCandidates(model_url):
    # example system for the object we want to get a point cloud of and the cameras used
    builder = DiagramBuilder()

    # Create the physics engine + scene graph.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    CustomConfigureParser(parser)
    object_name = 'object'
    yaml_string = generate_yaml_file(model_url=model_url, gripper_url=GRIPPER_MODEL_URL, object_name=object_name)
    parser.AddModelsFromString(yaml_string, '.dmd.yaml')
    plant.Finalize()

    plant: MultibodyPlant = plant
    diagram = builder.Build()
    diagram.set_name("gripper_sample_system")

    model_instance = plant.GetModelInstanceByName(object_name)
    body = plant.GetBodyByName(get_one_floating_body_base_frame(model_url), model_instance)
    point_cloud = GetCachedModelPointCloud(model_url, reset_cache=False)
    return Environment(
        diagram=diagram,
        plant=plant,
        scene_graph=scene_graph,
        model_id=plant.GetModelInstanceByName("gripper"),
        movable_bodies=None,
        context_update_function=None,
        point_clouds={body.index(): point_cloud},
        floating_bodies=[body],
        fixed_bodies=[]
    )


def generate_yaml_file(model_url, gripper_url, object_name):
    base_frame = get_one_floating_body_base_frame(model_url)
    return f"""
directives:

- add_model:
    name: {object_name}
    file: {model_url}

- add_weld:
    parent: world
    child: {object_name}::{base_frame}

- add_model:
    name: gripper
    file: {gripper_url}
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
