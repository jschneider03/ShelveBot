import os

PROJECT_PATH = os.path.dirname(__file__)
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
PACKAGE_XML_PATH = os.path.join(PROJECT_PATH, "package.xml")
CACHE_PATH = os.path.join(PROJECT_PATH, "cache")
DEFAULT_ENV_URL = "package://Shelve_Bot/models/main_environment.dmd.yaml"
EASY_ENV_URL = "package://Shelve_Bot/models/easy_environment.dmd.yaml"
PR2_MODEL_URL = "package://Shelve_Bot/models/pr2_simplified_wo_mimic.urdf"
# PR2_MODEL_URL = "package://Shelve_Bot/models/pr2_collisions_filtered.urdf"
GRIPPER_MODEL_URL = "package://Shelve_Bot/models/r_gripper_fixed_open_model.urdf"
GRIPPER_BASE_LINK = "r_gripper_palm_link"

PR2_MAIN_LINK = "base_link_for_rbt_compat"
MAX_GRASP_CANDIDATE_GENERATION = 20
FLOOR_FOR_BRICKS = 2  # must be between 0 and 2 corresponding to 1 and 3
PRESEEDED_IK = True # set to False to solve all IKs with random init

# make the necessary folders:
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)
