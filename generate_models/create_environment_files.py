import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from generate_models.generate_shelve_sdf import generate_shelve_sdf
from generate_models.generate_env_yaml import generate_env_yaml
from generate_models.generate_brick_sdf import generate_brick_sdf
from generate_models.generate_chain_sdf import generate_chain_sdf
from settings import MODELS_PATH


def create_environment_files():
    SHELVES_PATH = os.path.join(MODELS_PATH, "shelve.sdf")
    with open(SHELVES_PATH, "w") as f:
        f.write(generate_shelve_sdf())

    MAIN_ENV_PATH = os.path.join(MODELS_PATH, "main_environment.dmd.yaml")
    with open(MAIN_ENV_PATH, "w") as f:
        f.write(generate_env_yaml())

    BRICK_PATH = os.path.join(MODELS_PATH, "brick.sdf")
    with open(BRICK_PATH, "w") as f:
        f.write(generate_brick_sdf())

    CHAIN_PATH = os.path.join(MODELS_PATH, "chain.sdf")
    with open(CHAIN_PATH, "w") as f:
        f.write(generate_chain_sdf())


if __name__ == "__main__":
    create_environment_files()
