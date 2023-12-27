from generate_models.parameters import *
from settings import FLOOR_FOR_BRICKS


def generate_env_yaml():
    return f"""
{add_shelves()}
{add_all_items()}
    """
# {add_obstacles()}


# {add_floor()} remove floor because of the collision with robot


def add_shelves():
    return f"""
directives:

# Add shelves
- add_model:
    name: shelves_1
    file: package://Shelve_Bot/models/shelve.sdf

- add_frame:
    name: shelf_1_origin
    X_PF:
      base_frame: world
      translation: [{ROBOT_DISTANCE_SHELF}, 0, {SHELF_HEIGHT / 2}]
      rotation: !Rpy {{ deg: [0, 0, 180] }}

- add_weld:
    parent: shelf_1_origin
    child: shelves_1::shelve_body

- add_model:
    name: shelves_2
    file: package://Shelve_Bot/models/shelve.sdf

- add_frame:
    name: shelf_2_origin
    X_PF:
      base_frame: world
      translation: [0, {ROBOT_DISTANCE_SHELF}, {SHELF_HEIGHT / 2}]
      rotation: !Rpy {{ deg: [0, 0, -90] }}

- add_weld:
    parent: shelf_2_origin
    child: shelves_2::shelve_body

- add_model:
    name: shelves_3
    file: package://Shelve_Bot/models/shelve.sdf

- add_frame:
    name: shelf_3_origin
    X_PF:
      base_frame: world
      translation: [{-ROBOT_DISTANCE_SHELF}, 0, {SHELF_HEIGHT / 2}]
      rotation: !Rpy {{ deg: [0, 0, 0] }}

- add_weld:
    parent: shelf_3_origin
    child: shelves_3::shelve_body

- add_model:
    name: shelves_4
    file: package://Shelve_Bot/models/shelve.sdf

- add_frame:
    name: shelf_4_origin
    X_PF:
      base_frame: world
      translation: [0, {-ROBOT_DISTANCE_SHELF}, {SHELF_HEIGHT / 2}]
      rotation: !Rpy {{ deg: [0, 0, 90] }}

- add_weld:
    parent: shelf_4_origin
    child: shelves_4::shelve_body
    """


thing_counter = 0


def add_thing(init_shelf_idx, init_shelf_floor, dest_shelf_idx, dest_shelf_floor, url, move_up, rpy=(0, 0, 0)):
    epsilon = 1.5e-2
    global thing_counter
    thing_counter = thing_counter + 1
    return f"""
- add_model:
    name: thing_{thing_counter}
    file: {url}
- add_frame:
    name: thing_{thing_counter}_origin
    X_PF:
      base_frame: shelf_{init_shelf_idx}_origin
      translation: [0, 0, {init_shelf_floor * SHELF_FLOOR_HEIGHT + move_up - SHELF_HEIGHT / 2 + SHELF_THICKNESS + epsilon}]
      rotation: !Rpy {{ deg: [{','.join([str(x) for x in rpy])}] }}
- add_frame:
    name: thing_{thing_counter}_destination
    X_PF:
      base_frame: shelf_{dest_shelf_idx}_origin
      translation: [0, 0, {dest_shelf_floor * SHELF_FLOOR_HEIGHT + move_up - SHELF_HEIGHT / 2 + SHELF_THICKNESS + epsilon}]
      rotation: !Rpy {{ deg: [{','.join([str(x) for x in rpy])}] }}
        """


def add_brick_to_shelf_floor(init_shelf_idx, init_shelf_floor, dest_shelf_idx, dest_shelf_floor):
    return add_thing(init_shelf_idx, init_shelf_floor, dest_shelf_idx, dest_shelf_floor,
                     url='package://Shelve_Bot/models/brick.sdf',
                     move_up=BRICK_HEIGHT / 2)


def add_soup_can_to_shelf_floor(init_shelf_idx, init_shelf_floor, dest_shelf_idx, dest_shelf_floor):
    # todo change soup_can_cheat.sdf back to soup_can
    return add_thing(init_shelf_idx, init_shelf_floor, dest_shelf_idx, dest_shelf_floor,
                     url='package://Shelve_Bot/models/soup_can_cheat.sdf',
                     move_up=SOUP_CAN_HEIGHT / 2,
                     rpy=(90, 0, 0))


def add_rubiks_cube_to_shelf_floor(init_shelf_idx, init_shelf_floor, dest_shelf_idx, dest_shelf_floor):
    # todo change soup_can_cheat.sdf back to soup_can
    return add_thing(init_shelf_idx, init_shelf_floor, dest_shelf_idx, dest_shelf_floor,
                     url='package://Shelve_Bot/models/rubiks_cube_2_by_2.sdf',
                     move_up=RUBIKS_CUBE_HEIGHT / 2)


def add_all_items():
    yaml = ""
    yaml += add_brick_to_shelf_floor(1, FLOOR_FOR_BRICKS, 2, 1)
    yaml += add_brick_to_shelf_floor(2, FLOOR_FOR_BRICKS, 3, 1)
    yaml += add_brick_to_shelf_floor(3, FLOOR_FOR_BRICKS, 4, 1)
    # yaml += add_brick_to_shelf_floor(1, 2)  # todo this is to increase the speed of visualization. todo change later...
    # yaml += add_brick_to_shelf_floor(2, 2)  # todo this is to increase the speed of visualization. todo change later...

    # for shelf_idx in range(1, 5):
    #     for shelf_floor in range(SHELF_FLOORS):
    #         yaml += add_brick_to_shelf_floor(shelf_idx, shelf_floor)

    return yaml


def add_floor():
    return f"""
- add_model:
    name: floor
    file: package://Shelve_Bot/models/floor.sdf
- add_weld:
    parent: world
    child: floor::box
        """

def add_obstacles():
    return f"""
- add_model:
    name: obstacle_1
    file: package://Shelve_Bot/models/obstacle.sdf
    
- add_frame:
    name: obstacle_1_frame
    X_PF:
      base_frame: world
      translation: [{ROBOT_DISTANCE_SHELF/2}, 0, {SHELF_HEIGHT / 2}]
      rotation: !Rpy {{ deg: [0, 0, 0] }}

- add_weld:
    parent: obstacle_1_frame
    child: obstacle_1::obstacle_center
"""
