from generate_models.parameters import *


def generate_shelve_sdf():
    right_wall = f"""
        <pose> 0 {SHELF_WIDTH/2 - SHELF_THICKNESS/2} 0 0 0 0</pose>
        <geometry>
          <box>
            <size>{SHELF_DEPTH} {SHELF_THICKNESS} {SHELF_HEIGHT}</size>
          </box>
        </geometry>
    """

    left_wall = f"""
        <pose> 0 {-SHELF_WIDTH/2 + SHELF_THICKNESS/2} 0 0 0 0</pose>
        <geometry>
          <box>
            <size>{SHELF_DEPTH} {SHELF_THICKNESS} {SHELF_HEIGHT}</size>
          </box>
        </geometry>
    """

    def floor(i):
        return f"""
        <pose> 0 0 {i * SHELF_FLOOR_HEIGHT + SHELF_THICKNESS/2 - SHELF_HEIGHT/2} 0 0 0</pose>
        <geometry>
          <box>
            <size>{SHELF_DEPTH} {SHELF_WIDTH} {SHELF_THICKNESS}</size>
          </box>
        </geometry>
        """

    back = f"""
        <pose> {-SHELF_DEPTH/2 + SHELF_THICKNESS/2} 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>{SHELF_THICKNESS} {SHELF_WIDTH} {SHELF_HEIGHT}</size>
          </box>
        </geometry>    
    """

    sdf = ""
    sdf += f"""
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="shelve">
    <link name="{SHELVE_BODY_NAME}">
    """

    sdf += f"""
      <visual name="right_wall">
        {right_wall}
      </visual>
     <collision name="right_wall">
        {right_wall}
      </collision>
    """

    sdf += f"""
      <visual name="left_wall">
        {left_wall}
      </visual>
     <collision name="left_wall">
        {left_wall}
      </collision>
    """

    for i in range(SHELF_FLOORS + 1):
        sdf += f"""
          <visual name="floor{i}">
            {floor(i)}
          </visual>
          <collision name="floor{i}">
            {floor(i)}
          </collision>
        """

    sdf += f"""
      <visual name="back">
        {back}
      </visual>
     <collision name="back">
        {back}
      </collision>
    """

    sdf += f"""
    </link>
  </model>
</sdf>
    """

    return sdf
