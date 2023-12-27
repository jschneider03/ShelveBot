from generate_models.parameters import *


def generate_brick_sdf():
    sdf = ""
    sdf += f"""
    <?xml version="1.0"?>
    <sdf version="1.7">
      <model name="brick">
        <link name="brick_center">
            <inertial>
                <pose>0 0 0 0 0 0</pose>
                <mass>0.01</mass>
                <inertia>
                <ixx>0.01</ixx><ixy>0.0</ixy><ixz>0.0</ixz>
                                <iyy>0.01</iyy><iyz>0.0</iyz>
                                            <izz>0.001</izz>
                </inertia>
            </inertial>
        """

    box = f"""
        <pose> 0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>{BRICK_LENGTH} {BRICK_WIDTH} {BRICK_HEIGHT}</size>
          </box>
        </geometry>
    """

    sdf += f"""
      <visual name="brick_box">
        {box}
        <!-- Add Material with Red Color -->
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
          <specular>0 0 0 0</specular>
          <emissive>0 1 0 1</emissive>
        </material>
      </visual>
      
     <collision name="brick_box">
        {box}
        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus>1.0e8</drake:hydroelastic_modulus>
        </drake:proximity_properties>
      </collision>
    """

    sdf += f"""
    </link>
  </model>
</sdf>
    """

    return sdf
