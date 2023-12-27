import numpy as np

from pydrake.all import RollPitchYaw, RigidTransform
from generate_models.parameters import *


def generate_chain_sdf(n=CHAIN_N, length=CHAIN_LENGTH, radius=CHAIN_RADIUS, random_seed=0):
    np.random.seed(random_seed)  # fix the seed

    sample_model_sdf = f"""
    <?xml version="1.0" ?>
    <sdf version="1.7">
        <model name="chain_n={n}">
    """

    # adding links
    X_WL = RigidTransform()

    for i in range(n):
        go_up = RigidTransform(p=[0, 0, length])
        go_margin = RigidTransform(p=[0, 0, radius])
        then_rotate = RigidTransform(RollPitchYaw(np.random.rand(3) * np.pi / 2), p=[0, 0, 0])

        if i != 0:
            # add joint to previous sphere
            sample_model_sdf += f"""
                <joint name="joint_p_{i}" type="ball">
                    <parent>sphere{i - 1}</parent>
                    <child>link{i}</child>
                    <pose>0 0 {-radius} 0 0 0</pose>
                </joint>
            """
        # add joint to this sphere
        sample_model_sdf += f"""
            <joint name="joint_n_{i}" type="ball">
                <parent>link{i}</parent>
                <child>sphere{i}</child>
                <pose>0 0 0 0 0 0</pose>
            </joint>
        """

        sample_model_sdf += __generate_link_sdf(i, radius, length, X_WL)
        X_WL = X_WL @ go_up @ go_margin
        sample_model_sdf += __generate_sphere_sdf(i, radius, X_WL)
        X_WL = X_WL @ then_rotate @ go_margin

    sample_model_sdf += """
        </model>
    </sdf>
    """
    return sample_model_sdf


def __generate_link_sdf(idx, radius, length, X_WL):
    x, y, z = X_WL.translation()
    rpy = RollPitchYaw(X_WL.rotation())
    roll, pitch, yaw = rpy.roll_angle(), rpy.pitch_angle(), rpy.yaw_angle()

    return f"""
    <link name="link{idx}">
        <pose>{x} {y} {z} {roll} {pitch} {yaw}</pose>
        <inertial>
            <pose>0 0 {length / 2} 0 0 0</pose>
            <mass>0.1</mass>
            <inertia>
            <ixx>0.1</ixx><ixy>0.0</ixy><ixz>0.0</ixz>
                            <iyy>0.1</iyy><iyz>0.0</iyz>
                                        <izz>0.01</izz>
            </inertia>
        </inertial>
        <visual name="visual_box{idx}">
            <pose>0 0 {length / 2} 0 0 0</pose>
            <geometry>
            <box>
                <size>{2 * radius} {2 * radius} {length}</size>
            </box>
            </geometry>
        </visual>
        <collision name="collision_box{idx}">
            <pose>0 0 {length / 2} 0 0 0</pose>
            <geometry>
            <box>
                <size>{2 * radius} {2 * radius} {length}</size>
            </box>
            </geometry>
            <drake:proximity_properties>
                <drake:compliant_hydroelastic/>
                <drake:hydroelastic_modulus>1.0e8</drake:hydroelastic_modulus>
            </drake:proximity_properties>
        </collision>
    </link>
    """


def __generate_sphere_sdf(idx, radius, X_WL):
    x, y, z = X_WL.translation()
    rpy = RollPitchYaw(X_WL.rotation())
    roll, pitch, yaw = rpy.roll_angle(), rpy.pitch_angle(), rpy.yaw_angle()

    return f"""
    <link name="sphere{idx}">
        <pose>{x} {y} {z} {roll} {pitch} {yaw}</pose>
        <inertial>
            <mass>0.1</mass>
        </inertial>
        <visual name="visual_sphere{idx}">
            <geometry>
            <sphere>
                <radius>{radius}</radius>
            </sphere>
            </geometry>
        </visual>
        <collision name="collision_sphere{idx}">
            <geometry>
            <sphere>
                <radius>{radius}</radius>
            </sphere>
            </geometry>

            <drake:proximity_properties>
                <drake:compliant_hydroelastic/>
                <drake:mesh_resolution_hint>0.1</drake:mesh_resolution_hint>
                <drake:hydroelastic_modulus>5e7</drake:hydroelastic_modulus>
                <drake:hunt_crossley_dissipation>1.25</drake:hunt_crossley_dissipation>
            </drake:proximity_properties>

        </collision>
    </link>
    """
