import numpy as np
from abc import ABC, abstractmethod

from robots import PR2


class Command:
    def __init__(self, pr2: PR2, position_command: np.ndarray):
        self.pr2 = pr2
        self.position_command = position_command

    def new_command_ignore_grippers(self, new_position_command: np.ndarray) -> "Command":
        my_positions = self.position_command.copy()
        self.pr2.set_ignoring_gripper_position(
            positions_from=new_position_command,
            positions_to=my_positions
        )
        return Command(pr2=self.pr2, position_command=my_positions)


class Hand(ABC):
    def __init__(self, pr2: PR2):
        self.pr2 = pr2

    @abstractmethod
    def open(self, command: Command) -> Command:
        pass

    @abstractmethod
    def close(self, command: Command) -> Command:
        pass

    @abstractmethod
    def connecting_frame_name(self):
        pass


class LeftHand(Hand):
    def open(self, command: Command) -> Command:
        position = command.position_command
        command.pr2.get_open_gripper_position(
            gripper_joint_name=command.pr2.l_gripper_joint_name,
            initial_positions=position
        )
        return Command(pr2=command.pr2, position_command=position)

    def close(self, command: Command) -> Command:
        position = command.position_command
        command.pr2.get_close_gripper_position(
            gripper_joint_name=command.pr2.l_gripper_joint_name,
            initial_positions=position
        )
        return Command(pr2=command.pr2, position_command=position)

    def connecting_frame_name(self):
        return self.pr2.l_gripper_base_link_name


class RightHand(Hand):
    def open(self, command: Command) -> Command:
        position = command.position_command
        command.pr2.get_open_gripper_position(
            gripper_joint_name=command.pr2.r_gripper_joint_name,
            initial_positions=position
        )
        return Command(pr2=command.pr2, position_command=position)

    def close(self, command: Command) -> Command:
        position = command.position_command
        command.pr2.get_close_gripper_position(
            gripper_joint_name=command.pr2.r_gripper_joint_name,
            initial_positions=position
        )
        return Command(pr2=command.pr2, position_command=position)

    def connecting_frame_name(self):
        return self.pr2.r_gripper_base_link_name
