from generate_models import Playground
from planner.Action import Action
from planner.Command import Hand, Command


class OpenGripper(Action):
    """
    Opens the gripper to ready for grasping
    """

    def __init__(self, playground: Playground, hand: Hand):
        super(OpenGripper, self).__init__(playground)
        self.hand = hand
        self.plan_duration = 1
        self.start_time = 0

    def state_init(self):
        self.start_time = self.time

    def run(self, prev_command: Command):
        finished = (self.time - self.start_time) > self.plan_duration
        command = self.hand.open(prev_command)
        return command, finished

