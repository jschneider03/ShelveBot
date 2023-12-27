from generate_models import Playground
from planner.Action import Action
from planner.Command import Hand, Command


class CloseGripper(Action):
    """
    closes the gripper $hand
    """

    def __init__(self, playground: Playground, hand: Hand):
        super(CloseGripper, self).__init__(playground)
        self.hand = hand
        self.plan_duration = 1
        self.start_time = 0

    def state_init(self):
        self.start_time = self.time

    def run(self, prev_command: Command):
        finished = (self.time - self.start_time) > self.plan_duration
        command = self.hand.close(prev_command)
        return command, finished
