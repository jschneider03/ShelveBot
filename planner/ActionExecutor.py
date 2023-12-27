import numpy as np
from pydrake.all import LeafSystem, DiagramBuilder, QueryObject, AbstractValue

from generate_models import Playground

from robots import PR2
from .Action import Action
from .Command import Command


class ActionExecutor(LeafSystem):
    def __init__(self, action: Action):
        LeafSystem.__init__(self)

        playground = action.playground
        self.action = action

        ################################
        # declare input/output

        self.DeclareVectorInputPort(
            name="plant_continuous_state",  # assume this is q, dq
            size=playground.env.plant.get_state_output_port().size()
        )
        self.DeclareVectorInputPort(
            name="pr2.torque_external",
            size=playground.env.plant.get_generalized_contact_forces_output_port(playground.env.model_id).size()
        )

        self.DeclareAbstractInputPort(
            name="query_object",
            model_value=AbstractValue.Make(QueryObject())
        )

        self.DeclareVectorOutputPort(
            name="pr2_position_command",
            size=playground.env.plant.num_actuated_dofs(),
            calc=self.CalcPositionCommand
        )
        self.DeclareVectorOutputPort(
            name="feedforward_torque",
            size=playground.env.plant.num_actuated_dofs(),
            calc=self.CalcFeedForwardTorque
        )
        # end declare input/output
        ################################
        self.is_finished = False

        # this part is the only non-beautiful part about this framework which is a result of technical debt
        # we need to construct pr2 in order to make the command
        pr2 = PR2(playground.construct_welded_sim(playground.default_continuous_state()))
        self.command = Command(pr2=pr2, position_command=playground.construct_welded_sim(playground.default_continuous_state()).plant.GetDefaultPositions()) # default command is all zeros

    def CalcPositionCommand(self, context, output):
        if self.is_finished:
            return
        continuous_state = self.GetInputPort("plant_continuous_state").Eval(context)
        self.action.set_data(
            continuous_state=continuous_state,
            time=context.get_time(),
            torque_external=self.GetInputPort("pr2.torque_external").Eval(context)
        )
        self.command, done = self.action.run_or_init(self.command)
        if done:
            self.action.state_finished()
        output.SetFromVector(self.command.position_command)
        self.is_finished = self.is_finished or done

    # todo
    def CalcFeedForwardTorque(self, context, output):
        # maybe use later for gripper
        output.SetFromVector(np.zeros(output.size()))


def connect_to_the_world(playground: Playground, action_executor: ActionExecutor):
    builder = DiagramBuilder()
    inner_diagram = builder.AddSystem(playground.env.diagram)
    builder.AddSystem(action_executor)

    builder.Connect(
        action_executor.GetOutputPort("pr2_position_command"),
        inner_diagram.GetInputPort("pr2.position")
    )
    builder.Connect(
        action_executor.GetOutputPort("feedforward_torque"),
        inner_diagram.GetInputPort("pr2.feedforward_torque")
    )
    builder.Connect(
        inner_diagram.GetOutputPort("query_object"),
        action_executor.GetInputPort("query_object")
    )
    builder.Connect(
        inner_diagram.GetOutputPort("plant_continuous_state"),
        action_executor.GetInputPort("plant_continuous_state")
    )
    builder.Connect(
        inner_diagram.GetOutputPort("pr2.torque_external"),
        action_executor.GetInputPort("pr2.torque_external")
    )
    return builder.Build()
