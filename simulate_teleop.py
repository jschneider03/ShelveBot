import os
import sys

from pydrake.all import (
    StartMeshcat,
    Simulator,
)
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import DiagramBuilder, Diagram, System

from generate_models import Playground
from generate_models.create_environment import AddPR2Plant
from settings import MODELS_PATH, PACKAGE_XML_PATH
from pydrake.all import JointSliders

sys.path.append(os.path.dirname(__file__))

meshcat = StartMeshcat()


def add_teleop(inner_diagram, plant):
    builder = DiagramBuilder()
    inner_diagram = builder.AddSystem(inner_diagram)
    position_port = inner_diagram.GetInputPort("pr2.position")
    # feedforward_torque_port = inner_diagram.GetInputPort("pr2.feedforward_torque")
    # input_feedforward_torque = builder.AddSystem(ConstantVectorSource(np.zeros(feedforward_torque_port.size())))
    # builder.Connect(input_feedforward_torque.get_output_port(), feedforward_torque_port)

    controller_plant: MultibodyPlant = MultibodyPlant(time_step=plant.time_step())
    AddPR2Plant(controller_plant)
    controller_plant.Finalize()

    teleop: JointSliders = builder.AddSystem(JointSliders(
        meshcat=meshcat, plant=controller_plant)
    )
    builder.Connect(teleop.get_output_port(), position_port)
    return builder.Build()


def simulate_env():
    playground = Playground(meshcat=meshcat, time_step=0.0001, regenerate_sdf_files=True)
    diagram = playground.env.diagram
    plant = playground.env.plant
    teleop_diagram = add_teleop(inner_diagram=diagram, plant=plant)

    simulator = Simulator(teleop_diagram)
    simulator.AdvanceTo(1000.0)
    # while True:  # to make simulation meshcat run
    #     simulator.AdvanceTo(2.0)

    # meshcat.StartRecording()
    # simulator.AdvanceTo(1.0)
    # meshcat.StopRecording()
    # meshcat.PublishRecording()


if __name__ == "__main__":
    simulate_env()
