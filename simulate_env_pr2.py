import os
import sys
import time
import numpy as np
import pydot

from pydrake.all import StartMeshcat, Simulator
from manipulation.station import load_scenario, MakeHardwareStation
from manipulation.scenarios import AddMultibodyTriad
from utils import *

sys.path.append(os.path.dirname(__file__))

from settings import MODELS_PATH, PACKAGE_XML_PATH

meshcat = StartMeshcat()


def simulate_env():
    scenario = load_scenario(filename=os.path.join(MODELS_PATH, "main_environment.dmd.yaml"))
    station = MakeHardwareStation(scenario, meshcat, package_xmls=[os.path.join(PACKAGE_XML_PATH)])
    simulator = Simulator(station)
    context = simulator.get_mutable_context()

    scene_graph = station.GetSubsystemByName("scene_graph")
    # plant_context = plant.GetMyContextFromRoot(context)
    sg_context = scene_graph.GetMyContextFromRoot(context)

    filterPR2CollsionGeometry(scene_graph, sg_context)

    x0 = station.GetOutputPort("pr2.state_estimated").Eval(context)
    station.GetInputPort("pr2.desired_state").FixValue(context, x0)
    
    meshcat.StartRecording()
    simulator.AdvanceTo(10.0)
    meshcat.PublishRecording()

    while True:  # to make simulation meshcat run
        time.sleep(1)


if __name__ == "__main__":
    simulate_env()
