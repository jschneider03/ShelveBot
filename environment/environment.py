from dataclasses import dataclass

import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, Parser, SceneGraph
from pydrake.multibody.tree import Body, ModelInstanceIndex, BodyIndex
from pydrake.systems.framework import DiagramBuilder, LeafSystem, BasicVector, Diagram
from pydrake.all import PointCloud

from pydrake.all import MultibodyPlant, RigidTransform
from pydrake.all import Context
from typing import Dict, Tuple, List, Callable

from .fixed_body import FixedBody, Shelve
from .movable_body import MovableBody


@dataclass
class Environment:
    diagram: Diagram
    plant: MultibodyPlant
    scene_graph: SceneGraph
    model_id: ModelInstanceIndex
    movable_bodies: List[MovableBody]
    fixed_bodies: List[FixedBody]
    context_update_function: Callable[["Environment", Context, np.ndarray], Context]

    # todo depricated. not removed fot the sake of compatibility. but use MovableBody instead
    point_clouds: Dict[Tuple[BodyIndex], PointCloud]  # todo is it possible to do anything better than Dict[Tuple]?
    floating_bodies: List[Body]

    def get_fresh_diagram_context(self, playground_continuous_state=None):
        context = self.diagram.CreateDefaultContext()
        if playground_continuous_state is None:
            return context
        return self.context_update_function(self, context, playground_continuous_state)

    def get_updated_diagram_context(self, context: Context, playground_continuous_state: np.ndarray):
        return self.context_update_function(self, context, playground_continuous_state)

    def get_fresh_plant_context(self, playground_continuous_state=None):
        diagram_context = self.get_fresh_diagram_context(playground_continuous_state)
        return self.plant.GetMyContextFromRoot(diagram_context)

    def get_fresh_scene_graph_context(self, playground_continuous_state=None):
        diagram_context = self.get_fresh_diagram_context(playground_continuous_state)
        return self.scene_graph.GetMyContextFromRoot(diagram_context)

    def get_all_shelves(self):
        return [item for item in self.fixed_bodies if isinstance(item, Shelve)]

    def ForcedPublish(self, context=None):
        if context is None:
            context = self.get_fresh_diagram_context()
        self.diagram.ForcedPublish(context)
