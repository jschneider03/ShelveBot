import pydot
import numpy as np
from pydrake.all import (
    GeometrySet,
    CollisionFilterDeclaration,
    Role,
    PrismaticJoint, 
    RevoluteJoint
)
from robot import ConfigurationSpace, Range

# todo later add the collision fixes to the URDF instead
def filterPR2CollsionGeometry(scene_graph, context=None):
    """Some robot models may appear to have self collisions due to overlapping collision geometries.
    This function filters out such problems for our PR2 model."""
    if context is None:
        filter_manager = scene_graph.collision_filter_manager()
    else:
        filter_manager = scene_graph.collision_filter_manager(context)
    inspector = scene_graph.model_inspector()

    pr2 = {}

    for gid in inspector.GetGeometryIds(
        GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity
    ):
        gid_name = inspector.GetName(inspector.GetFrameId(gid))
        if "pr2" in gid_name:
            link_name = gid_name.split("::")[1]
            pr2[link_name] = [gid]

    def add_exclusion(set1, set2=None):
        if set2 is None:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeWithin(GeometrySet(set1))
            )
        else:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeBetween(
                    GeometrySet(set1), GeometrySet(set2)
                )
            )

    # Robot-to-self collisions
    add_exclusion(
        pr2["base_link"],
        pr2["l_shoulder_pan_link"]
        + pr2["r_shoulder_pan_link"]
        + pr2["l_upper_arm_link"]
        + pr2["r_upper_arm_link"]
        + pr2["head_pan_link"]
        + pr2["head_tilt_link"],
    )
    add_exclusion(
        pr2["torso_lift_link"], pr2["head_pan_link"] + pr2["head_tilt_link"]
    )
    add_exclusion(
        pr2["l_shoulder_pan_link"] + pr2["torso_lift_link"],
        pr2["l_upper_arm_link"],
    )
    add_exclusion(
        pr2["r_shoulder_pan_link"] + pr2["torso_lift_link"],
        pr2["r_upper_arm_link"],
    )
    # forearm to gripper
    add_exclusion(pr2["l_forearm_link"], pr2["l_gripper_palm_link"])
    add_exclusion(pr2["r_forearm_link"], pr2["r_gripper_palm_link"])
    # upper arm to forearm
    add_exclusion(pr2["r_upper_arm_link"], pr2["r_forearm_link"])
    add_exclusion(pr2["l_upper_arm_link"], pr2["l_forearm_link"])
    # forearm to finger
    add_exclusion(pr2["l_forearm_link"], pr2["l_gripper_r_finger_link"])
    add_exclusion(pr2["r_forearm_link"], pr2["r_gripper_l_finger_link"])
    
    # l_gripper_l
    add_exclusion(pr2["l_gripper_l_finger_link"], pr2["l_gripper_r_finger_link"])
    add_exclusion(pr2["l_gripper_l_finger_link"], pr2["l_gripper_l_finger_link"])
    add_exclusion(pr2["l_gripper_l_finger_tip_link"], pr2["l_gripper_r_finger_tip_link"])
    add_exclusion(pr2["l_gripper_l_finger_link"], pr2["l_gripper_r_finger_tip_link"])
    # r_gripper_l
    add_exclusion(pr2["r_gripper_l_finger_link"], pr2["r_gripper_r_finger_link"])
    add_exclusion(pr2["r_gripper_l_finger_link"], pr2["r_gripper_r_finger_tip_link"])
    add_exclusion(pr2["r_gripper_l_finger_tip_link"], pr2["r_gripper_r_finger_tip_link"])
    # r_gripper_r
    add_exclusion(pr2["r_gripper_l_finger_link"], pr2["r_gripper_r_finger_link"])
    add_exclusion(pr2["r_gripper_r_finger_link"], pr2["r_gripper_r_finger_link"])
    add_exclusion(pr2["r_gripper_r_finger_link"], pr2["r_gripper_l_finger_tip_link"])
    add_exclusion(pr2["r_gripper_l_finger_tip_link"], pr2["r_gripper_r_finger_tip_link"])
    add_exclusion(pr2["r_gripper_l_finger_link"], pr2["r_gripper_r_finger_tip_link"])
    # l gripper r
    add_exclusion(pr2["l_gripper_r_finger_link"], pr2["l_gripper_l_finger_tip_link"])
    # finger tips
    add_exclusion(pr2["r_gripper_l_finger_tip_link"], pr2["r_gripper_r_finger_tip_link"])
    add_exclusion(pr2["l_gripper_r_finger_tip_link"], pr2["l_gripper_l_finger_tip_link"])

def make_svg(fn, obj):
    # print(station.GetGraphvizString())
    with open(f"{fn}.svg", "wb") as f:
        f.write(pydot.graph_from_dot_data(obj.GetGraphvizString())[0].create_svg())

def l2_distance(q: tuple):
    sum = 0
    for q_i in q:
        sum += q_i**2
    return np.sqrt(sum)

def get_non_weld_joints(plant, model_id):
    lst = []
    for ix in plant.GetJointIndices(model_id):
        joint = plant.get_joint(ix)
        if isinstance(joint, PrismaticJoint) or isinstance(joint, RevoluteJoint):
            lst.append(joint)
    return lst

def get_ik_joints_ix(plant, model_id):
    lst = []
    for i, ix in enumerate(plant.GetActuatedJointIndices(model_id)):
        joint = plant.get_joint(ix)
        if (isinstance(joint, PrismaticJoint) or isinstance(joint, RevoluteJoint)) and not joint.name() in ["x", "y", "theta"]:
            lst.append(i)
    return lst

def get_pr2_non_base_lower_bounds(plant, model_id):
    lst = []
    lb = plant.GetPositionLowerLimits()
    for i, ix in enumerate(plant.GetActuatedJointIndices(model_id)):
        joint = plant.get_joint(ix)
        if (isinstance(joint, PrismaticJoint) or isinstance(joint, RevoluteJoint)) and not joint.name() in ["x", "y", "theta"]:
            lst.append(lb[i])
    return lst

def get_pr2_non_base_upper_bounds(plant, model_id):
    lst = []
    ub = plant.GetPositionUpperLimits()
    for i, ix in enumerate(plant.GetActuatedJointIndices(model_id)):
        joint = plant.get_joint(ix)
        if (isinstance(joint, PrismaticJoint) or isinstance(joint, RevoluteJoint)) and not joint.name() in ["x", "y", "theta"]:
            lst.append(ub[i])
    return lst

def get_actuated_ix(plant):
    """
    Helper function that maps actuated joints to PR2 and environment
    """
    indices = {"pr2": [], "brick": []}
    pos_names = plant.GetPositionNames()
    for ix, name in enumerate(pos_names):
        if "pr2" in name:
            indices["pr2"].append(ix)
        elif "brick" in name:
            indices["brick"].append(ix)
    return indices

def min_distance_collision_checker(plant, plant_context, distance):
    query_object = plant.get_geometry_query_input_port().Eval(plant_context)
    inspector = query_object.inspector()
    pairs = inspector.GetCollisionCandidates()
    for pair in pairs:
        val = query_object.ComputeSignedDistancePairClosestPoints(pair[0], pair[1]).distance, inspector.GetName(inspector.GetFrameId(pair[0])), inspector.GetName(inspector.GetFrameId(pair[1]))
        if val[0] <= distance:
            print(val)

def save_graphviz_to_txt(thing_to_save):
    text_file = open("viz_string.txt", "w")
    text_file.write(thing_to_save.GetGraphvizString())
    text_file.close()

def theta_adjusted_pos(pos):
    pos = np.array(pos)
    pos[2] %= (2 * np.pi)
    return pos

def choose_closest_heuristic(left_name, right_name, p_X_WG, plant, context):
    """
    Chooses which gripper to attempt to grab the object on using a closest distance
    heuristic. Preference for right gripper when there's a tie.
    """
    X_WL = plant.GetFrameByName(left_name).CalcPoseInWorld(context)
    X_WR = plant.GetFrameByName(right_name).CalcPoseInWorld(context)
    left_dist = np.linalg.norm(p_X_WG - X_WL.translation())
    right_dist = np.linalg.norm(p_X_WG - X_WR.translation())
    return right_name if right_dist <= left_dist else left_name

def matching_q_no_fingers(q_actual, q_des, atol=1e-8):
    q_actual = theta_adjusted_pos(q_actual)
    q_des = theta_adjusted_pos(q_des)
    res = np.isclose(q_actual, q_des, atol=atol)
    if (q_actual[2] <= 1e-2 and abs((2*np.pi) - q_des[2]) <= 1e-2) or \
        (q_des[2] <= 1e-2 and abs((2*np.pi) - q_actual[2]) <= 1e-2):
        res[2] = True
    res[13] = True
    res[21] = True
    return np.all(res)

def matching_q_holding_obj(q_actual, q_des, gripper_name, atol=1e-8):
    q_actual = theta_adjusted_pos(q_actual)
    q_des = theta_adjusted_pos(q_des)
    res = np.isclose(q_actual, q_des, atol=atol)
    if (q_actual[2] <= 1e-2 and abs((2*np.pi) - q_des[2]) <= 1e-2) or \
        (q_des[2] <= 1e-2 and abs((2*np.pi) - q_actual[2]) <= 1e-2):
        res[2] = True
    # always ignore fingers
    res[13] = True
    res[21] = True
    # ignore the wrist flex joint for the gripper holding the object
    # give more slack to roll joints in arm holding obj because obj 
    # increases tracking error
    if "l_gripper" in gripper_name:
        res[19] = True
        res[15] = np.isclose(q_actual[15], q_des[15], atol=0.25)
        res[16] = np.isclose(q_actual[16], q_des[16], atol=0.25)
        res[17] = np.isclose(q_actual[17], q_des[17], atol=0.25)
        res[18] = np.isclose(q_actual[18], q_des[18], atol=0.25)
        res[20] = np.isclose(q_actual[20], q_des[20], atol=0.25)
    else:
        res[11] = True
        res[7] = np.isclose(q_actual[7], q_des[7], atol=0.25)
        res[8] = np.isclose(q_actual[8], q_des[8], atol=0.25)
        res[9] = np.isclose(q_actual[9], q_des[9], atol=0.25)
        res[10] = np.isclose(q_actual[10], q_des[10], atol=0.25)
        res[12] = np.isclose(q_actual[12], q_des[12], atol=0.25)
    return np.all(res)

def get_body_ix_names(self):
    res = []
    for ix in self.pr2.plant.GetBodyIndices(self.pr2.model_id):
        res.append((ix, self.pr2.plant.get_body(ix).name()))
    return res

# -----------------------------------RRT utils-----------------------------------

class TreeNode:
    def __init__(self, value, parent=None):
        self.value = value  # tuple of floats representing a configuration
        self.parent = parent  # another TreeNode
        self.children = []  # list of TreeNodes

class RRT:
    """
    RRT Tree.
    """

    def __init__(self, root: TreeNode, cspace: ConfigurationSpace):
        self.root = root  # root TreeNode
        self.cspace = cspace  # robot.ConfigurationSpace
        self.size = 1  # int length of path
        self.max_recursion = 1000  # int length of longest possible path

    def add_configuration(self, parent_node, child_value):
        child_node = TreeNode(child_value, parent_node)
        parent_node.children.append(child_node)
        self.size += 1
        return child_node

    # Brute force nearest, handles general distance functions
    def nearest(self, configuration):
        """
        Finds the nearest node by distance to configuration in the
             configuration space.

        Args:
            configuration: tuple of floats representing a configuration of a
                robot

        Returns:
            closest: TreeNode. the closest node in the configuration space
                to configuration
            distance: float. distance from configuration to closest
        """
        assert self.cspace.valid_configuration(configuration)

        def recur(node, depth=0):
            closest, distance = node, self.cspace.distance(
                node.value, configuration
            )
            if depth < self.max_recursion:
                for child in node.children:
                    (child_closest, child_distance) = recur(child, depth + 1)
                    if child_distance < distance:
                        closest = child_closest
                        child_distance = child_distance
            return closest, distance

        return recur(self.root)[0]
    
class RRT_tools:
    def __init__(self, problem):
        # rrt is a tree
        self.rrt_tree = RRT(TreeNode(problem.start), problem.cspace)
        problem.rrts = [self.rrt_tree]
        self.problem = problem

    def find_nearest_node_in_RRT_graph(self, q_sample):
        nearest_node = self.rrt_tree.nearest(q_sample)
        return nearest_node

    def sample_node_in_configuration_space(self):
        q_sample = self.problem.cspace.sample()
        return q_sample

    def calc_intermediate_qs_wo_collision(self, q_start, q_end):
        """create more samples by linear interpolation from q_start
        to q_end. Return all samples that are not in collision

        Example interpolated path:
        q_start, qa, qb, (Obstacle), qc , q_end
        returns >>> q_start, qa, qb
        """
        return self.problem.safe_path(q_start, q_end)

    def grow_rrt_tree(self, parent_node, q_sample):
        """
        add q_sample to the rrt tree as a child of the parent node
        returns the rrt tree node generated from q_sample
        """
        child_node = self.rrt_tree.add_configuration(parent_node, q_sample)
        return child_node

    def node_reaches_goal(self, node):
        return node.value == self.problem.goal

    def backup_path_from_node(self, node):
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
        path.reverse()
        return path