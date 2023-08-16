from globals import *


def manhattan_distance_nodes(node1, node2):
    return abs(node1.x-node2.x) + abs(node1.y-node2.y)


def rename_nodes_in_path(path):
    for t, node in enumerate(path):
        node.t = t
        node.ID = f'{node.x}_{node.y}_{t}'

