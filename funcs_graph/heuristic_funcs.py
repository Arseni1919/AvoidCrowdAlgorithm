import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import concurrent.futures
import threading
import asyncio
import logging

from simulator_objects import ListNodes
from funcs_graph.map_dimensions import map_dimensions_dict
from funcs_graph.nodes_from_pic import build_graph_nodes, get_dims_from_pic
from funcs_plotter.plotter import Plotter


def dist_heuristic(from_node, to_node):
    return np.abs(from_node.x - to_node.x) + np.abs(from_node.y - to_node.y)
    # return np.sqrt((from_node.x - to_node.x) ** 2 + (from_node.y - to_node.y) ** 2)


def h_func_creator(h_dict):
    def h_func(from_node, to_node):
        if to_node.xy_name in h_dict:
            h_value = h_dict[to_node.xy_name][from_node.x, from_node.y]
            if h_value > 0:
                return h_value
        return np.abs(from_node.x - to_node.x) + np.abs(from_node.y - to_node.y)
        # return np.sqrt((from_node.x - to_node.x) ** 2 + (from_node.y - to_node.y) ** 2)
    return h_func


def get_node(successor_xy_name, node_current, nodes_dict):
    if node_current.xy_name == successor_xy_name:
        return None
    return nodes_dict[successor_xy_name]
    # for node in nodes:
    #     if node.xy_name == successor_xy_name and node_current.xy_name != successor_xy_name:
    #         return node
    # return None


def parallel_update_h_table(node, nodes, map_dim, to_save, plotter, middle_plot, h_dict, node_index):
    print(f'[HEURISTIC]: Thread {node_index} started.')
    h_table = build_heuristic_for_one_target(node, nodes, map_dim, to_save, plotter, middle_plot)
    h_dict[node.xy_name] = h_table
    print(f'[HEURISTIC]: Thread {node_index} finished.')


def parallel_build_heuristic_for_multiple_targets(target_nodes, nodes, map_dim, to_save=True, plotter=None, middle_plot=False):
    print('Started to build heuristic...')
    h_dict = {}
    reset_nodes(nodes, target_nodes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(target_nodes)) as executor:
        for node_index, node in enumerate(target_nodes):
            executor.submit(parallel_update_h_table, node, nodes, map_dim, to_save, plotter, middle_plot, h_dict, node_index)

    print(f'\nFinished to build heuristic for all nodes.')
    return h_dict


def build_heuristic_for_multiple_targets(target_nodes, nodes, map_dim, to_save=True, plotter=None, middle_plot=False):
    print('Started to build heuristic...')
    h_dict = {}
    reset_nodes(nodes, target_nodes)
    iteration = 0
    for node in target_nodes:
        h_table = build_heuristic_for_one_target(node, nodes, map_dim, to_save, plotter, middle_plot)
        h_dict[node.xy_name] = h_table

        print(f'\nFinished to build heuristic for node {iteration}.')
        iteration += 1
    return h_dict


def reset_nodes(nodes, target_nodes=None):
    _ = [node.reset(target_nodes) for node in nodes]


def build_heuristic_for_one_target(target_node, nodes, map_dim, to_save=True, plotter=None, middle_plot=False):
    # print('Started to build heuristic...')
    copy_nodes = nodes
    nodes_dict = {node.xy_name: node for node in copy_nodes}
    target_name = target_node.xy_name
    target_node = nodes_dict[target_name]
    # target_node = [node for node in copy_nodes if node.xy_name == target_node.xy_name][0]
    # open_list = []
    # close_list = []
    open_nodes = ListNodes(target_name=target_node.xy_name)
    closed_nodes = ListNodes(target_name=target_node.xy_name)
    # open_list.append(target_node)
    open_nodes.add(target_node)
    iteration = 0
    # while len(open_list) > 0:
    while len(open_nodes) > 0:
        iteration += 1
        # node_current = get_node_from_open(open_list, target_name)
        node_current = open_nodes.pop()
        # if node_current.xy_name == '30_12':
        #     print()
        for successor_xy_name in node_current.neighbours:
            node_successor = get_node(successor_xy_name, node_current, nodes_dict)
            if node_successor:
                successor_current_g = node_current.g_dict[target_name] + 1  # h(now, next)

                # INSIDE OPEN LIST
                if node_successor.xy_name in open_nodes.dict:
                    if node_successor.g_dict[target_name] <= successor_current_g:
                        continue
                    open_nodes.remove(node_successor)
                    node_successor.g_dict[target_name] = successor_current_g
                    node_successor.parent = node_current
                    open_nodes.add(node_successor)

                # INSIDE CLOSED LIST
                elif node_successor.xy_name in closed_nodes.dict:
                    if node_successor.g_dict[target_name] <= successor_current_g:
                        continue
                    closed_nodes.remove(node_successor)
                    node_successor.g_dict[target_name] = successor_current_g
                    node_successor.parent = node_current
                    open_nodes.add(node_successor)

                # NOT IN CLOSED AND NOT IN OPEN LISTS
                else:
                    node_successor.g_dict[target_name] = successor_current_g
                    node_successor.parent = node_current
                    open_nodes.add(node_successor)

                # node_successor.g_dict[target_name] = successor_current_g
                # node_successor.parent = node_current

        # open_nodes.remove(node_current, target_name=target_node.xy_name)
        closed_nodes.add(node_current)

        if plotter and middle_plot and iteration % 1000 == 0:
            plotter.plot_lists(open_list=open_nodes.get_nodes_list(),
                               closed_list=closed_nodes.get_nodes_list(), start=target_node, nodes=copy_nodes)
        if iteration % 100 == 0:
            print(f'\riter: {iteration}', end='')

    if plotter and middle_plot:
        plotter.plot_lists(open_list=open_nodes.get_nodes_list(),
                           closed_list=closed_nodes.get_nodes_list(), start=target_node, nodes=copy_nodes)

    h_table = np.zeros(map_dim)
    for node in copy_nodes:
        h_table[node.x, node.y] = node.g_dict[target_name]
    # h_dict = {target_node.xy_name: h_table}
    # print(f'\rFinished to build heuristic at iter {iteration}.')
    return h_table


def main():
    # img_dir = 'den520d.png'
    img_dir = 'Berlin_1_256.map'  # 256-256
    map_dim = get_dims_from_pic(img_dir=img_dir, path='../maps')
    nodes, nodes_dict = build_graph_nodes(img_dir=img_dir, path='../maps', show_map=False)
    x_goal, y_goal = 38, 89
    node_goal = [node for node in nodes if node.x == x_goal and node.y == y_goal][0]
    plotter = Plotter(map_dim=map_dim, subplot_rows=1, subplot_cols=3)
    reset_nodes(nodes, target_nodes=[node_goal])
    h_table = build_heuristic_for_one_target(node_goal, nodes, map_dim, plotter=plotter, middle_plot=True)
    # h_table = build_heuristic_for_one_target(node_goal, nodes, map_dim, plotter=plotter, middle_plot=False)
    print(h_table)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()


# class ParallelHDict:
#     def __init__(self):
#         self.h_dict = {}
#         self._lock = threading.Lock()


# def get_node_from_open(open_list, target_name):
#     v_list = open_list
#     v_dict = {}
#     v_g_list = []
#     for node in v_list:
#         curr_g = node.g_dict[target_name]
#         v_g_list.append(curr_g)
#         if curr_g not in v_dict:
#             v_dict[curr_g] = []
#         v_dict[curr_g].append(node)
#
#     # some_v = min([node.g for node in v_list])
#     # out_nodes = [node for node in v_list if node.g == some_v]
#     # print([node.ID for node in t_nodes])
#     # next_node = random.choice(out_nodes)
#     next_node = random.choice(v_dict[min(v_g_list)])
#     return next_node
