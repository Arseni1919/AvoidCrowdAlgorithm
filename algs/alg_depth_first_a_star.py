from algs.alg_a_star_space_time import a_star
import random
import time
import numpy as np
import cProfile
import pstats
import matplotlib.pyplot as plt
from funcs_graph.map_dimensions import map_dimensions_dict
from funcs_graph.nodes_from_pic import build_graph_nodes
from funcs_plotter.plotter import Plotter
from funcs_graph.heuristic_funcs import h_func_creator, build_heuristic_for_multiple_targets


def df_a_star(start, goal, nodes, h_func, **kwargs):
    start_time = time.time()
    n_open = 0
    n_closed = 0
    initial_path, a_s_info = a_star(start, goal, nodes, h_func, df_dict={}, **kwargs)
    n_open += a_s_info['n_open']
    n_closed += a_s_info['n_closed']
    future_constr = a_s_info['future_constr']
    if initial_path:
        last_node = initial_path[-1]
        df_dict = {}
        iteration = 0
        while last_node.xy_name != goal.xy_name or future_constr:
            iteration += 1
            print(f'\ndf_a_star iteration {1}')
            df_dict.update({last_node.xy_name: True})
            if len(initial_path) - 1 != last_node.t:
                raise ValueError('len(initial_path) - 1 != last_node.t')
            new_path, a_s_info = a_star(last_node, goal, nodes, h_func, df_dict=df_dict, start_time=last_node.t, **kwargs)
            n_open += a_s_info['n_open']
            n_closed += a_s_info['n_closed']
            future_constr = a_s_info['future_constr']
            if new_path:
                initial_path.extend(new_path[1:])
                last_node = new_path[-1]
            else:
                return None, {'runtime': time.time() - start_time, 'n_open': n_open, 'n_closed': n_closed}
    return initial_path, {'runtime': time.time() - start_time, 'n_open': n_open, 'n_closed': n_closed}


def main():
    img_png = 'warehouse-10-20-10-2-1.png'
    map_dim = map_dimensions_dict[img_png]
    nodes, nodes_dict = build_graph_nodes(img_dir=img_png, path='../maps', show_map=False)
    # ------------------------- #
    # x_start, y_start = 97, 99
    # x_goal, y_goal = 38, 89
    # node_start = [node for node in nodes if node.x == x_start and node.y == y_start][0]
    # node_goal = [node for node in nodes if node.x == x_goal and node.y == y_goal][0]
    # ------------------------- #
    # node_start = nodes[100]
    # node_goal = nodes[-1]
    # ------------------------- #
    node_start = random.choice(nodes)
    node_goal = random.choice(nodes)
    print(f'start: {node_start.x}, {node_start.y} -> goal: {node_goal.x}, {node_goal.y}')
    # ------------------------- #
    # ------------------------- #
    plotter = Plotter(map_dim=map_dim, subplot_rows=1, subplot_cols=3)
    # plotter = None
    # ------------------------- #
    # ------------------------- #
    h_dict = build_heuristic_for_multiple_targets([node_goal], nodes, map_dim, plotter=plotter, middle_plot=False)
    h_func = h_func_creator(h_dict)
    # ------------------------- #
    # h_func = dist_heuristic
    # ------------------------- #
    # ------------------------- #
    # constraint_dict = None
    v_constr_dict = {node.xy_name: [] for node in nodes}
    # v_constr_dict = {'30_12': [69], '29_12': [68, 69]}
    perm_constr_dict = {node.xy_name: [] for node in nodes}
    perm_constr_dict['74_9'].append(10)
    # ------------------------- #
    # ------------------------- #
    # result = a_star(start=node_start, goal=node_goal, nodes=nodes, h_func=h_func, plotter=plotter, middle_plot=False)
    profiler.enable()
    result, info = df_a_star(node_start, node_goal, nodes, h_func,
                             v_constr_dict=v_constr_dict, perm_constr_dict=perm_constr_dict,
                             plotter=plotter, middle_plot=True, nodes_dict=nodes_dict)
    profiler.disable()
    if result:
        print('The result is:', *[node.xy_name for node in result], sep='->')
        print('The result is:', *[node.ID for node in result], sep='->')
    # ------------------------- #
    # ------------------------- #
    plt.show()
    # plt.close()


if __name__ == '__main__':
    # random_seed = True
    random_seed = False
    seed = random.choice(range(1000)) if random_seed else 121
    random.seed(seed)
    np.random.seed(seed)
    print(f'SEED: {seed}')
    # main()
    profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats.print_stats()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('../stats/results_df_a_star.pstat')
