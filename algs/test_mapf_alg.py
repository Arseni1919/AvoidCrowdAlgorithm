import random
import matplotlib.pyplot as plt
import numpy as np

from funcs_plotter.plotter import Plotter
from funcs_graph.map_dimensions import map_dimensions_dict
from funcs_graph.nodes_from_pic import build_graph_nodes, get_dims_from_pic
from funcs_graph.heuristic_funcs import dist_heuristic, h_func_creator, build_heuristic_for_multiple_targets
from funcs_graph.heuristic_funcs import parallel_build_heuristic_for_multiple_targets


def test_mapf_alg_from_pic(algorithm, **kwargs):
    random_seed = kwargs['random_seed']
    seed = kwargs['seed']
    if random_seed:
        seed = random.choice(range(1000))
    print(f'SEED: {seed}')

    random.seed(seed)
    np.random.seed(seed)

    if 'img_dir' in kwargs:
        img_dir = kwargs['img_dir']
    else:
        # img_dir = 'lak109d.png'
        # img_dir = '19_20_warehouse.png'
        # img_dir = 'warehouse-10-20-10-2-1.png'
        # img_dir = 'warehouse-10-20-10-2-2.png'
        # img_dir = 'ht_chantry.png'
        # img_dir = 'Berlin_1_256.png'
        # img_dir = 'den101d.png'
        # img_dir = 'rmtst.png'
        # img_dir = 'lak505d.png'
        # img_dir = 'lak503d.png'
        # img_dir = 'ost003d.png'
        # img_dir = 'brc202d.png'
        # img_dir = 'den520d.png'

        # img_dir = 'random-64-64-10.map'  # 64-64
        img_dir = 'warehouse-10-20-10-2-1.map'  # 63-161
        # img_dir = 'lt_gallowstemplar_n.map'  # 180-251

        # img_dir = 'random-32-32-10.map'  # 32-32
        # img_dir = 'room-64-64-8.map'  # 64-64
        # img_dir = 'random-64-64-20.map'  # 64-64
        # img_dir = 'warehouse-10-20-10-2-2.map'  # 84-170
        # img_dir = 'warehouse-20-40-10-2-1.map'  # 123-321
        # img_dir = 'maze-128-128-2.map'  # 128-128
        # img_dir = 'ht_chantry.map'  # 141-162
        # img_dir = 'ost003d.map'  # 194-194
        # img_dir = 'lak303d.map'  # 194-194
        # img_dir = 'warehouse-20-40-10-2-2.map'  # 164-340
        # img_dir = 'Berlin_1_256.map'  # 256-256
        # img_dir = 'den520d.map'  # 257-256
        # img_dir = 'ht_mansion_n.map'  # 270-133
        # img_dir = 'brc202d.map'  # 481-530

    map_dim = get_dims_from_pic(img_dir=img_dir, path='../maps')
    nodes, nodes_dict = build_graph_nodes(img_dir=img_dir, path='../maps', show_map=False)
    # ------------------------- #
    # x_start, y_start = 97, 99
    # x_goal, y_goal = 38, 89
    # node_start = [node for node in nodes if node.x == x_start and node.y == y_start][0]
    # node_goal = [node for node in nodes if node.x == x_goal and node.y == y_goal][0]
    # ------------------------- #
    # node_start = nodes[100]
    # node_goal = nodes[-1]
    # ------------------------- #
    n_agents = kwargs['n_agents']
    # sample_nodes = random.sample(nodes, 2 * n_agents)
    # start_nodes = sample_nodes[:n_agents]
    # goal_nodes = sample_nodes[n_agents:]
    start_nodes = random.sample(nodes, n_agents)
    goal_nodes = random.sample(nodes, n_agents)
    # ------------------------- #
    # ------------------------- #
    plotter = Plotter(map_dim=map_dim)
    # ------------------------- #
    # ------------------------- #
    # h_dict = build_heuristic_for_multiple_targets(goal_nodes, nodes, map_dim, plotter=plotter, middle_plot=False)
    h_dict = parallel_build_heuristic_for_multiple_targets(goal_nodes, nodes, map_dim, plotter=plotter, middle_plot=False)
    h_func = h_func_creator(h_dict)

    # h_func = dist_heuristic

    # ------------------------- #
    # ------------------------- #
    plan, info = algorithm(start_nodes=start_nodes, goal_nodes=goal_nodes, nodes=nodes, nodes_dict=nodes_dict, h_func=h_func,
                           plotter=plotter, middle_plot=False, map_dim=map_dim, **kwargs)

    res_to_print = plan
    if plan:
        res_to_print = {k: len(v) for k, v in plan.items()}
    print(f'The result: {res_to_print}')
    # ------------------------- #
    # ------------------------- #
    # plt.show()
    # plt.close()
    return plan, info


def main():
    pass


if __name__ == '__main__':
    main()
