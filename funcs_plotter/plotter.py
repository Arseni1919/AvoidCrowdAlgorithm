import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from funcs_plotter.plot_functions import *


class Plotter:
    def __init__(self, map_dim=None, subplot_rows=2, subplot_cols=4):
        if map_dim:
            self.side_x, self.side_y = map_dim
        self.subplot_rows = subplot_rows
        self.subplot_cols = subplot_cols
        self.fig, self.ax = plt.subplots(subplot_rows, subplot_cols, figsize=(14, 7))

    def plot_lists(self, open_list, closed_list, start, goal=None, path=None, nodes=None, a_star_run=False):
        plt.close()
        self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 7))
        field = np.zeros((self.side_x, self.side_y))

        if nodes:
            for node in nodes:
                field[node.x, node.y] = -1

        for node in open_list:
            field[node.x, node.y] = 1

        for node in closed_list:
            field[node.x, node.y] = 2

        if path:
            for node in path:
                field[node.x, node.y] = 3

        field[start.x, start.y] = 4
        if goal:
            field[goal.x, goal.y] = 5

        self.ax[0].imshow(field, origin='lower')
        self.ax[0].set_title('general')

        # if path:
        #     for node in path:
        #         field[node.x, node.y] = 3
        #         self.ax[0].text(node.x, node.y, f'{node.ID}', bbox={'facecolor': 'yellow', 'alpha': 1, 'pad': 10})

        field = np.zeros((self.side_x, self.side_y))
        for node in open_list:
            if a_star_run:
                field[node.x, node.y] = node.g
            else:
                field[node.x, node.y] = node.g_dict[start.xy_name]
        self.ax[1].imshow(field, origin='lower')
        self.ax[1].set_title('open_list')

        field = np.zeros((self.side_x, self.side_y))
        for node in closed_list:
            if a_star_run:
                field[node.x, node.y] = node.g
            else:
                field[node.x, node.y] = node.g_dict[start.xy_name]
        self.ax[2].imshow(field, origin='lower')
        self.ax[2].set_title('closed_list')

        self.fig.tight_layout()
        # plt.pause(1)
        plt.pause(0.01)
        # plt.show()

    def plot_mapf_paths(self, paths_dict, nodes=None, **kwargs):
        plt.close()
        plot_per = kwargs['plot_per']
        plot_rate = kwargs['plot_rate']
        self.fig, self.ax = plt.subplots()
        longest_path = max([len(path) for path in paths_dict.values()])

        for t in range(longest_path):
            if t % plot_per == 0:
                info = {
                    'paths_dict': paths_dict, 'nodes': nodes,
                    'side_x': self.side_x, 'side_y': self.side_y, 't': t,
                    'img_dir': kwargs['img_dir'] if 'img_dir' in kwargs else '',
                }
                plot_step_in_mapf_paths(self.ax, info)
                # plt.pause(1)
                plt.pause(plot_rate)

    def plot_big_test(self, statistics_dict, runs_per_n_agents, algs_to_test_dict, n_agents_list, img_png='',
                      is_json=False, **kwargs):
        print('big plot starts')
        info = {
            'statistics_dict': statistics_dict,
            'runs_per_n_agents': runs_per_n_agents,
            'algs_to_test_dict': algs_to_test_dict,
            'n_agents_list': n_agents_list,
            'is_json': is_json
        }
        plot_success_rate(self.ax[0, 0], info)

        plot_sol_quality(self.ax[0, 1], info)

        plot_runtime_cactus(self.ax[0, 2], info)

        plot_a_star_calls_counters(self.ax[0, 3], info)

        plot_n_nei(self.ax[1, 0], info)

        plot_n_steps_iters(self.ax[1, 1], info)

        plot_n_messages(self.ax[1, 2], info)

        plot_n_expanded_cactus(self.ax[1, 3], info)

        time_per_alg_limit = f'{kwargs["time_per_alg_limit"]}' if 'time_per_alg_limit' in kwargs else '-'
        # self.fig.tight_layout()
        if 'i_run' in kwargs:
            title = f'{img_png}, {kwargs["i_run"]+1}/{runs_per_n_agents}, time limit: {float(time_per_alg_limit) * 60:0.0f} sec.'
        else:
            title = f'{img_png}'
        self.fig.suptitle(title, fontsize=16)
        plt.pause(0.1)
        print('big plot ends')
        # plt.show()
