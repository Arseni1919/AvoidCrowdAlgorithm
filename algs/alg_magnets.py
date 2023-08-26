import random
from typing import List

import cProfile
import pstats

import numpy as np

from functions import *

from algs.alg_a_star_space_time import a_star_xyt
from algs.alg_a_star_space import a_star_xy
from algs.test_mapf_alg import test_mapf_alg_from_pic
from algs.metrics import c_v_check_for_agent, c_e_check_for_agent, build_constraints, \
    get_agents_in_conf, check_plan, get_alg_info_dict, iteration_print
from algs.metrics import just_check_k_step_plans, just_check_plans
from algs.metrics import check_single_agent_k_step_c_v, check_single_agent_k_step_c_e
from algs.metrics import build_k_step_perm_constr_dict
from funcs_plotter.plotter import Plotter


class MagnetsAgent:
    def __init__(self, index, start_node, goal_node, nodes, nodes_dict, h_func,
                 plotter, middle_plot,
                 iter_limit=1e100, map_dim=None):
        self.index = index
        self.name = f'agent_{index}'
        self.start_node = start_node
        self.curr_node = start_node
        self.next_node = start_node
        self.goal_node = goal_node
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.nei_nodes = []
        self.nei_nodes_dict = {}
        self.h_func = h_func
        self.plotter = plotter
        self.middle_plot = middle_plot
        self.iter_limit = iter_limit
        self.path = []
        self.path_names = []
        self.full_path = []
        self.full_path_names = []
        self.map_dim = map_dim

        # stats
        self.stats_n_closed = 0
        self.stats_n_calls = 0
        self.stats_runtime = 0
        self.stats_n_messages = 0
        self.stats_n_step_m = 0
        self.stats_n_step_m_list = []
        self.stats_nei_list = []
        # nei
        self.nei_list = []
        self.nei_dict = {}
        self.nei_info_dict = {}

        # map
        self.b_map = np.ones(self.map_dim) * -1
        self.b_my_magnet_mask = np.zeros(self.map_dim)
        self.b_full_magnet_field = None
        self.b_my_magnet_list = []
        self.init_grid_weights()

    def my_h_func_creator(self):
        my_h = self.b_map[self.curr_node.x, self.curr_node.y]
        new_h_table = np.copy(self.b_full_magnet_field) / np.max(self.b_full_magnet_field) * my_h

        def h_func(from_node, to_node):
            h_value = new_h_table[from_node.x, from_node.y]
            # h_value = self.b_map[from_node.x, from_node.y]
            return h_value

        return h_func

    def get_nei_lowest_point(self):
        map_to_use = self.b_full_magnet_field
        # map_to_use = self.b_map
        lowest_node = self.nei_nodes[0]
        lowest_value = map_to_use[lowest_node.x, lowest_node.y]
        for node in self.nei_nodes:
            curr_value = map_to_use[node.x, node.y]
            if curr_value < lowest_value:
                lowest_node = node
                lowest_value = curr_value
        return lowest_node, lowest_value

    def calc_a_star_plan(self, v_constr_dict=None, e_constr_dict=None, perm_constr_dict=None, k_time=None, **kwargs):
        start_time = time.time()
        lowest_node, lowest_value = self.get_nei_lowest_point()
        goal = lowest_node
        new_path, a_s_info = a_star_xy(start=self.curr_node, goal=goal, nodes=self.nei_nodes,
                                       nodes_dict=self.nei_nodes_dict,
                                       h_func=self.my_h_func_creator(), iter_limit=self.iter_limit, **kwargs)

        # goal = self.goal_node
        # new_path, a_s_info = a_star_xy(start=self.curr_node, goal=goal, nodes=self.nodes,
        #                                nodes_dict=self.nodes_dict,
        #                                h_func=self.my_h_func_creator(), iter_limit=self.iter_limit, **kwargs)

        # if not v_constr_dict:
        #     v_constr_dict = {node.xy_name: [] for node in self.nodes}
        # if not e_constr_dict:
        #     e_constr_dict = {node.xy_name: [] for node in self.nodes}
        # if not perm_constr_dict:
        #     perm_constr_dict = {node.xy_name: [] for node in self.nodes}
        # lowest_node, lowest_value = self.get_the_lowest_point()
        # new_path, a_s_info = a_star_xyt(start=self.curr_node, goal=lowest_node, nodes=self.nodes,
        #                                 nodes_dict=self.nodes_dict, h_func=self.my_h_func_creator(),
        #                                 v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict,
        #                                 perm_constr_dict=perm_constr_dict, iter_limit=self.iter_limit)
        if new_path is not None:
            self.path = new_path
            succeeded = True
        else:
            succeeded = False
        # rename_nodes_in_path(self.path)
        self.path_names = [node.xy_name for node in self.path]
        return succeeded, {'a_s_time': time.time() - start_time, 'a_s_info': a_s_info}

    def init_grid_weights(self):
        # h_func(node_successor, goal)
        for node in self.nodes:
            self.b_map[node.x, node.y] = self.h_func(node, self.goal_node)

    def update_nei(self, agents, **kwargs):
        nei_r = kwargs['k']
        # nei_dist_const = 2 * nei_r + 1
        self.nei_list, self.nei_dict, self.nei_info_dict = [], {}, {}
        for agent in agents:
            if agent.name != self.name:
                # self.nei_list.append(agent)
                # self.nei_dict[agent.name] = agent
                # self.nei_info_dict[agent.name] = {}
                # curr_distance = manhattan_distance_nodes(self.curr_node, agent.curr_node)
                curr_distance = euclidean_distance_nodes(self.curr_node, agent.curr_node)
                if curr_distance <= nei_r:
                    self.nei_list.append(agent)
                    self.nei_dict[agent.name] = agent
                    self.nei_info_dict[agent.name] = {}
        self.stats_nei_list.append(len(self.nei_list) - 1)

        self.nei_nodes = []
        self.nei_nodes_dict = {}
        for node in self.nodes:
            curr_distance = euclidean_distance_nodes(self.curr_node, node)
            if curr_distance <= nei_r:
                self.nei_nodes.append(node)
                self.nei_nodes_dict[node.xy_name] = node

    def set_area_circle(self):
        self.b_my_magnet_mask = np.zeros(self.map_dim)
        if self.curr_node.xy_name == self.goal_node.xy_name:
            self.b_my_magnet_mask[self.curr_node.x, self.curr_node.y] += 100
        max_r = len(self.b_my_magnet_list)
        # max_r = min(5, len(self.b_my_magnet_list))
        # for i_node in self.nodes:
        for i_node in self.nei_nodes:
            if abs(i_node.x - self.curr_node.x) > max_r or abs(i_node.y - self.curr_node.y) > max_r:
                continue
            # around the curr_node
            distance = math.floor(euclidean_distance_nodes(i_node, self.curr_node))
            if distance < max_r:
                self.b_my_magnet_mask[i_node.x, i_node.y] += self.b_my_magnet_list[distance]
                # self.b_my_magnet_mask[i_node.x, i_node.y] += 1

    def set_area_line(self):
        self.b_my_magnet_mask = np.zeros(self.map_dim)
        next_node = self.curr_node
        for m_value_index, m_value in enumerate(self.b_my_magnet_list):
            self.b_my_magnet_mask[next_node.x, next_node.y] += m_value
            next_pos_dict = {}
            for i_next_pos_name in next_node.neighbours:
                i_next_pos = self.nodes_dict[i_next_pos_name]
                next_pos_dict[i_next_pos_name] = self.b_map[i_next_pos.x, i_next_pos.y]
                # next_pos_dict[i_next_pos_name] = self.b_full_magnet_field[i_next_pos.x, i_next_pos.y]
            min_value = min(next_pos_dict.values())
            min_pos_list = [k for k, v in next_pos_dict.items() if v == min_value]
            next_pos_name = random.choice(min_pos_list)
            next_node = self.nodes_dict[next_pos_name]
            if m_value_index > 4:
                break

    def set_area_spear(self):
        self.b_my_magnet_mask = np.zeros(self.map_dim)
        max_r = len(self.b_my_magnet_list)
        # max_r = min(5, len(self.b_my_magnet_list))
        curr_h_value = self.b_map[self.curr_node.x, self.curr_node.y]
        # for i_node in self.nodes:
        for i_node in self.nei_nodes:
            if abs(i_node.x - self.curr_node.x) > max_r or abs(i_node.y - self.curr_node.y) > max_r:
                continue
            # spear (---->-) from the curr_node
            # distance = math.floor(euclidean_distance_nodes(i_node, self.curr_node))
            distance = manhattan_distance_nodes(i_node, self.curr_node)
            if distance < max_r:
                i_h_value = self.b_map[i_node.x, i_node.y]
                diff = curr_h_value - i_h_value
                if 0 <= diff < max_r:
                    self.b_my_magnet_mask[i_node.x, i_node.y] += self.b_my_magnet_list[distance]

    def set_my_magnetism(self):
        self.b_my_magnet_list = []
        if self.curr_node.xy_name != self.goal_node.xy_name:
            h_value = self.b_map[self.curr_node.x, self.curr_node.y]
            # h_value = h_value*4
            self.b_my_magnet_list.append(h_value)
            while h_value > 0.5:
                # h_value /= 4
                h_value /= 2
                # h_value /= 1.5
                # h_value /= 1.2
                self.b_my_magnet_list.append(h_value)
        # set area
        # self.set_area_circle()
        self.set_area_line()
        # self.set_area_spear()

    def exchange_info(self, **kwargs):
        # set b_my_magnet_list
        self.set_my_magnetism()

        for nei in self.nei_list:
            nei.nei_info_dict[self.name]['path'] = self.path
            nei.nei_info_dict[self.name]['curr_node'] = self.curr_node
            nei.nei_info_dict[self.name]['next_node'] = self.next_node
            nei.nei_info_dict[self.name]['b_magnet_list'] = self.b_my_magnet_list
            nei.nei_info_dict[self.name]['b_magnet_mask'] = self.b_my_magnet_mask
            nei.nei_info_dict[self.name]['h'] = self.b_map[self.curr_node.x, self.curr_node.y]

    def build_full_magnet_field(self):
        self.b_full_magnet_field = np.copy(self.b_map)
        for nei in self.nei_list:
            nei_magnet_mask = self.nei_info_dict[nei.name]['b_magnet_mask']
            self.b_full_magnet_field += nei_magnet_mask
            nei_curr_node = self.nei_info_dict[nei.name]['curr_node']
            self.b_full_magnet_field[nei_curr_node.x, nei_curr_node.y] += 50
            # nei_next_node = self.nei_info_dict[nei.name]['next_node']
            # self.b_full_magnet_field[nei_next_node.x, nei_next_node.y] += 50

    def plan(self, **kwargs):
        alpha = kwargs['alpha']
        self.build_full_magnet_field()
        self.calc_a_star_plan()

        if len(self.path) > 1:
            self.next_node = self.path[1]
        else:
            self.next_node = self.path[0]
        if len(self.nei_list) > 0 and self.curr_node.xy_name != self.goal_node.xy_name and random.random() < alpha:
            self.next_node = self.curr_node
            # next_pos_name = random.choice(self.curr_node.neighbours)
            # self.next_node = self.nodes_dict[next_pos_name]

        # if self.name == 'agent_0':
        #     print(f'\nagent_0 curr pos -> {self.curr_node.xy_name}')
        #     print(f'agent_0 next pos -> {self.next_node.xy_name}\n')

        # next_pos_dict = {}
        # for i_next_pos_name in self.curr_node.neighbours:
        #     i_next_pos = self.nodes_dict[i_next_pos_name]
        #     next_pos_dict[i_next_pos_name] = self.b_full_magnet_field[i_next_pos.x, i_next_pos.y]
        # # next_pos_name = min(next_pos_dict, key=next_pos_dict.get)
        # min_value = min(next_pos_dict.values())
        # min_pos_list = [k for k, v in next_pos_dict.items() if v == min_value]
        # next_pos_name = random.choice(min_pos_list)
        # del next_pos_dict[next_pos_name]
        # if len(self.nei_list) > 0 and self.curr_node.xy_name != self.goal_node.xy_name and random.random() < alpha:
        #     # next_pos_name = min(next_pos_dict, key=next_pos_dict.get)
        #     next_pos_name = random.choice(list(next_pos_dict.keys()))
        # self.next_node = self.nodes_dict[next_pos_name]

        return True, {}

    def correct_next_step(self):
        my_h = self.b_map[self.curr_node.x, self.curr_node.y]
        for nei in self.nei_list:
            nei_curr_node = self.nei_info_dict[nei.name]['curr_node']
            next_nei_node = self.nei_info_dict[nei.name]['next_node']
            nei_h = self.nei_info_dict[nei.name]['h']
            # v_c + v_e
            if next_nei_node.xy_name == self.next_node.xy_name or nei_curr_node.xy_name == self.next_node.xy_name:
                self.next_node = self.curr_node
                return
            # e_c
            # if next_nei_node.xy_name == self.curr_node.xy_name and nei_curr_node.xy_name == self.next_node.xy_name:
            #     self.next_node = self.curr_node
            #     return

    def make_step(self, **kwargs):
        self.correct_next_step()
        self.curr_node = self.next_node
        # print(f'{self.name} goes to {self.curr_node.xy_name}')
        self.full_path.append(self.curr_node)
        self.full_path_names = [node.xy_name for node in self.full_path]
        return self.curr_node.xy_name == self.goal_node.xy_name


def create_agents(start_nodes, goal_nodes, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit, map_dim):
    # Creating agents
    agents = []
    agents_dict = {}
    n_agent = 0
    for start_node, goal_node in zip(start_nodes, goal_nodes):
        agent = MagnetsAgent(n_agent, start_node, goal_node, nodes, nodes_dict, h_func, plotter, middle_plot,
                             iter_limit,
                             map_dim)
        agents.append(agent)
        agents_dict[agent.name] = agent
        n_agent += 1

    return agents, agents_dict


def all_find_nei(agents: List[MagnetsAgent], **kwargs):
    runtime, runtime_dist = 0, []
    succeeded_list = []
    for agent in agents:
        # find_nei
        start_time = time.time()
        agent.update_nei(agents, **kwargs)

        # stats
        end_time = time.time() - start_time
        runtime += end_time
        runtime_dist.append(end_time)

    func_info = {
        'runtime': runtime,
        'dist_runtime': max(runtime_dist),
        'all_succeeded': all(succeeded_list),
    }
    return func_info


def all_plan(agents: List[MagnetsAgent], alg_info, **kwargs):
    i_run_str = f'[i_run: {kwargs["i_run"]}]' if 'i_run' in kwargs else ''
    img_str = f'[{kwargs["img_dir"]}]' if 'img_dir' in kwargs else ''
    print(f'\n\n[runtime={alg_info["runtime"]:0.2f} sec.][dist_runtime={alg_info["dist_runtime"]:0.2f} sec.]'
          f'\n{img_str}({kwargs["alg_name"]}){i_run_str}[finished: {kwargs["number_of_finished"]}/{kwargs["n_agents"]}]'
          f'[step: {kwargs["step_iteration"]}]\n')

    runtime, runtime_dist = 0, []
    succeeded_list = []
    for agent in agents:
        # create initial plan
        start_time = time.time()
        succeeded, info = agent.plan(**kwargs)
        # stats
        end_time = time.time() - start_time
        runtime += end_time
        runtime_dist.append(end_time)
        succeeded_list.append(succeeded)

    func_info = {
        'runtime': runtime,
        'dist_runtime': max(runtime_dist),
        'all_succeeded': all(succeeded_list),
    }
    return func_info


def all_move_a_step(agents: List[MagnetsAgent], **kwargs):
    all_paths_are_finished_list = []
    for agent in agents:
        agent_is_finished = agent.make_step(**kwargs)
        all_paths_are_finished_list.append(agent_is_finished)

    number_of_finished = sum(all_paths_are_finished_list)
    all_paths_are_finished = all(all_paths_are_finished_list)
    func_info = {}
    return all_paths_are_finished, number_of_finished, func_info


def all_exchange_info(agents: List[MagnetsAgent], **kwargs):
    # k = kwargs['k']
    # check_radius = k
    runtime, runtime_dist = 0, []

    # exchange paths
    for agent in agents:
        start_time = time.time()
        agent.exchange_info(**kwargs)
        # stats
        end_time = time.time() - start_time
        runtime += end_time
        runtime_dist.append(end_time)

    # check for collisions
    # plans = {agent.name: agent.path for agent in agents}
    # there_are_collisions, c_v, c_e = just_check_k_step_plans(plans, check_radius + 1, immediate=True)
    # there_are_collisions, c_v, c_e = just_check_k_step_plans(plans, check_radius+1, immediate=False)

    func_info = {
        'runtime': runtime,
        'dist_runtime': max(runtime_dist)
    }
    # return there_are_collisions, c_v, c_e, func_info
    return False, [], [], func_info


def run_magnets(start_nodes, goal_nodes, nodes, nodes_dict, h_func, **kwargs):
    if 'k' not in kwargs:
        raise RuntimeError("'k' is not in kwargs")
    if 'k_step_iteration_limit' in kwargs:
        k_step_iteration_limit = kwargs['k_step_iteration_limit']
    else:
        k_step_iteration_limit = 1000
        kwargs['k_step_iteration_limit'] = k_step_iteration_limit
        kwargs['k_small_iter_limit'] = 40
    alg_name = kwargs['alg_name'] if 'alg_name' in kwargs else f'k-SDS'
    iter_limit = kwargs['a_star_iter_limit'] if 'a_star_iter_limit' in kwargs else 1e100
    map_dim = kwargs['map_dim'] if 'map_dim' in kwargs else None
    middle_plot = kwargs['middle_plot'] if 'middle_plot' in kwargs else False
    final_plot = kwargs['final_plot'] if 'final_plot' in kwargs else True
    img_dir = kwargs['img_dir'] if 'img_dir' in kwargs else ''
    inner_plot = kwargs['inner_plot'] if 'inner_plot' in kwargs else False
    plotter = kwargs['plotter'] if 'plotter' in kwargs else None
    # plotter = None
    if plotter:
        plotter.close()
    plotter = Plotter(map_dim=map_dim, subplot_rows=1, subplot_cols=2)
    stats_small_iters_list = []
    number_of_finished = 0

    # Creating agents
    agents, agents_dict = create_agents(start_nodes, goal_nodes, nodes, nodes_dict, h_func, plotter, middle_plot,
                                        iter_limit, map_dim)
    kwargs['n_agents'] = len(agents)

    # alg info dict
    alg_info = get_alg_info_dict()

    # Distributed Part
    for step_iteration in range(1000000):
        kwargs['step_iteration'] = step_iteration
        kwargs['small_iteration'] = 0
        kwargs['number_of_finished'] = number_of_finished

        func_info = all_find_nei(agents, **kwargs)  # agents - find nei
        if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
            return None, {'agents': agents, 'success_rate': 0}

        _, _, _, func_info = all_exchange_info(agents, **kwargs)  # agents - exchange
        if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
            return None, {'agents': agents, 'success_rate': 0}

        func_info = all_plan(agents, alg_info, **kwargs)  # agents - plan
        if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
            return None, {'agents': agents, 'success_rate': 0}

        _, _, _, func_info = all_exchange_info(agents, **kwargs)  # agents - exchange
        if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
            return None, {'agents': agents, 'success_rate': 0}

        all_paths_are_finished, number_of_finished, func_info = all_move_a_step(agents, **kwargs)  # agents - step
        if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
            return None, {'agents': agents, 'success_rate': 0}

        # plot
        if step_iteration > 0 and plotter and inner_plot:
            full_plans = {agent.name: agent.full_path for agent in agents}
            plotter.plot_magnets_run(paths_dict=full_plans, nodes=nodes, t=step_iteration, img_dir=img_dir,
                                     agent=agents[0])

        full_plans = {agent.name: agent.full_path for agent in agents}
        # iteration_print(agents, full_plans, alg_name, alg_info, alg_info['runtime'], step_iteration)
        if all_paths_are_finished:
            there_is_col, c_v, c_e, cost = just_check_plans(full_plans)
            if there_is_col:
                raise RuntimeError('uff')
            else:
                if final_plot:
                    print(f'#########################################################')
                    print(f'#########################################################')
                    print(f'#########################################################')
                    print(f"runtime: {alg_info['runtime']}\n{alg_info['dist_runtime']=}\n{cost=}")
                    print(f"a_star_n_closed: {sum(alg_info['a_star_n_closed'])}\n{alg_info['a_star_n_closed_dist']=}")
                    # plotter.plot_mapf_paths(paths_dict=full_plans, nodes=nodes, **kwargs)
                alg_info['success_rate'] = 1
                alg_info['sol_quality'] = cost
                alg_info['n_messages'] = np.sum([agent.stats_n_messages for agent in agents])
                alg_info['m_per_step'] = np.sum([np.mean(agent.stats_n_step_m_list) for agent in agents])
                alg_info['n_steps'] = step_iteration + 1
                alg_info['n_nei'] = np.sum([np.mean(agent.stats_nei_list) for agent in agents])
                # alg_info['avr_n_nei'] = np.mean([np.mean(agent.stats_nei_list) for agent in agents])
            return full_plans, alg_info

        if step_iteration > k_step_iteration_limit - 1:
            print(f'\n[LIMIT]: step_iteration: {step_iteration} > limit: {k_step_iteration_limit}')
            break

    return None, {'agents': agents, 'success_rate': 0}


def main():
    n_agents = 300
    # img_dir = 'my_map_10_10_room.map'  # 10-10
    img_dir = 'empty-48-48.map'  # 48-48
    # img_dir = 'random-64-64-10.map'  # 64-64
    # img_dir = 'warehouse-10-20-10-2-1.map'  # 63-161
    # img_dir = 'lt_gallowstemplar_n.map'  # 180-251
    # img_dir = 'random-32-32-10.map'  # 32-32               | LNS |
    # img_dir = 'ht_chantry.map'  # 162-141   | Up to 230 agents with h=w=30, lim=10sec.

    # random_seed = True
    random_seed = False
    seed = 878
    PLOT_PER = 1
    PLOT_RATE = 0.5

    # --------------------------------------------------- #
    # --------------------------------------------------- #
    # for the algorithms
    k = 10
    alpha = 0.1
    alg_name = f'Magnet'
    inner_plot = True
    # inner_plot = False

    # --------------------------------------------------- #
    # --------------------------------------------------- #

    to_use_profiler = True
    # to_use_profiler = False
    profiler = cProfile.Profile()
    if to_use_profiler:
        profiler.enable()
    for i in range(3):
        print(f'\n[run {i}]')
        result, info = test_mapf_alg_from_pic(
            algorithm=run_magnets,
            img_dir=img_dir,
            alg_name=alg_name,
            k=k,
            alpha=alpha,
            n_agents=n_agents,
            random_seed=random_seed,
            seed=seed,
            final_plot=True,
            a_star_iter_limit=5e7,
            # limit_type='norm_time',
            limit_type='dist_time',
            max_time=50,
            a_star_closed_nodes_limit=1e6,
            inner_plot=inner_plot,
            plot_per=PLOT_PER,
            plot_rate=PLOT_RATE,
        )

        if not random_seed:
            break

        # plt.show()
        plt.close()

    if to_use_profiler:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        dir_to_save = f'../stats/results_magnets.pstat'
        stats.dump_stats(dir_to_save)
        print(f'[STATS]: stats are saved in {dir_to_save}')


if __name__ == '__main__':
    main()

# def build_full_magnet_field(self):
#     self.b_full_magnet_field = np.copy(self.b_map)
#     for nei in self.nei_list:
#         nei_curr_node = self.nei_info_dict[nei.name]['curr_node']
#         nei_magnet_list = self.nei_info_dict[nei.name]['b_magnet_list']
#         open_list, next_open_list, closed_list = [nei_curr_node.xy_name], [], []
#         for m_value in nei_magnet_list:
#             next_gen_list = []
#             while len(open_list) > 0:
#                 i_name = open_list.pop()
#                 next_gen_list.append(i_name)
#                 i_node = self.nodes_dict[i_name]
#                 for near_name in i_node.neighbours:
#                     if near_name not in closed_list:
#                         next_open_list.append(near_name)
#             for i_name in next_gen_list:
#                 i_node = self.nodes_dict[i_name]
#                 self.b_full_magnet_field[i_node.x, i_node.y] += m_value
#                 closed_list.append(i_name)
#             open_list = next_open_list
#             next_open_list = []
