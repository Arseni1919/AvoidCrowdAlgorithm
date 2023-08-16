import random
from typing import List

import cProfile
import pstats

import numpy as np

from functions import *

from algs.alg_a_star_space_time import a_star
from algs.test_mapf_alg import test_mapf_alg_from_pic
from algs.metrics import c_v_check_for_agent, c_e_check_for_agent, build_constraints, \
    limit_is_crossed, get_agents_in_conf, check_plan, get_alg_info_dict, iteration_print
from algs.metrics import just_check_k_step_plans, just_check_plans
from algs.metrics import check_single_agent_k_step_c_v, check_single_agent_k_step_c_e
from algs.metrics import build_k_step_perm_constr_dict


class KSDSAgent:
    def __init__(self, index, start_node, goal_node, nodes, nodes_dict, h_func,
                 plotter, middle_plot,
                 iter_limit=1e100, map_dim=None):
        self.index = index
        self.name = f'agent_{index}'
        self.start_node = start_node
        self.curr_node = start_node
        self.goal_node = goal_node
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.h_func = h_func
        self.plotter = plotter
        self.middle_plot = middle_plot
        self.iter_limit = iter_limit
        self.path = []
        self.path_names = []
        self.last_path_change_iter = 0
        self.h = 0
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
        self.nei_paths_dict = {}
        self.nei_h_dict = {}
        self.conf_agents_names = []
        self.names_to_consider_list = []

    def calc_a_star_plan(self, v_constr_dict=None, e_constr_dict=None, perm_constr_dict=None, k_time=None, **kwargs):
        start_time = time.time()
        if not v_constr_dict:
            v_constr_dict = {node.xy_name: [] for node in self.nodes}
        if not e_constr_dict:
            e_constr_dict = {node.xy_name: [] for node in self.nodes}
        if not perm_constr_dict:
            perm_constr_dict = {node.xy_name: [] for node in self.nodes}
        if kwargs["small_iteration"] > 100:
            print(f'\n ---------- ({kwargs["alg_name"]}) '
                  f'[finished: {kwargs["number_of_finished"]}]'
                  f'[step: {kwargs["k_step_iteration"]}]'
                  f'[iter: {kwargs["small_iteration"]}] A* {self.name} ---------- \n')
        if k_time:
            new_path, a_s_info = a_star(start=self.curr_node, goal=self.goal_node, nodes=self.nodes,
                                        nodes_dict=self.nodes_dict, h_func=self.h_func,
                                        v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict,
                                        perm_constr_dict=perm_constr_dict,
                                        plotter=self.plotter, middle_plot=self.middle_plot,
                                        iter_limit=self.iter_limit, k_time=k_time)
        else:
            new_path, a_s_info = a_star(start=self.curr_node, goal=self.goal_node, nodes=self.nodes,
                                        nodes_dict=self.nodes_dict, h_func=self.h_func,
                                        v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict,
                                        perm_constr_dict=perm_constr_dict,
                                        plotter=self.plotter, middle_plot=self.middle_plot, iter_limit=self.iter_limit)
        if new_path is not None:
            self.path = new_path
            self.h = self.path[-1].h
            succeeded = True
        else:
            # self.h += len(self.path) - 1
            # self.path = [self.curr_node]
            succeeded = False
        rename_nodes_in_path(self.path)
        self.path_names = [node.xy_name for node in self.path]
        return succeeded, {'a_s_time': time.time() - start_time, 'a_s_info': a_s_info}

    def update_nei(self, agents, **kwargs):
        k = kwargs['k']
        h = kwargs['h']
        # nei_r = k
        nei_r = h
        self.last_path_change_iter = 0
        self.conf_agents_names = []
        self.nei_list, self.nei_dict, self.nei_paths_dict, self.nei_h_dict = [], {}, {}, {}
        nei_dist_const = 2 * nei_r + 1
        for agent in agents:
            if agent.name != self.name:
                curr_distance = manhattan_distance_nodes(self.curr_node, agent.curr_node)
                if curr_distance <= nei_dist_const:
                    self.nei_list.append(agent)
                    self.nei_dict[agent.name] = agent
                    self.nei_h_dict[agent.name] = None
                    self.nei_paths_dict[agent.name] = None

        self.stats_nei_list.append(len(self.nei_list) - 1)

    def init_plan(self, **kwargs):
        k = kwargs['k']
        self.names_to_consider_list = list(self.nei_paths_dict.keys())
        if len(self.path) == 0 or self.path[-1].xy_name != self.goal_node.xy_name:
            succeeded, info = self.calc_a_star_plan(k_time=k, **kwargs)
            return succeeded, info
        return False, {}

    def exchange_paths(self):
        for nei in self.nei_list:
            nei.nei_paths_dict[self.name] = self.path
            nei.nei_h_dict[self.name] = self.h
            self.stats_n_messages += 1
            self.stats_n_step_m += 1

    def update_conf_agents_names(self, check_r, immediate=False):
        self.conf_agents_names = []
        # self.nei_paths_dict
        conf_list = check_single_agent_k_step_c_v(self.name, self.path, self.nei_paths_dict, check_r+1, immediate=immediate)
        c_e_list = check_single_agent_k_step_c_e(self.name, self.path, self.nei_paths_dict, check_r+1, immediate=immediate)
        conf_list.extend(c_e_list)
        conf_agents_names = []
        for conf in conf_list:
            conf_agents_names.append(conf[1])
        self.conf_agents_names = list(set(conf_agents_names))

    def pref_index(self, **kwargs):
        p_h, p_l = kwargs['p_h'], kwargs['p_l']
        paths_to_consider_dict = {}
        # just index
        for agent_name, path in self.nei_paths_dict.items():
            if self.nei_dict[agent_name].index < self.index:
                if random.random() < p_h:
                    paths_to_consider_dict[agent_name] = self.nei_paths_dict[agent_name]
            elif random.random() < p_l:
                paths_to_consider_dict[agent_name] = self.nei_paths_dict[agent_name]
        return paths_to_consider_dict

    def pref_path_length(self, **kwargs):
        p_h, p_l = kwargs['p_h'], kwargs['p_l']
        paths_to_consider_dict = {}
        # path length + index
        for agent_name, path in self.nei_paths_dict.items():
            if len(self.path) + self.h > len(path) + self.nei_h_dict[agent_name]:
                if random.random() < p_l:
                    paths_to_consider_dict[agent_name] = self.nei_paths_dict[agent_name]
            elif len(self.path) + self.h < len(path) + self.nei_h_dict[agent_name]:
                if random.random() < p_h:
                    paths_to_consider_dict[agent_name] = self.nei_paths_dict[agent_name]
            elif self.index > self.nei_dict[agent_name].index:
                if random.random() < p_l:
                    paths_to_consider_dict[agent_name] = self.nei_paths_dict[agent_name]
            else:
                if random.random() < p_h:
                    paths_to_consider_dict[agent_name] = self.nei_paths_dict[agent_name]
        return paths_to_consider_dict

    def pref_path_rand(self, paths_to_consider_dict, **kwargs):
        new_paths_to_consider_dict = {}
        for agent_name, path in paths_to_consider_dict.items():
            if agent_name not in self.conf_agents_names:
                new_paths_to_consider_dict[agent_name] = path
        return new_paths_to_consider_dict

    def get_paths_to_consider_dict(self, **kwargs):
        # p_h, p_l = 0.9, 0.1
        pref_paths_type = kwargs['pref_paths_type']
        if pref_paths_type == 'pref_index':
            paths_to_consider_dict = self.pref_index(**kwargs)
        elif pref_paths_type == 'pref_path_length':
            paths_to_consider_dict = self.pref_path_length(**kwargs)
        else:
            raise RuntimeError('no noooo')

        names_to_consider_list = list(paths_to_consider_dict.keys())
        self.names_to_consider_list = names_to_consider_list
        return paths_to_consider_dict, names_to_consider_list

    @staticmethod
    def k_transform(**kwargs):
        k = kwargs['k']
        # use also kwargs['k_step_iteration'] and kwargs['k_step_iteration_limit']
        # k = k + k * round(kwargs['small_iteration'] / 20)  kwargs['k_small_iter_limit']
        ratio = kwargs['k_step_iteration'] / kwargs['k_step_iteration_limit']
        # ratio = kwargs['small_iteration'] / kwargs['k_small_iter_limit']
        if ratio > 0.87:
            # print('!!!!!!!!!!!!')
            return k * 8
        if ratio > 0.75:
            # print('!!!!!!!!!!!!')
            return k * 4
        if ratio > 0.5:
            # print('!!!!!!!!!!!!')
            return k * 2
        return k

    def check_if_all_around_finished(self, succeeded, info, paths_to_consider_dict, check_r, **kwargs):
        all_finished = True
        for agent_name, path in paths_to_consider_dict.items():
            if check_r < len(path):
                all_finished = False
        if all_finished and self.path[-1].xy_name != self.goal_node.xy_name:
            self.init_plan(**kwargs)
            return False, info
        return succeeded, info

    def check_nei_last_change(self, paths_to_consider_dict, names_to_consider_list, **kwargs):
        curr_iteration = kwargs["small_iteration"]
        nei_changed_dict = {}
        changed = False
        for nei_name in names_to_consider_list:
            nei_agent = self.nei_dict[nei_name]
            nei_changed_dict[nei_name] = nei_agent.last_path_change_iter
            if nei_agent.last_path_change_iter >= curr_iteration - 1:
                changed = True
                break
        return changed

    def replan(self, **kwargs):
        check_r = self.k_transform(**kwargs)
        self.update_conf_agents_names(check_r)
        # self.update_conf_agents_names(h)
        if len(self.conf_agents_names) == 0:
            return True, {}
        # probabilities to use: p_ch, p_h, p_l
        p_ch = self.set_p_ch(**kwargs)
        if random.random() < p_ch:
            paths_to_consider_dict, names_to_consider_list = self.get_paths_to_consider_dict(**kwargs)
            # v change paths to consider v
            # paths_to_consider_dict, names_to_consider_list = self.check_nei_last_change(paths_to_consider_dict, names_to_consider_list, **kwargs)
            # changed = self.check_nei_last_change(paths_to_consider_dict, names_to_consider_list, **kwargs)
            # if not changed and random.random() < 0.5:
            #     return True, {}
            v_constr_dict, e_constr_dict, _ = build_constraints(self.nodes, paths_to_consider_dict)
            # full_paths_dict = {agent_name: self.nei_paths_dict[agent_name] for agent_name in names_to_consider_list}
            perm_constr_dict = build_k_step_perm_constr_dict(self.nodes, paths_to_consider_dict, check_r)
            # succeeded: if True - changed, if False - did not
            succeeded, info = self.calc_a_star_plan(v_constr_dict, e_constr_dict, perm_constr_dict, k_time=check_r, **kwargs)
            if succeeded:
                self.last_path_change_iter = kwargs["small_iteration"]
            succeeded, info = self.check_if_all_around_finished(succeeded, info, paths_to_consider_dict, check_r,
                                                                **kwargs)
            return succeeded, info
        return True, {}

    def set_p_ch(self, **kwargs):
        p_ch_type = kwargs['p_ch_type']

        if len(self.path) == 0:
            return 1

        if len(self.conf_agents_names) == 1:
            return 0.5

        if p_ch_type == 'simple':
            alpha = kwargs['alpha']
            return alpha

        elif p_ch_type == 'max_prev':
            #  gather all paths
            full_path_lngths = []
            for agent_name, path in self.nei_paths_dict.items():
                if agent_name in self.conf_agents_names:
                    full_path_lngths.append(int(len(path) + self.nei_h_dict[agent_name]))
            # break by index if equal
            my_full_path_len = int(len(self.path) + self.h)
            if len(full_path_lngths) == 1 and full_path_lngths[0] == my_full_path_len:
                return 0.5
                # return 0.9 if self.index < self.nei_dict[self.conf_agents_names[0]].index else 0.1
            full_path_lngths.append(my_full_path_len)
            full_path_lngths.sort()
            my_order = full_path_lngths.index(my_full_path_len)
            my_alpha = 0.9 - 0.8 * (my_order / (len(full_path_lngths) - 1))
            return my_alpha

        else:
            raise RuntimeError('no such p_ch_type')

    def update_full_path_no_k(self):
        if len(self.full_path) == 0:
            self.full_path.extend(self.path)
        else:
            self.full_path.extend(self.path[1:])
        self.full_path_names = [node.xy_name for node in self.full_path]
        self.curr_node = self.full_path[-1]
        self.path = self.path[-1:]
        self.path_names = [node.xy_name for node in self.path]
        return self.curr_node.xy_name == self.goal_node.xy_name

    def update_full_path(self, **kwargs):
        # stats
        self.stats_n_step_m_list.append(self.stats_n_step_m)
        self.stats_n_step_m = 0

        k = kwargs['k']
        h = kwargs['h']

        if max(k, h) > 1e6:
            return self.update_full_path_no_k()

        step = h + 1
        # step = k + 1
        # step = 1

        while len(self.path) < step:
            self.path.append(self.path[-1])

        if len(self.full_path) == 0:
            self.full_path.extend(self.path[:step])
        else:
            self.full_path.extend(self.path[1:step])
        self.full_path_names = [node.xy_name for node in self.full_path]
        self.curr_node = self.full_path[-1]
        self.curr_node.t = 0
        self.path = self.path[step - 1:]
        rename_nodes_in_path(self.path)
        self.path_names = [node.xy_name for node in self.path]
        return self.curr_node.xy_name == self.goal_node.xy_name

    def cut_back_full_path(self):
        len_full_path = len(self.full_path)
        if len_full_path > 1:
            if self.full_path[-1].xy_name == self.goal_node.xy_name:
                for backwards_node in self.full_path[:-1][::-1]:
                    if backwards_node.xy_name == self.goal_node.xy_name:
                        len_full_path -= 1
                    else:
                        break
        self.full_path = self.full_path[:len_full_path]
        self.full_path_names = [node.xy_name for node in self.full_path]


def create_agents(start_nodes, goal_nodes, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit, map_dim):
    # Creating agents
    agents = []
    agents_dict = {}
    n_agent = 0
    for start_node, goal_node in zip(start_nodes, goal_nodes):
        agent = KSDSAgent(n_agent, start_node, goal_node, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit,
                          map_dim)
        agents.append(agent)
        agents_dict[agent.name] = agent
        n_agent += 1

    return agents, agents_dict


def check_if_limit_is_crossed(func_info, alg_info, **kwargs):
    # runtime - the sequential time in seconds - number
    if 'runtime' in func_info:
        alg_info['runtime'] += func_info['runtime']
    # alg_info['dist_runtime'] - distributed time in seconds - number
    if 'dist_runtime' in func_info:
        alg_info['dist_runtime'] += func_info['dist_runtime']

    # alg_info['a_star_calls_counter'] - number
    if 'a_star_calls_counter' in func_info:
        alg_info['a_star_calls_counter'] += func_info['a_star_calls_counter']
    # alg_info['a_star_calls_counter_dist'] - number
    if 'a_star_calls_counter_dist' in func_info:
        alg_info['a_star_calls_counter_dist'] += func_info['a_star_calls_counter_dist']

    # alg_info['a_star_n_closed'] - list
    if 'a_star_n_closed' in func_info:
        alg_info['a_star_n_closed'].append(func_info['a_star_n_closed'])
    # alg_info['a_star_n_closed_dist'] - number
    if 'a_star_n_closed_dist' in func_info:
        alg_info['a_star_n_closed_dist'] += func_info['a_star_n_closed_dist']

    return limit_is_crossed(alg_info['runtime'], alg_info, **kwargs)


def all_plan_and_find_nei(agents: List[KSDSAgent], **kwargs):
    runtime, runtime_dist = 0, []
    a_star_calls_counter, a_star_calls_counter_dist = 0, []
    a_star_n_closed, a_star_n_closed_dist = 0, []
    succeeded_list = []
    for agent in agents:

        # find_nei
        agent.update_nei(agents, **kwargs)

        # create initial plan
        start_time = time.time()
        # info: {'a_s_time': time.time() - start_time, 'a_s_info': a_s_info}
        # succeeded, info = agent.calc_a_star_plan(**kwargs)
        succeeded, info = agent.init_plan(**kwargs)
        # stats
        end_time = time.time() - start_time
        runtime += end_time
        runtime_dist.append(end_time)
        succeeded_list.append(succeeded)
        if succeeded:
            a_star_calls_counter += 1
            a_star_n_closed += info['a_s_info']['n_closed']
            a_star_n_closed_dist.append(info['a_s_info']['n_closed'])

    func_info = {
        'runtime': runtime,
        'dist_runtime': max(runtime_dist),
        'a_star_calls_counter': a_star_calls_counter,
        'a_star_calls_counter_dist': 1,
        'a_star_n_closed': a_star_n_closed,
        'a_star_n_closed_dist': max(a_star_n_closed_dist) if a_star_n_closed > 0 else 0,
        'all_succeeded': all(succeeded_list),
    }
    return func_info


def all_exchange_k_step_paths(agents: List[KSDSAgent], **kwargs):
    k = kwargs['k']
    h = kwargs['h']
    # check_radius = k
    check_radius = h
    runtime, runtime_dist = 0, []

    # exchange paths
    for agent in agents:
        start_time = time.time()
        agent.exchange_paths()
        # stats
        end_time = time.time() - start_time
        runtime += end_time
        runtime_dist.append(end_time)

    # check for collisions
    plans = {agent.name: agent.path for agent in agents}
    there_are_collisions, c_v, c_e = just_check_k_step_plans(plans, check_radius+1, immediate=True)
    # there_are_collisions, c_v, c_e = just_check_k_step_plans(plans, check_radius+1, immediate=False)

    func_info = {
        'runtime': runtime,
        'dist_runtime': max(runtime_dist)
    }
    return there_are_collisions, c_v, c_e, func_info


def all_replan(agents: List[KSDSAgent], alg_info, **kwargs):
    # inner print
    i_run_str = f'[i_run: {kwargs["i_run"]}]' if 'i_run' in kwargs else ''
    img_str = f'[{kwargs["img_dir"]}]' if 'img_dir' in kwargs else ''
    print(f'\n\n[runtime={alg_info["runtime"]:0.2f} sec.][dist_runtime={alg_info["dist_runtime"]:0.2f} sec.]'
          f'\n{img_str}({kwargs["alg_name"]}){i_run_str}[finished: {kwargs["number_of_finished"]}/{kwargs["n_agents"]}]'
          f'[step: {kwargs["k_step_iteration"]}][iter: {kwargs["small_iteration"]}]\n')

    runtime, runtime_dist = 0, [0]
    a_star_calls_counter, a_star_calls_counter_dist = 0, []
    a_star_n_closed, a_star_n_closed_dist = 0, [0]
    succeeded_list = []
    failed_paths_dict = {}

    for agent in agents:
        start_time = time.time()
        succeeded, info = agent.replan(**kwargs)
        # stats
        end_time = time.time() - start_time
        runtime += end_time
        runtime_dist.append(end_time)
        succeeded_list.append(succeeded)
        if not succeeded:
            failed_paths_dict[agent.name] = agent.path_names
        if len(info) > 0:
            a_star_calls_counter += 1
            a_star_n_closed += info['a_s_info']['n_closed']
            a_star_n_closed_dist.append(info['a_s_info']['n_closed'])

    func_info = {
        'runtime': runtime,
        'dist_runtime': max(runtime_dist),
        'a_star_calls_counter': a_star_calls_counter,
        'a_star_calls_counter_dist': 1,
        'a_star_n_closed': a_star_n_closed,
        'a_star_n_closed_dist': max(a_star_n_closed_dist),
        'all_succeeded': all(succeeded_list),
        'failed_paths_dict': failed_paths_dict,
    }
    return func_info


def all_move_k_steps(agents: List[KSDSAgent], **kwargs):
    all_paths_are_finished_list = []
    for agent in agents:
        agent_is_finished = agent.update_full_path(**kwargs)
        all_paths_are_finished_list.append(agent_is_finished)

    number_of_finished = sum(all_paths_are_finished_list)
    all_paths_are_finished = all(all_paths_are_finished_list)
    func_info = {}
    return all_paths_are_finished, number_of_finished, func_info


def all_cut_full_paths(agents: List[KSDSAgent], **kwargs):
    for agent in agents:
        agent.cut_back_full_path()


def run_k_sds(start_nodes, goal_nodes, nodes, nodes_dict, h_func, **kwargs):
    if 'k' not in kwargs or 'h' not in kwargs:
        raise RuntimeError("'k' or 'h' not in kwargs")
    if 'k_step_iteration_limit' in kwargs:
        k_step_iteration_limit = kwargs['k_step_iteration_limit']
    else:
        k_step_iteration_limit = 200
        kwargs['k_step_iteration_limit'] = k_step_iteration_limit
        kwargs['k_small_iter_limit'] = 40
    alg_name = kwargs['alg_name'] if 'alg_name' in kwargs else f'k-SDS'
    iter_limit = kwargs['a_star_iter_limit'] if 'a_star_iter_limit' in kwargs else 1e100
    plotter = kwargs['plotter'] if 'plotter' in kwargs else None
    middle_plot = kwargs['middle_plot'] if 'middle_plot' in kwargs else False
    final_plot = kwargs['final_plot'] if 'final_plot' in kwargs else True
    map_dim = kwargs['map_dim'] if 'map_dim' in kwargs else None
    stats_small_iters_list = []
    number_of_finished = 0

    # Creating agents
    agents, agents_dict = create_agents(start_nodes, goal_nodes, nodes, nodes_dict, h_func, plotter, middle_plot,
                                        iter_limit, map_dim)
    kwargs['n_agents'] = len(agents)

    # alg info dict
    alg_info = get_alg_info_dict()

    # Distributed Part
    for k_step_iteration in range(1000000):
        kwargs['k_step_iteration'] = k_step_iteration
        kwargs['small_iteration'] = 0
        kwargs['number_of_finished'] = number_of_finished

        func_info = all_plan_and_find_nei(agents, **kwargs)  # agents
        if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
            return None, {'agents': agents, 'success_rate': 0}

        there_are_collisions = True
        while there_are_collisions:
            kwargs['small_iteration'] += 1

            there_are_collisions, c_v, c_e, func_info = all_exchange_k_step_paths(agents, **kwargs)  # agents
            if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
                return None, {'agents': agents, 'success_rate': 0}

            if not there_are_collisions:
                break

            func_info = all_replan(agents, alg_info, **kwargs)  # agents
            if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
                return None, {'agents': agents, 'success_rate': 0}

        stats_small_iters_list.append(kwargs['small_iteration'])
        all_paths_are_finished, number_of_finished, func_info = all_move_k_steps(agents, **kwargs)  # agents
        if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
            return None, {'agents': agents, 'success_rate': 0}

        full_plans = {agent.name: agent.full_path for agent in agents}
        iteration_print(agents, full_plans, alg_name, alg_info, alg_info['runtime'], k_step_iteration)
        if all_paths_are_finished:
            # there_is_col_0, c_v_0, c_e_0, cost_0 = just_check_plans(full_plans)
            all_cut_full_paths(agents, **kwargs)
            cut_full_plans = {agent.name: agent.full_path for agent in agents}
            there_is_col, c_v, c_e, cost = just_check_plans(cut_full_plans)
            if there_is_col:
                raise RuntimeError('uff')
            else:
                if final_plot:
                    print(f'#########################################################')
                    print(f'#########################################################')
                    print(f'#########################################################')
                    print(f"runtime: {alg_info['runtime']}\n{alg_info['dist_runtime']=}\n{cost=}")
                    print(f"a_star_n_closed: {sum(alg_info['a_star_n_closed'])}\n{alg_info['a_star_n_closed_dist']=}")
                    plotter.plot_mapf_paths(paths_dict=cut_full_plans, nodes=nodes, **kwargs)
                alg_info['success_rate'] = 1
                alg_info['sol_quality'] = cost
                alg_info['a_star_calls_per_agent'] = [agent.stats_n_calls for agent in agents]
                alg_info['n_messages'] = np.sum([agent.stats_n_messages for agent in agents])
                alg_info['m_per_step'] = np.sum([np.mean(agent.stats_n_step_m_list) for agent in agents])
                alg_info['n_steps'] = k_step_iteration + 1
                alg_info['n_small_iters'] = float(np.mean(stats_small_iters_list))
                alg_info['n_nei'] = np.sum([np.mean(agent.stats_nei_list) for agent in agents])
                # alg_info['avr_n_nei'] = np.mean([np.mean(agent.stats_nei_list) for agent in agents])
            return cut_full_plans, alg_info

        if k_step_iteration > k_step_iteration_limit-1:
            print(f'\n[LIMIT]: k_step_iteration: {k_step_iteration} > limit: {k_step_iteration_limit}')
            break

    return None, {'agents': agents, 'success_rate': 0}


def main():
    n_agents = 200
    # img_dir = 'my_map_10_10_room.map'  # 10-10
    # img_dir = 'empty-48-48.map'  # 48-48
    img_dir = 'random-64-64-10.map'  # 64-64
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
    h = 10
    p_ch_type = 'max_prev'
    # p_ch_type = 'simple'
    alpha = 1.0
    # pref_paths_type = 'pref_index'
    pref_paths_type = 'pref_path_length'
    # p_h = 1
    # p_l = 1
    p_h = 0.9
    p_l = 0.9
    alg_name = f'{k}-{h}-SDS-{p_l}-{p_h}'

    # for the PP algorithm
    # alg_name = 'k-PrP'
    # k = 10
    # h = 10
    # p_h_type = 'simple'
    # alpha = 1.0
    # pref_paths_type = 'pref_index'
    # p_h = 1.0
    # p_l = 0.1
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
            algorithm=run_k_sds,
            img_dir=img_dir,
            alg_name=alg_name,
            k=k,
            h=h,
            p_ch_type=p_ch_type,
            alpha=alpha,
            p_h=p_h,
            p_l=p_l,
            pref_paths_type=pref_paths_type,
            n_agents=n_agents,
            random_seed=random_seed,
            seed=seed,
            final_plot=True,
            a_star_iter_limit=5e7,
            # limit_type='norm_time',
            limit_type='dist_time',
            max_time=50,
            a_star_closed_nodes_limit=1e6,
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
        stats.dump_stats('../stats/results_k_sds.pstat')


if __name__ == '__main__':
    main()
