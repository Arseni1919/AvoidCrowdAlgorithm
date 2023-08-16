import random
from typing import List

import cProfile
import pstats

from functions import *

from algs.alg_a_star_space_time import a_star
from algs.alg_k_SDS import KSDSAgent, check_if_limit_is_crossed
from algs.alg_k_SDS import all_plan_and_find_nei, all_exchange_k_step_paths, all_replan, all_move_k_steps
from algs.alg_k_SDS import all_cut_full_paths
from algs.test_mapf_alg import test_mapf_alg_from_pic
from algs.metrics import build_constraints, limit_is_crossed, just_check_plans, build_k_step_perm_constr_dict
from algs.metrics import get_alg_info_dict, iteration_print, just_check_k_step_plans


class KMGDSAgent(KSDSAgent):
    def __init__(self, index, start_node, goal_node, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit=1e100,
                 map_dim=None):
        super().__init__(index, start_node, goal_node, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit,
                         map_dim)
        # nei
        self.gain = 0
        self.nei_gains_dict = {}

    def update_nei(self, agents, **kwargs):
        k = kwargs['k']
        h = kwargs['h']
        # nei_r = k
        nei_r = h
        self.gain = 0
        self.nei_gains_dict = {}
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
                    self.nei_gains_dict[agent.name] = 0

        self.stats_nei_list.append(len(self.nei_list))

    def exchange_gains(self, **kwargs):
        # update gain
        self.gain = 0
        check_r = self.k_transform(**kwargs)
        self.update_conf_agents_names(check_r)
        self.gain = len(self.conf_agents_names)

        # send the gain to nei
        for nei in self.nei_list:
            if nei.name in self.conf_agents_names:
                nei.nei_gains_dict[self.name] = self.gain
                self.stats_n_messages += 1
                self.stats_n_step_m += 1
        return self.gain

    def pref_max_gain(self, **kwargs):
        p_gain_h = kwargs['p_gain_h']
        p_gain_l = kwargs['p_gain_l']

        # the length of nei_gain_list must be more than 0
        max_nei_gain_list = []
        for nei_name, gain in self.nei_gains_dict.items():
            if nei_name in self.conf_agents_names:  # the agent looks only on those it is in conflict with
                max_nei_gain_list.append(gain)
        max_nei_gain = max(max_nei_gain_list)

        if max_nei_gain < self.gain:
            return p_gain_h
        elif max_nei_gain == self.gain:
            max_nei_gain_indecies = []
            for nei_name, gain in self.nei_gains_dict.items():
                if gain == max_nei_gain and nei_name in self.conf_agents_names:
                    max_nei_gain_indecies.append(self.nei_dict[nei_name].index)
            min_nei_gain_indx = min(max_nei_gain_indecies)
            return p_gain_h if self.index < min_nei_gain_indx else p_gain_l
        else:
            return p_gain_l

    def pref_min_gain(self, **kwargs):
        p_gain_h = kwargs['p_gain_h']
        p_gain_l = kwargs['p_gain_l']

        # the length of nei_gain_list must be more than 0
        max_nei_gain_list = []
        for nei_name, gain in self.nei_gains_dict.items():
            if nei_name in self.conf_agents_names:  # the agent looks only on those it is in conflict with
                max_nei_gain_list.append(gain)
        min_nei_gain = max(max_nei_gain_list)

        if self.gain < min_nei_gain:
            return p_gain_h
        elif min_nei_gain == self.gain:
            max_nei_gain_indecies = []
            for nei_name, gain in self.nei_gains_dict.items():
                if gain == min_nei_gain and nei_name in self.conf_agents_names:
                    max_nei_gain_indecies.append(self.nei_dict[nei_name].index)
            min_nei_gain_indx = min(max_nei_gain_indecies)
            if self.index < min_nei_gain_indx:
                return p_gain_h
        return p_gain_l

    def set_p_gain(self, **kwargs):

        if len(self.path) == 0:
            return 1

        p_gain_type = 'pref_max_gain'
        # p_gain_type = 'pref_min_gain'

        if p_gain_type == 'pref_max_gain':
            return self.pref_max_gain(**kwargs)

        elif p_gain_type == 'pref_min_gain':
            return self.pref_min_gain(**kwargs)

        else:
            raise RuntimeError('baam')

    def replan(self, **kwargs):
        if self.gain == 0:
            return False, {}
        # probabilities to use: p_ch, p_h, p_l
        p_gain = self.set_p_gain(**kwargs)
        if random.random() < p_gain:
            paths_to_consider_dict, names_to_consider_list = self.get_paths_to_consider_dict(**kwargs)
            v_constr_dict, e_constr_dict, _ = build_constraints(self.nodes, paths_to_consider_dict)
            full_paths_dict = {agent_name: self.nei_paths_dict[agent_name] for agent_name in names_to_consider_list}
            check_r = self.k_transform(**kwargs)
            perm_constr_dict = build_k_step_perm_constr_dict(self.nodes, full_paths_dict, check_r)
            succeeded, info = self.calc_a_star_plan(v_constr_dict, e_constr_dict, perm_constr_dict, **kwargs)
            return succeeded, info
        return True, {}


def create_agents(start_nodes, goal_nodes, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit, map_dim):
    # Creating agents
    agents = []
    agents_dict = {}
    n_agent = 0
    for start_node, goal_node in zip(start_nodes, goal_nodes):
        agent = KMGDSAgent(n_agent, start_node, goal_node, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit,
                           map_dim)
        agents.append(agent)
        agents_dict[agent.name] = agent
        n_agent += 1

    return agents, agents_dict


def all_exchange_gains(agents: List[KMGDSAgent], **kwargs):
    runtime, runtime_dist = 0, []
    gain_list = []

    # exchange paths
    for agent in agents:
        start_time = time.time()
        gain = agent.exchange_gains(**kwargs)
        gain_list.append(gain)
        # stats
        end_time = time.time() - start_time
        runtime += end_time
        runtime_dist.append(end_time)
    if sum(gain_list) == 0:
        raise RuntimeError('no no')
    func_info = {
        'runtime': runtime,
        'dist_runtime': max(runtime_dist)
    }
    return func_info


def run_k_mgds(start_nodes, goal_nodes, nodes, nodes_dict, h_func, **kwargs):
    if 'k' not in kwargs or 'h' not in kwargs:
        raise RuntimeError("'k' or 'h' not in kwargs")
    if 'k_step_iteration_limit' in kwargs:
        k_step_iteration_limit = kwargs['k_step_iteration_limit']
    else:
        k_step_iteration_limit = 200
        kwargs['k_step_iteration_limit'] = k_step_iteration_limit
    alg_name = kwargs['alg_name'] if 'alg_name' in kwargs else f'k-MGDS'
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

            func_info = all_exchange_gains(agents, **kwargs)  # agents
            if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
                return None, {'agents': agents, 'success_rate': 0}

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
                    print(f"runtime: {alg_info['runtime']}\n{alg_info['dist_runtime']=}")
                    print(f"a_star_n_closed: {sum(alg_info['a_star_n_closed'])}\n{alg_info['a_star_n_closed_dist']=}")
                    plotter.plot_mapf_paths(paths_dict=cut_full_plans, nodes=nodes, **kwargs)
                alg_info['success_rate'] = 1
                alg_info['sol_quality'] = cost
                alg_info['a_star_calls_per_agent'] = [agent.stats_n_calls for agent in agents]
                alg_info['n_messages'] = np.sum([agent.stats_n_messages for agent in agents])
                alg_info['m_per_step'] = np.sum([np.mean(agent.stats_n_step_m_list) for agent in agents])
                alg_info['n_steps'] = k_step_iteration + 1
                alg_info['n_small_iters'] = np.mean(stats_small_iters_list)
                alg_info['n_nei'] = np.sum([np.mean(agent.stats_nei_list) for agent in agents])
                alg_info['avr_n_nei'] = np.mean([np.mean(agent.stats_nei_list) for agent in agents])
            return cut_full_plans, alg_info

        if k_step_iteration > k_step_iteration_limit - 1:
            print(f'\n[LIMIT]: k_step_iteration: {k_step_iteration} > limit: {k_step_iteration_limit}')
            break

    return None, {'agents': agents, 'success_rate': 0}


def main():
    n_agents = 400
    # img_dir = 'my_map_10_10_room.map'  # 10-10
    # img_dir = 'empty-48-48.map'  # 48-48
    # img_dir = 'random-64-64-10.map'  # 64-64
    img_dir = 'warehouse-10-20-10-2-1.map'  # 63-161
    # img_dir = 'lt_gallowstemplar_n.map'  # 180-251

    # random_seed = True
    random_seed = False
    seed = 839
    PLOT_PER = 1
    PLOT_RATE = 0.5

    # --------------------------------------------------- #
    # --------------------------------------------------- #
    # for the algorithms
    k = 30
    h = 5
    p_gain_h = 0.9
    p_gain_l = 0.1
    # pref_paths_type = 'pref_index'
    pref_paths_type = 'pref_path_length'
    # p_h = 1
    # p_l = 1
    p_h = 0.9
    p_l = 0.9
    alg_name = f'{k}-{h}-MGDS-{p_gain_l}-{p_gain_h}-{p_l}-{p_h}'
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
            algorithm=run_k_mgds,
            img_dir=img_dir,
            alg_name=alg_name,
            k=k,
            h=h,
            p_gain_h=p_gain_h,
            p_gain_l=p_gain_l,
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
        stats.dump_stats('../stats/results_k_mgds.pstat')


if __name__ == '__main__':
    main()
