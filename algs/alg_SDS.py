import logging
import random
import time
import matplotlib.pyplot as plt
import cProfile
import pstats

import numpy as np

from algs.alg_a_star_space_time import a_star
from algs.test_mapf_alg import test_mapf_alg_from_pic
from algs.metrics import c_v_check_for_agent, c_e_check_for_agent, build_constraints, \
    limit_is_crossed, get_agents_in_conf, check_plan, get_alg_info_dict, iteration_print


class SDSAgent:
    def __init__(self, index, start_node, goal_node, nodes, nodes_dict, h_func,
                 plotter, middle_plot,
                 iter_limit=1e100, map_dim=None, limit_type='simple'):
        self.index = index
        self.name = f'agent_{index}'
        self.start_node = start_node
        self.goal_node = goal_node
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.h_func = h_func
        self.plotter = plotter
        self.middle_plot = middle_plot
        self.iter_limit = iter_limit
        self.path = []
        self.other_paths = {}
        self.conf_paths = {}
        self.map_dim = map_dim
        self.limit_type = limit_type
        self.stats_n_closed = 0
        self.stats_n_calls = 0
        self.stats_runtime = 0
        self.stats_n_messages = 0
        self.stats_confs_per_iter = []

    def exchange(self, agents):
        self.other_paths = {agent.name: agent.path for agent in agents if agent.name != self.name}
        self.stats_n_messages += len(agents) - 1

    def decision_bool(self, decision_type, agents_in_confs, agents_dict, **kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.5

        if len(self.path) == 0:
            return True

        if len(agents_in_confs) == 0:
            raise RuntimeError('len(agents_in_confs) == 0')

        if len(agents_in_confs) == 1:
            return random.random() < 0.5

        if decision_type == 'simple':
            return random.random() < alpha

        # elif decision_type == 'min_prev_1':
        #     path_lngths = [len(self.other_paths[agent_name]) for agent_name in agents_in_confs]
        #     max_n = max(path_lngths)
        #     min_n = min(path_lngths)
        #     # priority on smaller paths
        #     if len(self.path) > max_n and random.random() < 0.9:
        #         return True
        #     elif len(self.path) < min_n and random.random() < 0.1:
        #         return True
        #     elif random.random() < alpha:
        #         return True
        #     return

        elif decision_type == 'min_prev_2':
            # A MORE SMART VERSION
            path_lngths = [len(self.other_paths[agent_name]) for agent_name in agents_in_confs]
            max_n = max(path_lngths)
            min_n = min(path_lngths)
            # priority on smaller paths
            if len(self.path) > max_n and random.random() < 0.9:
                return True
            elif len(self.path) < min_n and random.random() < 0.1:
                return True
            else:
                path_lngths.append(len(self.path))
                path_lngths.sort()
                my_order = path_lngths.index(len(self.path))
                my_alpha = 0.1 + 0.8 * (my_order / len(path_lngths))
                if random.random() < my_alpha:
                    return True

        # elif decision_type == 'max_prev_1':
        #     path_lngths = [len(self.other_paths[agent_name]) for agent_name in agents_in_confs]
        #     max_n = max(path_lngths)
        #     min_n = min(path_lngths)
        #     # priority on bigger paths
        #     if len(self.path) > max_n and random.random() < 0.1:
        #         return True
        #     elif len(self.path) < min_n and random.random() < 0.9:
        #         return True
        #     elif random.random() < alpha:
        #         return True

        # elif decision_type == 'index_1':
        #     agents_indecies = [agents_dict[agent_name].index for agent_name in agents_in_confs]
        #     max_i = max(agents_indecies)
        #     min_i = min(agents_indecies)
        #     # priority on bigger paths
        #     if self.index > max_i and random.random() < 0.1:
        #         return True
        #     elif self.index < min_i and random.random() < 0.9:
        #         return True
        #     elif random.random() < alpha:
        #         return True

        # elif decision_type == 'index_2':
        #     agents_indecies = [agents_dict[agent_name].index for agent_name in agents_in_confs]
        #     max_i = max(agents_indecies)
        #     min_i = min(agents_indecies)
        #     # priority on bigger paths
        #     if self.index > max_i and random.random() < 0.9:
        #         return True
        #     elif self.index < min_i and random.random() < 0.1:
        #         return True
        #     else:
        #         agents_indecies.append(self.index)
        #         agents_indecies.sort()
        #         my_order = agents_indecies.index(self.index)
        #         my_alpha = 0.1 + 0.8 * (my_order / len(agents_indecies))
        #         if random.random() < my_alpha:
        #             return True

        else:
            raise RuntimeError('no such decision_type')

        return False

    def get_a_star_iter_limit(self, agents_in_confs):
        iter_limit = self.iter_limit
        if self.limit_type == 'simple':
            return iter_limit
        if self.limit_type == 'smart':
            if len(agents_in_confs) > 0 and self.map_dim:
                agent_in_conf_paths_len = [len(self.other_paths[agent]) for agent in agents_in_confs]
                max_path_len = max(agent_in_conf_paths_len)
                alt_iter_limit = max_path_len * self.map_dim[0] * self.map_dim[1]
                iter_limit = min(iter_limit, alt_iter_limit)
        return iter_limit

    def plan(self, decision_type, agents_dict=None, **kwargs):
        start_time = time.time()

        c_v_list = c_v_check_for_agent(self.name, self.path, self.other_paths)
        c_e_list = c_e_check_for_agent(self.name, self.path, self.other_paths)
        self.stats_confs_per_iter.append(len(c_v_list) + len(c_e_list))

        if len(self.path) > 0 and len(c_v_list) == 0 and len(c_e_list) == 0:
            # print(f'\n ---------- (DS {decision_type}) NO NEED FOR A* {self.name} ---------- \n')
            return False, {'elapsed': None, 'a_s_info': None}, True

        agents_in_confs = get_agents_in_conf(c_v_list, c_e_list)
        to_change = self.decision_bool(decision_type, agents_in_confs, agents_dict, **kwargs)
        if to_change:
            v_constr_dict, e_constr_dict, perm_constr_dict = build_constraints(self.nodes, self.other_paths)
            iter_limit = self.get_a_star_iter_limit(agents_in_confs)
            # print(f'\n ---------- ({kwargs["alg_name"]})[iteration: {kwargs["iteration"]}] A* {self.name} ---------- \n')
            a_star_func = kwargs['a_star_func']
            new_path, a_s_info = a_star_func(start=self.start_node, goal=self.goal_node,
                                             nodes=self.nodes, nodes_dict=self.nodes_dict, h_func=self.h_func,
                                             v_constr_dict=v_constr_dict,
                                             e_constr_dict=e_constr_dict,
                                             perm_constr_dict=perm_constr_dict,
                                             plotter=self.plotter, middle_plot=self.middle_plot,
                                             iter_limit=iter_limit)
            # stats
            self.stats_n_calls += 1
            self.stats_n_closed += a_s_info['n_closed']
            self.stats_runtime += time.time() - start_time
            if new_path is not None:
                self.path = new_path
            return True, {'elapsed': time.time() - start_time, 'a_s_info': a_s_info,
                          'n_agents_conf': len(agents_in_confs)}, False

        # print(f'\n ---------- (DS {decision_type}) NO NEED FOR A* {self.name} ---------- \n')
        return False, {'elapsed': None, 'a_s_info': None}, False


def run_sds(start_nodes, goal_nodes, nodes, nodes_dict, h_func, **kwargs):
    runtime = 0
    iter_limit = kwargs['a_star_iter_limit'] if 'a_star_iter_limit' in kwargs else 1e100
    plotter = kwargs['plotter'] if 'plotter' in kwargs else None
    middle_plot = kwargs['middle_plot'] if 'middle_plot' in kwargs else False
    final_plot = kwargs['final_plot'] if 'final_plot' in kwargs else True
    plot_per = kwargs['plot_per'] if 'plot_per' in kwargs else 10
    map_dim = kwargs['map_dim'] if 'map_dim' in kwargs else None
    limit_type = kwargs['limit_type'] if 'limit_type' in kwargs else 'simple'
    alg_name = kwargs['alg_name'] if 'alg_name' in kwargs else f'DS'

    # Creating agents
    agents = []
    agents_dict = {}
    n_agent = 0
    for start_node, goal_node in zip(start_nodes, goal_nodes):
        agent = SDSAgent(n_agent, start_node, goal_node, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit,
                         map_dim, limit_type)
        agents.append(agent)
        agents_dict[agent.name] = agent
        n_agent += 1

    alg_info = get_alg_info_dict()

    # Distributed Part
    for iteration in range(1000000):
        print(f'\n\n[runtime={alg_info["runtime"]:0.2f} sec.][dist_runtime={alg_info["dist_runtime"]:0.2f} sec.]'
              f'\n({kwargs["alg_name"]})[n agents: {len(agents)}][iter: {iteration}]\n')

        # start_time = time.time()
        kwargs['iteration'] = iteration
        max_time_list = []
        max_n_closed_list = []
        no_confs_list = []

        # LIMITS
        if limit_is_crossed(runtime, alg_info, **kwargs):
            break

        # PLAN
        for agent in agents:
            start_time = time.time()

            succeeded, info, no_confs = agent.plan(agents_dict=agents_dict, **kwargs)
            no_confs_list.append(no_confs)
            if info['elapsed']:
                max_time_list.append(info['elapsed'])
                max_n_closed_list.append(info['a_s_info']['n_closed'])
                alg_info['a_star_runtimes'].append(info['a_s_info']['runtime'])
                alg_info['a_star_n_closed'].append(info['a_s_info']['n_closed'])
                if iteration > 0:
                    alg_info['n_agents_conf'].append(info['n_agents_conf'])
                alg_info['a_star_calls_counter'] += 1

            # STATS + LIMITS
            runtime += time.time() - start_time
            if limit_is_crossed(runtime, alg_info, **kwargs):
                break

        if len(max_time_list) > 0 and max(max_time_list) > 0:
            alg_info['dist_runtime'] += max(max_time_list)
            alg_info['a_star_n_closed_dist'] = max([curr_a.stats_n_closed for curr_a in agents])
            alg_info['a_star_calls_counter_dist'] += 1

        # EXCHANGE
        for agent in agents:
            start_time = time.time()

            agent.exchange(agents=agents)

            # STATS + LIMITS
            runtime += time.time() - start_time
            if limit_is_crossed(runtime, alg_info, **kwargs):
                break

        # CHECK PLAN
        plan = {agent.name: agent.path for agent in agents}
        # iteration_print(agents, plan, alg_name, alg_info, runtime, iteration)
        if all(no_confs_list):
            there_is_col, c_v, c_e, cost = check_plan(agents, plan, alg_name, alg_info, runtime, iteration)
            if not there_is_col:
                if final_plot:
                    print(f'#########################################################')
                    print(f'#########################################################')
                    print(f'#########################################################')
                    plotter.plot_mapf_paths(paths_dict=plan, nodes=nodes, **kwargs)
                alg_info['success_rate'] = 1
                alg_info['sol_quality'] = cost
                alg_info['runtime'] = runtime
                alg_info['a_star_calls_per_agent'] = [agent.stats_n_calls for agent in agents]
                alg_info['n_messages_per_agent'] = [agent.stats_n_messages for agent in agents]
                alg_info['n_messages'] = np.sum([agent.stats_n_messages for agent in agents])
                alg_info['m_per_step'] = np.sum([agent.stats_n_messages for agent in agents])
                alg_info['n_steps'] = 1
                alg_info['n_small_iters'] = iteration
                alg_info['n_nei'] = (len(agents) - 1) ** 2
                return plan, alg_info

    # partial order
    pass

    return None, {'agents': agents, 'success_rate': 0}


def main():
    n_agents = 400
    # img_dir = 'my_map_10_10_room.map'  # 10-10
    # img_dir = 'empty-48-48.map'  # 48-48
    # img_dir = 'random-64-64-10.map'  # 64-64
    # img_dir = 'warehouse-10-20-10-2-1.map'  # 63-161
    # img_dir = 'lt_gallowstemplar_n.map'  # 180-251
    img_dir = 'random-32-32-10.map'  # 32-32               | LNS |

    random_seed = True
    # random_seed = False
    seed = 277
    PLOT_PER = 1
    PLOT_RATE = 0.5
    to_use_profiler = True
    # to_use_profiler = False
    # DECISION_TYPE = 'simple'
    # DECISION_TYPE = 'min_prev_1'
    DECISION_TYPE = 'min_prev_2'
    # DECISION_TYPE = 'max_prev_1'
    # DECISION_TYPE = 'index_1'
    # DECISION_TYPE = 'index_2'

    profiler = cProfile.Profile()
    if to_use_profiler:
        profiler.enable()
    for i in range(3):
        print(f'\n[run {i}]')
        result, info = test_mapf_alg_from_pic(
            algorithm=run_sds,
            img_dir=img_dir,
            initial_ordering=[],
            n_agents=n_agents,
            random_seed=random_seed,
            seed=seed,
            final_plot=True,
            a_star_iter_limit=5e7,
            max_time=5,
            limit_type='norm_time',
            alg_name='SDS',
            a_star_func=a_star,
            decision_type=DECISION_TYPE,
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
        stats.dump_stats('../stats/results_ds_mapf.pstat')


if __name__ == '__main__':
    main()