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
from algs.metrics import check_single_agent_k_step_c_v, check_single_agent_k_step_c_e


class KPrPAgent(KSDSAgent):
    def __init__(self, index, start_node, goal_node, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit=1e100,
                 map_dim=None):
        super().__init__(index, start_node, goal_node, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit,
                         map_dim)
        # nei
        pass

    def change_priority(self, priority, **kwargs):
        self.index = priority
        self.name = f'agent_{self.index}'
        reset_type = kwargs['reset_type']
        if reset_type == 'reset_start':
            self.reset_start()
        elif reset_type == 'reset_step':
            pass
        else:
            raise RuntimeError('mmmmm nope')

    def reset_start(self):
        self.curr_node = self.start_node
        self.path = []
        self.path_names = []
        self.h = 0
        self.full_path = []
        self.full_path_names = []

    def replan(self, **kwargs):
        check_r = self.k_transform(**kwargs)
        self.update_conf_agents_names(check_r, immediate=False)
        nei_index_list = [self.nei_dict[nei_name].index for nei_name in self.conf_agents_names]
        # self.update_conf_agents_names(h)
        if len(self.conf_agents_names) == 0 or min(nei_index_list) > self.index:
            return True, {}
        paths_to_consider_dict, names_to_consider_list = self.get_paths_to_consider_dict(**kwargs)
        v_constr_dict, e_constr_dict, _ = build_constraints(self.nodes, paths_to_consider_dict)
        perm_constr_dict = build_k_step_perm_constr_dict(self.nodes, paths_to_consider_dict, check_r)
        succeeded, info = self.calc_a_star_plan(v_constr_dict, e_constr_dict, perm_constr_dict, k_time=check_r, **kwargs)
        succeeded, info = self.check_if_all_around_finished(succeeded, info, paths_to_consider_dict, check_r, **kwargs)
        return succeeded, info


def create_agents(start_nodes, goal_nodes, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit, map_dim):
    # Creating agents
    agents = []
    agents_dict = {}
    n_agent = 0
    for start_node, goal_node in zip(start_nodes, goal_nodes):
        agent = KPrPAgent(n_agent, start_node, goal_node, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit,
                          map_dim)
        agents.append(agent)
        agents_dict[agent.name] = agent
        n_agent += 1

    return agents, agents_dict


def all_change_priority(agents: List[KPrPAgent], **kwargs):
    runtime, runtime_dist = 0, []

    priorities_list = list(range(len(agents)))
    random.shuffle(priorities_list)
    # exchange paths
    for new_priority, agent in zip(priorities_list, agents):
        start_time = time.time()
        agent.change_priority(new_priority, **kwargs)
        # stats
        end_time = time.time() - start_time
        runtime += end_time
        runtime_dist.append(end_time)
    func_info = {
        'runtime': runtime,
        'dist_runtime': max(runtime_dist)
    }
    return func_info


def run_k_distr_pp(start_nodes, goal_nodes, nodes, nodes_dict, h_func, **kwargs):
    if 'k' not in kwargs or 'h' not in kwargs:
        raise RuntimeError("'k' or 'h' not in kwargs")
    if 'k_step_iteration_limit' in kwargs:
        k_step_iteration_limit = kwargs['k_step_iteration_limit']
    else:
        k_step_iteration_limit = 200
        kwargs['k_step_iteration_limit'] = k_step_iteration_limit
    alg_name = kwargs['alg_name'] if 'alg_name' in kwargs else f'k-MGDS'
    reset_type = kwargs['reset_type']
    iter_limit = kwargs['a_star_iter_limit'] if 'a_star_iter_limit' in kwargs else 1e100
    plotter = kwargs['plotter'] if 'plotter' in kwargs else None
    middle_plot = kwargs['middle_plot'] if 'middle_plot' in kwargs else False
    final_plot = kwargs['final_plot'] if 'final_plot' in kwargs else True
    map_dim = kwargs['map_dim'] if 'map_dim' in kwargs else None
    stats_small_iters_list = []
    number_of_finished = 0
    need_to_reset_start = False

    # Creating agents
    agents, agents_dict = create_agents(start_nodes, goal_nodes, nodes, nodes_dict, h_func, plotter, middle_plot,
                                        iter_limit, map_dim)
    kwargs['n_agents'] = len(agents)

    # alg info dict
    alg_info = get_alg_info_dict()

    # STEPS
    for k_step_iteration in range(1000000):
        kwargs['k_step_iteration'] = k_step_iteration
        kwargs['small_iteration'] = 0
        kwargs['number_of_finished'] = number_of_finished

        func_info = all_plan_and_find_nei(agents, **kwargs)  # agents
        if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
            return None, {'agents': agents, 'success_rate': 0}

        # SMALL ITERATIONS
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

            if not func_info['all_succeeded']:
                print(f"\n###########################\nPRIORITY CHANGE {reset_type}\n###########################\n")
                func_info = all_change_priority(agents, **kwargs)  # agents
                if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
                    return None, {'agents': agents, 'success_rate': 0}
                if reset_type == 'reset_step':
                    func_info = all_plan_and_find_nei(agents, **kwargs)  # agents
                    if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
                        return None, {'agents': agents, 'success_rate': 0}
                    continue
                if reset_type == 'reset_start':
                    need_to_reset_start = True
                    break

        if need_to_reset_start:
            need_to_reset_start = False
            number_of_finished = 0
            continue

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

        if k_step_iteration > k_step_iteration_limit - 1:
            print(f'\n[LIMIT]: k_step_iteration: {k_step_iteration} > limit: {k_step_iteration_limit}')
            break

    return None, {'agents': agents, 'success_rate': 0}


def main():
    n_agents = 230
    # img_dir = 'my_map_10_10_room.map'  # 10-10
    # img_dir = 'empty-48-48.map'  # 48-48
    # img_dir = 'random-64-64-10.map'  # 64-64
    # img_dir = 'warehouse-10-20-10-2-1.map'  # 63-161
    img_dir = 'lt_gallowstemplar_n.map'  # 180-251

    # --------------------------------------------------- #
    # --------------------------------------------------- #
    # for the PP algorithm
    k = 30
    h = 30
    pref_paths_type = 'pref_index'
    p_h = 1
    p_l = 0
    # reset_type = 'reset_start'
    reset_type = 'reset_step'
    alg_name = f'{k}-{h}-PrP'
    # --------------------------------------------------- #
    # --------------------------------------------------- #

    # random_seed = True
    random_seed = False
    seed = 839
    PLOT_PER = 1
    PLOT_RATE = 0.5
    final_plot = True
    # final_plot = False

    to_use_profiler = True
    # to_use_profiler = False
    profiler = cProfile.Profile()
    if to_use_profiler:
        profiler.enable()
    for i in range(3):
        print(f'\n[run {i}]')
        result, info = test_mapf_alg_from_pic(
            algorithm=run_k_distr_pp,
            img_dir=img_dir,
            alg_name=alg_name,
            k=k,
            h=h,
            reset_type=reset_type,
            p_h=p_h,
            p_l=p_l,
            pref_paths_type=pref_paths_type,
            n_agents=n_agents,
            random_seed=random_seed,
            seed=seed,
            a_star_iter_limit=5e7,
            # limit_type='norm_time',
            limit_type='dist_time',
            max_time=50,
            a_star_closed_nodes_limit=1e6,
            final_plot=final_plot,
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
        stats.dump_stats('../stats/results_k_pp.pstat')


if __name__ == '__main__':
    main()


    # def update_conf_agents_names(self, check_r):
    #     self.conf_agents_names = []
    #     # self.nei_paths_dict
    #     h_nei_paths_dict = {}
    #     for nei_name, nei_path in self.nei_paths_dict.items():
    #         if self.nei_dict[nei_name].index < self.index:
    #             h_nei_paths_dict[nei_name] = nei_path
    #     conf_list = check_single_agent_k_step_c_v(self.name, self.path, h_nei_paths_dict, check_r+1, immediate=True)
    #     c_e_list = check_single_agent_k_step_c_e(self.name, self.path, h_nei_paths_dict, check_r+1, immediate=True)
    #     conf_list.extend(c_e_list)
    #     conf_agents_names = []
    #     for conf in conf_list:
    #         conf_agents_names.append(conf[1])
    #     self.conf_agents_names = list(set(conf_agents_names))