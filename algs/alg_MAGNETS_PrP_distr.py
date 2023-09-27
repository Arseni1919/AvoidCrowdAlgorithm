import random
from typing import List

import cProfile
import pstats

from functions import *
from funcs_plotter.plot_functions import plot_magnet_field
from algs.alg_a_star_space_time import a_star_xyt
from algs.alg_k_SDS import KSDSAgent, check_if_limit_is_crossed
from algs.alg_k_SDS import all_move_k_steps
from algs.alg_k_SDS import all_cut_full_paths
from algs.test_mapf_alg import test_mapf_alg_from_pic
from algs.metrics import build_constraints, just_check_plans, build_k_step_perm_constr_dict
from algs.metrics import get_alg_info_dict, iteration_print, just_check_k_step_plans
from algs.metrics import check_single_agent_k_step_c_v, check_single_agent_k_step_c_e


def build_nei_magnets(agents_to_consider_list, **kwargs):
    if len(agents_to_consider_list) == 0:
        return None
    map_dim = kwargs['map_dim']
    k = kwargs['k']
    nei_magnets = np.zeros((map_dim[0], map_dim[1], k))  # x, y, t
    for nei_agent in agents_to_consider_list:
        nei_magnets += nei_agent.magnet_field
    max_number_in_matrix = np.max(nei_magnets)
    if max_number_in_matrix > 0:
        nei_magnets /= max_number_in_matrix
    return nei_magnets


def build_mag_cost_func(agents_to_consider_list, nei_magnets,  **kwargs):
    k = kwargs['k']
    if len(agents_to_consider_list) == 0:
        return lambda x, y, t: 0
    return lambda x, y, t: nei_magnets[x, y, t] if t < k else 0


class KMagnetPrPAgent(KSDSAgent):
    def __init__(self, index, start_node, goal_node, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit=1e100,
                 map_dim=None):
        super().__init__(index, start_node, goal_node, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit,
                         map_dim)
        # spec for the alg
        self.map_dim = map_dim
        self.finished_k_iter = False
        self.nei_finished_dict = {}
        self.agents_to_consider_dict = {}
        self.magnet_field = None

    def reset_k_step(self, **kwargs):
        k = kwargs['k']
        self.finished_k_iter = False
        self.nei_finished_dict = {}
        self.magnet_field = np.zeros((self.map_dim[0], self.map_dim[1], k))
        self.create_paths_to_consider_dict(**kwargs)

    def update_nei(self, agents, **kwargs):
        h = kwargs['h']
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
        # reset
        self.reset_k_step(**kwargs)

    def create_paths_to_consider_dict(self, **kwargs):
        p_h, p_l = kwargs['p_h'], kwargs['p_l']
        self.agents_to_consider_dict = {}
        # just index
        for nei_agent in self.nei_list:
            if nei_agent.index < self.index:
                if random.random() < p_h:
                    self.agents_to_consider_dict[nei_agent.name] = nei_agent
            elif random.random() < p_l:
                self.agents_to_consider_dict[nei_agent.name] = nei_agent

    def change_priority(self, priority, **kwargs):
        self.index = priority
        self.name = f'agent_{self.index}'
        self.reset_k_step(**kwargs)

    def get_magnet_list(self):
        # h_value = self.h_func(self.start_node, self.goal_node)
        # h_value = 16
        h_value = 4
        magnet_list = [h_value]
        while h_value > 0.5:
            # h_value /= 4
            h_value /= 2
            # h_value /= 1.5
            # h_value /= 1.2
            magnet_list.append(h_value)
        return magnet_list

    def set_area_circle(self, i_time, curr_node, magnet_list, nei_nodes, nei_nodes_dict):
        max_r = len(magnet_list)
        # max_r = min(5, len(self.b_my_magnet_list))
        if max_r > 0:
            # for i_node in self.nodes:
            for i_node in nei_nodes:
                if abs(i_node.x - curr_node.x) > max_r or abs(i_node.y - curr_node.y) > max_r:
                    continue
                # around the curr_node
                distance = math.floor(euclidean_distance_nodes(i_node, curr_node))
                if distance < max_r:
                    self.magnet_field[i_node.x, i_node.y, i_time] += magnet_list[distance]

    def create_magnet_field(self, **kwargs):
        if self.curr_node.xy_name == self.goal_node.xy_name:  # if the agent reached the goal
            return
        magnet_list = self.get_magnet_list()
        for i_time, node in enumerate(self.path):
            nei_nodes, nei_nodes_dict = get_nei_nodes(node, len(magnet_list), self.nodes_dict)
            self.set_area_circle(i_time, node, magnet_list, nei_nodes, nei_nodes_dict)
        # plot_magnet_field(self.magnet_field)

    def exchange_data(self, **kwargs):
        # exchange paths
        for nei in self.nei_list:
            nei.nei_finished_dict[self.name] = self.finished_k_iter
            nei.nei_paths_dict[self.name] = self.path
            nei.nei_h_dict[self.name] = self.h
            self.stats_n_messages += 1
            self.stats_n_step_m += 1

    def agents_to_consider_are_finished(self):
        for agent_name, agent in self.agents_to_consider_dict.items():
            if not self.nei_finished_dict[agent_name]:
                return False
        return True

    def plan(self, **kwargs):
        """
        Output options:
        - already has a plan
        - other higher priority agents are not ready yet with their plans
        - failed to build the plan
        - successfully built the plan
        """
        we_good = True
        # two things to say: is the agent waiting
        if self.finished_k_iter:  # already has a plan
            return we_good, {}
        if not self.agents_to_consider_are_finished():  # if all above are done, or just nobody nearby
            return we_good, {}
        check_r = self.k_transform(**kwargs)  # basically just the k itself
        # consider all higher priority according to index
        agents_to_consider_list = list(self.agents_to_consider_dict.values())
        paths_to_consider_dict = {agent.name: agent.path for agent in agents_to_consider_list}
        # build constraints
        v_constr_dict, e_constr_dict, _ = build_constraints(self.nodes, paths_to_consider_dict)
        perm_constr_dict = build_k_step_perm_constr_dict(self.nodes, paths_to_consider_dict, check_r)
        # build magnetic fields
        nei_magnets = build_nei_magnets(agents_to_consider_list, **kwargs)
        mag_cost_func = build_mag_cost_func(agents_to_consider_list, nei_magnets, **kwargs)
        we_good, info = self.calc_a_star_plan(v_constr_dict, e_constr_dict, perm_constr_dict, k_time=check_r,
                                                mag_cost_func=mag_cost_func, **kwargs)
        if we_good:
            self.finished_k_iter = True
            self.create_magnet_field(**kwargs)
            # plot_magnet_field(self.path, nei_magnets)
        return we_good, info


def create_agents(start_nodes, goal_nodes, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit, map_dim):
    # Creating agents
    agents = []
    agents_dict = {}
    n_agent = 0
    for start_node, goal_node in zip(start_nodes, goal_nodes):
        agent = KMagnetPrPAgent(n_agent, start_node, goal_node, nodes, nodes_dict, h_func, plotter, middle_plot, iter_limit,
                                map_dim)
        agents.append(agent)
        agents_dict[agent.name] = agent
        n_agent += 1

    return agents, agents_dict


def all_find_nei(agents: List[KMagnetPrPAgent], **kwargs):
    for agent in agents:
        # find_nei
        agent.update_nei(agents, **kwargs)
    return {}


def all_plan(agents: List[KMagnetPrPAgent], alg_info, **kwargs):
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
        we_good, info = agent.plan(**kwargs)
        # stats
        end_time = time.time() - start_time
        runtime += end_time
        runtime_dist.append(end_time)
        succeeded_list.append(agent.finished_k_iter)
        if not we_good:
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


def all_exchange_data(agents: List[KMagnetPrPAgent], **kwargs):
    h = kwargs['h']
    # check_radius = k
    check_radius = h
    runtime, runtime_dist = 0, []

    # exchange paths
    for agent in agents:
        start_time = time.time()
        agent.exchange_data(**kwargs)
        # stats
        end_time = time.time() - start_time
        runtime += end_time
        runtime_dist.append(end_time)

    func_info = {
        'runtime': runtime,
        'dist_runtime': max(runtime_dist)
    }
    return func_info


def all_change_priority(agents: List[KMagnetPrPAgent], **kwargs):
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


def run_k_distr_magnets_pp(start_nodes, goal_nodes, nodes, nodes_dict, h_func, **kwargs):
    if 'k' not in kwargs or 'h' not in kwargs:
        raise RuntimeError("'k' or 'h' not in kwargs")
    if 'k_step_iteration_limit' in kwargs:
        k_step_iteration_limit = kwargs['k_step_iteration_limit']
    else:
        k_step_iteration_limit = 200
        kwargs['k_step_iteration_limit'] = k_step_iteration_limit
    alg_name = kwargs['alg_name'] if 'alg_name' in kwargs else f'k-Magnets-PrP'
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

        all_find_nei(agents, **kwargs)  # agents: find neighbours + reset

        # SMALL ITERATIONS
        while True:
            kwargs['small_iteration'] += 1

            func_info = all_exchange_data(agents, **kwargs)  # agents
            if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
                return None, {'agents': agents, 'success_rate': 0}

            func_info = all_plan(agents, alg_info, **kwargs)  # agents - replan - implemented here
            if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
                return None, {'agents': agents, 'success_rate': 0}
            all_succeeded = func_info['all_succeeded']

            if len(func_info['failed_paths_dict']) > 0:
                print(f"\n###########################\nPRIORITY CHANGE \n###########################\n")
                func_info = all_change_priority(agents, **kwargs)  # agents: new index + reset
                if check_if_limit_is_crossed(func_info, alg_info, **kwargs):
                    return None, {'agents': agents, 'success_rate': 0}
                continue

            if all_succeeded:
                break

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
                alg_info['space_metric'] = get_space_metric(cut_full_plans, radius=kwargs['k'])
            return cut_full_plans, alg_info

        if k_step_iteration > k_step_iteration_limit - 1:
            print(f'\n[LIMIT]: k_step_iteration: {k_step_iteration} > limit: {k_step_iteration_limit}')
            break

    return None, {'agents': agents, 'success_rate': 0}


def main():
    n_agents = 10
    # img_dir = 'my_map_10_10_room.map'  # 10-10
    img_dir = 'empty-48-48.map'  # 48-48
    # img_dir = 'random-64-64-10.map'  # 64-64
    # img_dir = 'warehouse-10-20-10-2-1.map'  # 63-161
    # img_dir = 'lt_gallowstemplar_n.map'  # 180-251

    # --------------------------------------------------- #
    # --------------------------------------------------- #
    # for the Magnets-PP algorithm
    # magnet_w = 0
    # magnet_w = 1
    # magnet_w = 2
    magnet_w = 5
    k = 5  # my planning
    h = 5  # my step
    pref_paths_type = 'pref_index'
    p_h = 1
    p_l = 0
    # reset_type = 'reset_start'
    reset_type = 'reset_step'
    alg_name = f'{k}-{h}-Magnets-PrP'
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
            algorithm=run_k_distr_magnets_pp,
            img_dir=img_dir,
            alg_name=alg_name,
            magnet_w=magnet_w,
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