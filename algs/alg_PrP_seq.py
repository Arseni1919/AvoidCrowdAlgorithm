import random
import time
import matplotlib.pyplot as plt
import cProfile
import pstats
from algs.test_mapf_alg import test_mapf_alg_from_pic

# from algs.metrics import check_for_collisions, c_v_check_for_agent, c_e_check_for_agent
from algs.metrics import build_constraints, get_agents_in_conf, check_plan, get_alg_info_dict
from algs.metrics import limit_is_crossed
from algs.alg_a_star_space_time import a_star
from algs.alg_depth_first_a_star import df_a_star


class PPAgent:
    def __init__(self, index: int, start_node, goal_node):
        self.index = index
        self.name = f'agent_{index}'
        self.start_node = start_node
        self.start_xy = self.start_node.xy_name
        self.goal_node = goal_node
        self.goal_xy = self.goal_node.xy_name
        self.path = []
        self.stats_n_closed = 0
        self.stats_n_calls = 0
        self.stats_runtime = 0


def create_agents(start_nodes, goal_nodes):
    agents, agents_dict = [], {}
    index = 0
    for start_node, goal_node in zip(start_nodes, goal_nodes):
        new_agent = PPAgent(index, start_node, goal_node)
        agents.append(new_agent)
        agents_dict[new_agent.name] = new_agent
        index += 1
    return agents, agents_dict


def update_path(update_agent, higher_agents, nodes, nodes_dict, h_func, **kwargs):
    # print('\rFUNC: update_path', end='')
    sub_results = {agent.name: agent.path for agent in higher_agents}
    v_constr_dict, e_constr_dict, perm_constr_dict = build_constraints(nodes, sub_results)
    # print('\rBEFORE A*', end='')
    print(f'\n ---------- ({kwargs["alg_name"]}) A* {update_agent.name} ---------- \n')
    a_star_func = kwargs['a_star_func']
    new_path, a_s_info = a_star_func(start=update_agent.start_node, goal=update_agent.goal_node,
                                     nodes=nodes, h_func=h_func,
                                     nodes_dict=nodes_dict,
                                     v_constr_dict=v_constr_dict,
                                     e_constr_dict=e_constr_dict,
                                     perm_constr_dict=perm_constr_dict, **kwargs)
    # stats
    update_agent.stats_n_calls += 1
    return new_path, a_s_info


def run_pp(start_nodes, goal_nodes, nodes, nodes_dict, h_func, **kwargs):
    runtime = 0
    plotter = kwargs['plotter'] if 'plotter' in kwargs else None
    final_plot = kwargs['final_plot'] if 'final_plot' in kwargs else True
    plot_per = kwargs['plot_per'] if 'plot_per' in kwargs else 10
    alg_name = kwargs['alg_name'] if 'alg_name' in kwargs else 'PP'

    agents, agents_dict = create_agents(start_nodes, goal_nodes)
    agent_names = [agent.name for agent in agents]

    plan = None
    alg_info = get_alg_info_dict()

    # ITERATIONS
    for iteration in range(1000000):

        if limit_is_crossed(runtime, alg_info, **kwargs):
            break

        # PICK A RANDOM ORDE
        random.shuffle(agent_names)
        new_order = [agents_dict[agent_name] for agent_name in agent_names]

        # PLAN
        higher_agents = []
        to_continue = False
        for agent in new_order:
            new_path, a_s_info = update_path(agent, higher_agents, nodes, nodes_dict, h_func, **kwargs)
            alg_info['a_star_calls_counter'] += 1
            alg_info['a_star_runtimes'].append(a_s_info['runtime'])
            alg_info['a_star_n_closed'].append(a_s_info['n_closed'])
            # STATS + LIMITS
            runtime += a_s_info['runtime']
            if new_path and not limit_is_crossed(runtime, alg_info, **kwargs):
                agent.path = new_path
            else:
                print('###################### random restart ######################')
                to_continue = True
                break
            higher_agents.append(agent)

        if to_continue:
            continue

        # CHECK PLAN
        plan = {agent.name: agent.path for agent in agents}
        there_is_col, c_v, c_e, cost = check_plan(agents, plan, alg_name, alg_info, runtime, iteration)
        if not there_is_col:
            if final_plot:
                print(f'#########################################################')
                print(f'#########################################################')
                print(f'#########################################################')
                print(f"runtime: {runtime}\n{cost=}")
                print(f"a_star_n_closed: {sum(alg_info['a_star_n_closed'])}")
                plotter.plot_mapf_paths(paths_dict=plan, nodes=nodes, **kwargs)

            alg_info['success_rate'] = 1
            alg_info['sol_quality'] = cost
            alg_info['runtime'] = runtime
            alg_info['a_star_calls_per_agent'] = [agent.stats_n_calls for agent in agents]
            return plan, alg_info

    return plan, alg_info


def main():
    n_agents = 400
    # img_dir = 'my_map_10_10_room.map'  # 10-10
    img_dir = 'empty-48-48.map'  # 48-48
    # img_dir = 'random-64-64-10.map'  # 64-64
    # img_dir = 'warehouse-10-20-10-2-1.map'  # 63-161
    # img_dir = 'lt_gallowstemplar_n.map'  # 180-251
    # img_dir = 'random-32-32-10.map'  # 32-32               | LNS |

    # --------------------------------------------------- #
    # --------------------------------------------------- #

    # random_seed = True
    random_seed = False
    seed = 839
    PLOT_PER = 1
    PLOT_RATE = 0.5
    final_plot = True
    # final_plot = False

    A_STAR_ITER_LIMIT = 5e7
    A_STAR_CALLS_LIMIT = 1000

    to_use_profiler = True
    profiler = cProfile.Profile()
    if to_use_profiler:
        profiler.enable()

    for i in range(3):
        print(f'\n[run {i}]')
        result, info = test_mapf_alg_from_pic(
            algorithm=run_pp,
            alg_name='PP',
            a_star_func=a_star,
            img_dir=img_dir,
            n_agents=n_agents,
            random_seed=random_seed,
            seed=seed,
            limit_type='norm_time',
            a_star_iter_limit=A_STAR_ITER_LIMIT,
            a_star_calls_limit=A_STAR_CALLS_LIMIT,
            max_time=50,
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
        stats.dump_stats('../stats/results_pp.pstat')


if __name__ == '__main__':
    main()
