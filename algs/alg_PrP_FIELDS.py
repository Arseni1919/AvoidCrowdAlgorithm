import numpy as np

from algs.test_mapf_alg import test_mapf_alg_from_pic
from functions import *
# from algs.metrics import check_for_collisions, c_v_check_for_agent, c_e_check_for_agent
from algs.metrics import build_constraints, get_agents_in_conf, check_plan, just_check_plans, get_alg_info_dict
from functions import limit_is_crossed
from algs.alg_a_star_space_time import a_star_xyt
from algs.alg_depth_first_a_star import df_a_star


def plot_magnet_field(path, data):
    plt.rcParams["figure.figsize"] = [8.00, 8.00]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot field
    if data is not None:
        x_l, y_l, z_l = np.nonzero(data > 0)
        col = data[data > 0]
        alpha_col = col / max(col)
        # alpha_col = np.exp(col) / max(np.exp(col))
        cm = plt.colormaps['Reds']  # , cmap=cm
        ax.scatter(x_l, y_l, z_l, c=col, alpha=alpha_col, marker='s', cmap=cm)
    # plot line
    if path:
        path_x = [node.x for node in path]
        path_y = [node.y for node in path]
        path_z = list(range(len(path_x)))
        ax.plot(path_x, path_y, path_z)
    plt.show()
    # plt.pause(2)


def get_nei_nodes(curr_node, nei_r, nodes_dict):
    nei_nodes_dict = {}
    open_list = [curr_node]
    while len(open_list) > 0:
        i_node = open_list.pop()
        i_node_distance = euclidean_distance_nodes(curr_node, i_node)
        if i_node_distance <= nei_r:
            nei_nodes_dict[i_node.xy_name] = i_node
            for node_nei_name in i_node.neighbours:
                if node_nei_name not in nei_nodes_dict:
                    open_list.append(nodes_dict[node_nei_name])
    nei_nodes = list(nei_nodes_dict.values())
    return nei_nodes, nei_nodes_dict


class PPAgent:
    def __init__(self, index: int, start_node, goal_node, nodes, nodes_dict, h_func, **kwargs):
        self.index = index
        self.name = f'agent_{index}'
        self.start_node = start_node
        self.start_xy = self.start_node.xy_name
        self.goal_node = goal_node
        self.goal_xy = self.goal_node.xy_name
        self.nodes = nodes
        self.nodes_dict = nodes_dict
        self.h_func = h_func
        self.map_dim = kwargs['map_dim']
        self.path = []
        self.stats_n_closed = 0
        self.stats_n_calls = 0
        self.stats_runtime = 0
        self.magnet_field = None

    def get_magnet_list(self):
        # h_value = self.h_func(self.start_node, self.goal_node)
        h_value = 100
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

    def create_magnet_field(self):
        self.magnet_field = np.zeros((self.map_dim[0], self.map_dim[1], len(self.path)))
        magnet_list = self.get_magnet_list()
        for i_time, node in enumerate(self.path):
            nei_nodes, nei_nodes_dict = get_nei_nodes(node, len(magnet_list), self.nodes_dict)
            self.set_area_circle(i_time, node, magnet_list, nei_nodes, nei_nodes_dict)
        # plot_magnet_field(self.magnet_field)


def create_agents(start_nodes, goal_nodes, nodes, nodes_dict, h_func, **kwargs):
    agents, agents_dict = [], {}
    index = 0
    for start_node, goal_node in zip(start_nodes, goal_nodes):
        new_agent = PPAgent(index, start_node, goal_node, nodes, nodes_dict, h_func, **kwargs)
        agents.append(new_agent)
        agents_dict[new_agent.name] = new_agent
        index += 1
    return agents, agents_dict


def build_nei_magnets(higher_agents, **kwargs):
    if len(higher_agents) == 0:
        return None, 0
    map_dim = kwargs['map_dim']
    longest_path_length = max([len(agent.path) for agent in higher_agents])
    nei_magnets = np.zeros((map_dim[0], map_dim[1], longest_path_length))  # x, y, t
    for nei_agent in higher_agents:
        curr_path_len = len(nei_agent.path)
        nei_magnets[:, :, :curr_path_len] += nei_agent.magnet_field
    nei_magnets /= np.max(nei_magnets)
    return nei_magnets, longest_path_length


def build_mag_cost_func(higher_agents, nei_magnets, longest_path_length,  **kwargs):
    if len(higher_agents) == 0:
        return lambda x, y, t: 0

    def mag_cost_func(x, y, t):
        if t >= longest_path_length:
            return 0
        return nei_magnets[x, y, t]
    return mag_cost_func


def update_path(update_agent, order_of_agent, higher_agents, nodes, nodes_dict, h_func, **kwargs):
    # print('\rFUNC: update_path', end='')
    sub_results = {agent.name: agent.path for agent in higher_agents}
    v_constr_dict, e_constr_dict, perm_constr_dict = build_constraints(nodes, sub_results)
    # build mag_cost function
    nei_magnets, longest_path_length = build_nei_magnets(higher_agents, **kwargs)
    mag_cost_func = build_mag_cost_func(higher_agents, nei_magnets, longest_path_length,  **kwargs)
    print(f'\n ---------- ({kwargs["alg_name"]}) A* order: {order_of_agent} ---------- \n')
    a_star_func = kwargs['a_star_func']
    new_path, a_s_info = a_star_func(start=update_agent.start_node, goal=update_agent.goal_node,
                                     nodes=nodes, h_func=h_func, mag_cost_func=mag_cost_func,
                                     nodes_dict=nodes_dict,
                                     v_constr_dict=v_constr_dict,
                                     e_constr_dict=e_constr_dict,
                                     perm_constr_dict=perm_constr_dict, **kwargs)
    # stats
    update_agent.stats_n_calls += 1
    return new_path, a_s_info, nei_magnets


def run_pp_fields(start_nodes, goal_nodes, nodes, nodes_dict, h_func, **kwargs):
    runtime = 0
    plotter = kwargs['plotter'] if 'plotter' in kwargs else None
    final_plot = kwargs['final_plot'] if 'final_plot' in kwargs else True
    alg_name = kwargs['alg_name']

    agents, agents_dict = create_agents(start_nodes, goal_nodes, nodes, nodes_dict, h_func, **kwargs)
    agent_names = [agent.name for agent in agents]

    alg_info = get_alg_info_dict()

    # ITERATIONS
    for iteration in range(1000000):

        if limit_is_crossed(runtime, alg_info, **kwargs):
            break

        # PICK A RANDOM ORDER
        random.shuffle(agent_names)
        new_order = [agents_dict[agent_name] for agent_name in agent_names]

        # PLAN
        higher_agents = []
        to_continue = False
        for order_of_agent, agent in enumerate(new_order):
            new_path, a_s_info, nei_magnets = update_path(agent, order_of_agent, higher_agents, nodes, nodes_dict, h_func, **kwargs)
            alg_info['a_star_calls_counter'] += 1
            alg_info['a_star_runtimes'].append(a_s_info['runtime'])
            alg_info['a_star_n_closed'].append(a_s_info['n_closed'])
            # STATS + LIMITS
            runtime += a_s_info['runtime']
            if new_path and not limit_is_crossed(runtime, alg_info, **kwargs):
                agent.path = new_path
                agent.create_magnet_field()
                if order_of_agent > 3:
                    plot_magnet_field(agent.path, nei_magnets)
            else:
                print('###################### random restart ######################')
                to_continue = True
                break
            higher_agents.append(agent)

        if to_continue:
            continue

        # CHECK PLAN
        plan = {agent.name: agent.path for agent in agents}
        there_is_col, c_v, c_e, cost = just_check_plans(plan)
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

    return None, alg_info


def main():
    n_agents = 200
    # img_dir = 'my_map_10_10_room.map'  # 10-10
    img_dir = 'empty-48-48.map'  # 48-48
    # img_dir = 'random-64-64-10.map'  # 64-64
    # img_dir = 'warehouse-10-20-10-2-1.map'  # 63-161
    # img_dir = 'lt_gallowstemplar_n.map'  # 180-251
    # img_dir = 'random-32-32-10.map'  # 32-32               | LNS |

    # --------------------------------------------------- #
    # --------------------------------------------------- #

    # for the alg
    # magnet_w = 0
    magnet_w = 2

    # random_seed = True
    random_seed = False
    seed = 839
    final_plot = True
    # final_plot = False
    PLOT_PER = 5
    PLOT_RATE = 0.5

    A_STAR_ITER_LIMIT = 5e7
    A_STAR_CALLS_LIMIT = 1000

    to_use_profiler = True
    profiler = cProfile.Profile()
    if to_use_profiler:
        profiler.enable()

    for i in range(3):
        print(f'\n[run {i}]')
        result, info = test_mapf_alg_from_pic(
            algorithm=run_pp_fields,
            alg_name='PrP-Magnets',
            magnet_w=magnet_w,
            a_star_func=a_star_xyt,
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
