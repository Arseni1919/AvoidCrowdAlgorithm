import copy
import time
import matplotlib.pyplot as plt

from algs.alg_a_star_space_time import a_star
from algs.test_mapf_alg import test_mapf_alg_from_pic
from algs.metrics import c_v_check_for_agent, c_e_check_for_agent, build_constraints, \
    limit_is_crossed, get_alg_info_dict, check_plan, iteration_print
from algs.topological_sorting import topological_sorting


def preprint_func_name(func):
    def inner(*args, **kwargs):
        print(f'\rFUNC: {func.__name__}', end='')
        return func(*args, **kwargs)

    return inner


class PBSAgent:
    def __init__(self, index: int, start_node, goal_node):
        self.index = index
        self.name = f'agent_{index}'
        self.start_node = start_node
        self.start_xy = self.start_node.xy_name
        self.goal_node = goal_node
        self.goal_xy = self.goal_node.xy_name
        self.stats_n_closed = 0
        self.stats_n_calls = 0
        self.stats_runtime = 0


class PBSNode:
    def __init__(self, agents, agents_dict, index):
        self.agents = agents
        self.agent_dict = agents_dict
        self.index = index
        self.name = f'PBSNode {index}'
        self.plan = {agent.name: [] for agent in agents}
        self.cost = -1
        self.constraints = {agent.name: [] for agent in agents}
        self.ordering_rules = []
        self.partial_order = []
        self.parent = None

    def calc_cost(self):
        self.cost = sum([len(path) for path in self.plan.values()])

    def agent_names(self):
        return [agent.name for agent in self.agents]

    def update_ordering_rules(self):
        if len(self.partial_order) > 0:
            self.ordering_rules = []
            for agent_index, agent_name in enumerate(self.partial_order):
                for i in range(agent_index):
                    self.ordering_rules.append((self.partial_order[i], agent_name))


def create_agents(start_nodes, goal_nodes):
    agents, agents_dict = [], {}
    index = 0
    for start_node, goal_node in zip(start_nodes, goal_nodes):
        new_agent = PBSAgent(index, start_node, goal_node)
        agents.append(new_agent)
        agents_dict[new_agent.name] = new_agent
        index += 1
    return agents, agents_dict


def get_order_lists(pbs_node, agent):
    agents_names = []
    for a_1, a_2 in pbs_node.ordering_rules:
        agents_names.append(a_1)
        agents_names.append(a_2)
    agents_names = list(set(agents_names))
    # if len(agents_names) > 0:
    #     print(f'\rpbs_node [{pbs_node.index}]: agents_names -> {agents_names}\n')
    pbs_node.partial_order = topological_sorting(nodes=agents_names, sorting_rules=pbs_node.ordering_rules)
    if agent.name in pbs_node.partial_order:
        agent_index = pbs_node.partial_order.index(agent.name)
        higher_order_names = [pbs_node.partial_order[i] for i in range(0, agent_index)]
        lower_order_names = [pbs_node.partial_order[i] for i in range(agent_index + 1, len(pbs_node.partial_order))]
        higher_order_list = [pbs_node.agent_dict[name] for name in higher_order_names]
        lower_order_list = [pbs_node.agent_dict[name] for name in lower_order_names]
        return higher_order_list, lower_order_list
    return [], []


def collide_check(pbs_node, update_agent):
    print('\rFUNC: collide_check', end='')
    higher_order_list, _ = get_order_lists(pbs_node, update_agent)
    sub_results = {agent.name: pbs_node.plan[agent.name] for agent in higher_order_list}
    c_v_list = c_v_check_for_agent(update_agent.name, pbs_node.plan[update_agent.name], sub_results)
    if len(c_v_list) > 0:
        return True
    e_v_list = c_e_check_for_agent(update_agent.name, pbs_node.plan[update_agent.name], sub_results)
    if len(e_v_list):
        return True
    return False


def update_path(pbs_node, update_agent, nodes, nodes_dict, h_func, iter_limit, **kwargs):
    print('\rFUNC: update_path', end='')
    higher_order_list, _ = get_order_lists(pbs_node, update_agent)
    sub_results = {agent.name: pbs_node.plan[agent.name] for agent in higher_order_list}

    # c_v_list = c_v_check_for_agent(update_agent.name, pbs_node.path[update_agent.name], sub_results)
    # c_e_list = c_e_check_for_agent(update_agent.name, pbs_node.path[update_agent.name], sub_results)

    v_constr_dict, e_constr_dict, perm_constr_dict = build_constraints(nodes, sub_results)

    print('\rBEFORE A*', end='')
    a_star_func = kwargs['a_star_func']
    new_path, a_s_info = a_star_func(start=update_agent.start_node, goal=update_agent.goal_node,
                                     nodes=nodes, nodes_dict=nodes_dict,
                                     h_func=h_func, v_constr_dict=v_constr_dict, e_constr_dict=e_constr_dict,
                                     perm_constr_dict=perm_constr_dict,
                                     plotter=None, middle_plot=False, iter_limit=iter_limit)

    # stats
    update_agent.stats_n_calls += 1
    update_agent.stats_runtime += a_s_info['runtime']
    update_agent.stats_n_closed += a_s_info['n_closed']

    return new_path, a_s_info


# @preprint_func_name
def update_plan(pbs_node, agent, nodes, nodes_dict, h_func, iter_limit, **kwargs):
    print('\rFUNC: update_plan', end='')
    a_star_calls_inner_counter = 0
    a_star_runtimes = []
    a_star_n_closed = []
    update_list_names = topological_sorting(nodes=[agent.name], sorting_rules=pbs_node.ordering_rules)
    update_list = [pbs_node.agent_dict[agent_name] for agent_name in update_list_names]
    for update_agent in update_list:
        if collide_check(pbs_node, update_agent) or update_agent.name == agent.name:
            new_path, a_s_info = update_path(pbs_node, update_agent, nodes, nodes_dict, h_func,
                                             iter_limit, **kwargs)
            a_star_calls_inner_counter += 1
            a_star_runtimes.append(a_s_info['runtime'])
            a_star_n_closed.append(a_s_info['n_closed'])
            if new_path is None:
                return False, {
                    'a_star_calls_inner_counter': a_star_calls_inner_counter,
                    'a_star_runtimes': a_star_runtimes,
                    'a_star_n_closed': a_star_n_closed,
                }
            pbs_node.plan[update_agent.name] = new_path
    return True, {
        'a_star_calls_inner_counter': a_star_calls_inner_counter,
        'a_star_runtimes': a_star_runtimes,
        'a_star_n_closed': a_star_n_closed,
    }


def choose_conf(c_v, c_e):
    if len(c_v) > 0:
        # random.shuffle(c_v)
        return c_v[0], 'vertex'
    elif len(c_e) > 0:
        # random.shuffle(c_e)
        return c_e[0], 'edge'
    else:
        raise RuntimeError('no collisions')


# @preprint_func_name
def add_new_ordering(NEW_pbs_node, NEXT_pbs_node, agent, conf):
    print('\rFUNC: add_new_ordering', end='')
    h_agent, l_agent = None, None
    if agent.name == conf[0]:
        h_agent = conf[1]
        l_agent = conf[0]
    elif agent.name == conf[1]:
        h_agent = conf[0]
        l_agent = conf[1]
    else:
        raise RuntimeError('no such name in constr')

    NEW_pbs_node.ordering_rules.append((h_agent, l_agent))

    for (i_1, i_2) in NEW_pbs_node.ordering_rules:
        if (i_2, i_1) in NEW_pbs_node.ordering_rules:
            raise RuntimeError('there are upsidedown orders')

    return agent


def run_pbs(start_nodes, goal_nodes, nodes, nodes_dict, h_func, **kwargs):
    runtime = 0
    iter_limit = kwargs['a_star_iter_limit'] if 'a_star_iter_limit' in kwargs else 1e100
    plotter = kwargs['plotter'] if 'plotter' in kwargs else None
    final_plot = kwargs['final_plot'] if 'final_plot' in kwargs else True
    partial_order = kwargs['initial_ordering'] if 'initial_ordering' in kwargs else []
    alg_name = kwargs['alg_name'] if 'alg_name' in kwargs else 'PBS'

    alg_info = get_alg_info_dict()
    pbs_node_index = 0
    agents, agents_dict = create_agents(start_nodes, goal_nodes)
    root = PBSNode(agents, agents_dict, pbs_node_index)
    root.partial_order = partial_order
    root.update_ordering_rules()
    start_time = time.time()
    for agent in agents:
        success, up_info = update_plan(root, agent, nodes, nodes_dict, h_func, iter_limit, **kwargs)
        alg_info['a_star_calls_counter'] += up_info['a_star_calls_inner_counter']
        alg_info['a_star_runtimes'].extend(up_info['a_star_runtimes'])
        alg_info['a_star_n_closed'].extend(up_info['a_star_n_closed'])
        if not success:
            return None, {'success_rate': 0}

    # STATS
    alg_info['runtime'] += time.time() - start_time
    root.calc_cost()
    stack = [root]
    iteration = 0
    if len(start_nodes) >= 120:
        return None, {'success_rate': 0}
    while len(stack) > 0 and not limit_is_crossed(alg_info['runtime'], alg_info, **kwargs):
        start_time = time.time()
        iteration += 1
        NEXT_pbs_node = stack.pop()
        there_is_col, c_v, c_e, cost = check_plan(agents, NEXT_pbs_node.plan, alg_name, alg_info, alg_info["runtime"],
                                                  iteration, immediate=False)
        # there_is_col, c_v, c_e = check_for_collisions(NEXT_pbs_node.plan)
        # iteration_print(agents, NEXT_pbs_node.plan, alg_name, alg_info, alg_info["runtime"], iteration)
        # print(f'\r---\n'
        #       f'[{kwargs["alg_name"]}][{len(agents)} agents][A* calls: {alg_info["a_star_calls_counter"]}][time: {alg_info["runtime"]:0.2f}s][iter {iteration}]\n'
        #       f'PBS Node {NEXT_pbs_node.index}, stack: {len(stack)}\n'
        #       f'partial order: {NEXT_pbs_node.partial_order}\n'
        #       f'cost: {NEXT_pbs_node.cost}\n'
        #       f'---\n')

        if not there_is_col:
            if final_plot:
                print(f'order: {topological_sorting(NEXT_pbs_node.agent_names(), NEXT_pbs_node.ordering_rules)}\n')
                print(f'#########################################################')
                print(f'#########################################################')
                print(f'#########################################################')
                plotter.plot_mapf_paths(paths_dict=NEXT_pbs_node.plan, nodes=nodes, plot_per=1)

            alg_info['success_rate'] = 1
            alg_info['sol_quality'] = cost
            alg_info['runtime'] += time.time() - start_time
            alg_info['a_star_calls_per_agent'] = [agent.stats_n_calls for agent in agents]
            return NEXT_pbs_node.plan, alg_info

        conf, conf_type = choose_conf(c_v, c_e)
        for i in range(2):
            pbs_node_index += 1
            NEW_pbs_node = PBSNode(agents, agents_dict, pbs_node_index)
            NEW_pbs_node.plan = copy.deepcopy(NEXT_pbs_node.plan)
            NEW_pbs_node.ordering_rules = copy.deepcopy(NEXT_pbs_node.ordering_rules)

            agent = NEXT_pbs_node.agent_dict[conf[i]]
            agent = add_new_ordering(NEW_pbs_node, NEXT_pbs_node, agent, conf)
            success, up_info = update_plan(NEW_pbs_node, agent, nodes, nodes_dict, h_func, iter_limit, **kwargs)
            alg_info['a_star_calls_counter'] += up_info['a_star_calls_inner_counter']
            alg_info['a_star_runtimes'].extend(up_info['a_star_runtimes'])
            alg_info['a_star_n_closed'].extend(up_info['a_star_n_closed'])

            if success:
                NEW_pbs_node.calc_cost()
                stack.append(NEW_pbs_node)
                NEW_pbs_node.parent = NEXT_pbs_node
                stack.sort(key=lambda x: x.cost, reverse=True)
        alg_info['runtime'] += time.time() - start_time

    return None, {'success_rate': 0}


def main():
    # initial_ordering = [f'agent_{n}' for n in range(n_agents)]
    # result = test_mapf_alg_from_pic(algorithm=run_pbs, initial_ordering=initial_ordering, n_agents=n_agents)
    for i in range(20):
        print(f'\n[run {i}]')
        result, info = test_mapf_alg_from_pic(
            algorithm=run_pbs,
            initial_ordering=[],
            n_agents=n_agents,
            random_seed=random_seed,
            seed=seed,
            max_time=5,
            final_plot=True
        )

        if not random_seed:
            break

        # plt.show()
        plt.close()


if __name__ == '__main__':
    random_seed = True
    # random_seed = False
    seed = 710
    n_agents = 50

    main()
