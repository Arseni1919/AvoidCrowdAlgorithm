import time


def limit_is_crossed(runtime, alg_info, **kwargs):
    if 'limit_type' not in kwargs:
        raise RuntimeError('limit_type not in kwargs')

    limit_type = kwargs['limit_type']
    max_time = kwargs['max_time'] if 'max_time' in kwargs else 60
    a_star_calls_limit = kwargs['a_star_calls_limit'] if 'a_star_calls_limit' in kwargs else 1e100
    a_star_closed_nodes_limit = kwargs['a_star_closed_nodes_limit'] if 'a_star_closed_nodes_limit' in kwargs else 1e100

    if limit_type == 'norm_time':
        crossed = runtime > max_time * 60
        if crossed:
            print(f'\n[LIMIT]: norm_time: {runtime} > limit: {max_time * 60}')
        return crossed
    elif limit_type == 'dist_time':
        crossed = alg_info['dist_runtime'] > max_time * 60
        if crossed:
            print(f"\n[LIMIT]: dist_runtime: {alg_info['dist_runtime']} > limit: {max_time * 60}")
        return crossed
    elif limit_type == 'norm_a_star_calls':
        crossed = alg_info['a_star_calls_counter'] >= a_star_calls_limit
        if crossed:
            print(f"\n[LIMIT]: a_star_calls_counter: {alg_info['a_star_calls_counter']} > limit: {a_star_calls_limit}")
        return crossed
    elif limit_type == 'dist_a_star_calls':
        crossed = alg_info['a_star_calls_counter_dist'] >= a_star_calls_limit
        if crossed:
            print(f"\n[LIMIT]: a_star_calls_counter_dist: {alg_info['a_star_calls_counter_dist']} > limit: {a_star_calls_limit}")
        return crossed
    elif limit_type == 'norm_a_star_closed':
        a_star_n_closed_counter = sum(alg_info['a_star_n_closed'])
        crossed = a_star_n_closed_counter >= a_star_closed_nodes_limit
        if crossed:
            print(f"\n[LIMIT]: a_star_n_closed_counter: {a_star_n_closed_counter} > limit: {a_star_closed_nodes_limit}")
        return crossed
    elif limit_type == 'dist_a_star_closed':
        crossed = alg_info['a_star_n_closed_dist'] >= a_star_closed_nodes_limit
        if crossed:
            print(f"\n[LIMIT]: a_star_n_closed_dist: {alg_info['a_star_n_closed_dist']} > limit: {a_star_closed_nodes_limit}")
        return crossed
    else:
        raise RuntimeError('no valid limit_type')


def get_alg_info_dict(**kwargs):
    alg_info = {'success_rate': 0,
                'sol_quality': 0,
                'runtime': 0,
                'a_star_calls_counter': 0,
                'a_star_calls_counter_dist': 0,
                'dist_runtime': 0,
                'a_star_n_closed_dist': 0,
                'a_star_runtimes': [],
                'a_star_n_closed': [],
                'n_agents_conf': [],
                'a_star_calls_per_agent': [],
                'n_messages_per_agent': [],
                'n_messages_per_iter': [],
                'confs_per_iter': []}
    alg_info.update(kwargs)
    return alg_info


def get_agents_in_conf(c_v_list, c_e_list):
    agents_in_conf = [conf[1] for conf in c_v_list]
    agents_in_conf.extend([conf[1] for conf in c_e_list])
    agents_in_conf = list(set(agents_in_conf))
    return agents_in_conf


def build_constraints(nodes, other_paths):
    v_constr_dict = {node.xy_name: [] for node in nodes}
    e_constr_dict = {node.xy_name: [] for node in nodes}
    perm_constr_dict = {node.xy_name: [] for node in nodes}

    for agent_name, path in other_paths.items():
        if len(path) > 0:
            final_node = path[-1]
            perm_constr_dict[final_node.xy_name].append(final_node.t)
            perm_constr_dict[final_node.xy_name] = [max(perm_constr_dict[final_node.xy_name])]

            prev_node = path[0]
            for node in path:
                # vertex
                v_constr_dict[f'{node.x}_{node.y}'].append(node.t)
                # edge
                if prev_node.xy_name != node.xy_name:
                    e_constr_dict[f'{prev_node.x}_{prev_node.y}'].append((node.x, node.y, node.t))
                prev_node = node
    return v_constr_dict, e_constr_dict, perm_constr_dict


def c_v_check_for_agent(agent_1: str, path_1, results, immediate=False):
    """
    c_v_for_agent_list: (agents name 1, agents name 2, x, y, t)
    """
    if type(agent_1) is not str:
        raise RuntimeError('type(agent_1) is not str')
    c_v_for_agent_list = []
    if len(path_1) < 1:
        return c_v_for_agent_list
    for agent_2, path_2 in results.items():
        # if type(agent_2) is not str:
        #     raise RuntimeError('type(agent_2) is not str')
        if agent_2 != agent_1:
            for t in range(max(len(path_1), len(path_2))):
                node_1 = path_1[min(t, len(path_1) - 1)]
                node_2 = path_2[min(t, len(path_2) - 1)]
                if (node_1.x, node_1.y) == (node_2.x, node_2.y):
                    c_v_for_agent_list.append((agent_1, agent_2, node_1.x, node_1.y, t))
                    if immediate:
                        return c_v_for_agent_list
    return c_v_for_agent_list


def vertex_col_check(results, immediate=False):
    vertex_col_list = []
    for agent_1, path_1 in results.items():
        c_v_for_agent_list = c_v_check_for_agent(agent_1, path_1, results, immediate=immediate)
        vertex_col_list.extend(c_v_for_agent_list)
        if immediate:
            return vertex_col_list
    return vertex_col_list


def c_e_check_for_agent(agent_1: str, path_1, results, immediate=False):
    """
    c_e_check_for_agent: (agents name 1, agents name 2, x, y, x, y, t)
    """
    if type(agent_1) is not str:
        raise RuntimeError('type(agent_1) is not str')
    c_e_for_agent_list = []
    if len(path_1) <= 1:
        return c_e_for_agent_list
    for agent_2, path_2 in results.items():
        if agent_2 != agent_1:
            if len(path_2) > 1:
                prev_node_1 = path_1[0]
                prev_node_2 = path_2[0]
                for t in range(1, min(len(path_1), len(path_2))):
                    node_1 = path_1[t]
                    node_2 = path_2[t]
                    if (prev_node_1.x, prev_node_1.y, node_1.x, node_1.y) == (node_2.x, node_2.y, prev_node_2.x, prev_node_2.y):
                        c_e_for_agent_list.append((agent_1, agent_2, prev_node_1.x, prev_node_1.y, node_1.x, node_1.y, t))
                        if immediate:
                            return c_e_for_agent_list
                    prev_node_1 = node_1
                    prev_node_2 = node_2
    return c_e_for_agent_list


def edge_col_check(results, immediate=False):
    edge_col_list = []
    for agent_1, path_1 in results.items():
        c_e_for_agent_list = c_e_check_for_agent(agent_1, path_1, results, immediate=immediate)
        edge_col_list.extend(c_e_for_agent_list)
        if immediate:
            return edge_col_list
    return edge_col_list


def check_for_collisions(results, immediate=False):
    """
    results: {'agents str': [Nodes heap_list]}
    """
    print('\nStart check_for_collisions...')
    if results:
        vertex_col_list = vertex_col_check(results, immediate=immediate)
        edge_col_list = edge_col_check(results, immediate=immediate)
        there_is_col = len(vertex_col_list) > 0 or len(edge_col_list) > 0
        return there_is_col, vertex_col_list, edge_col_list
    return True, [], []


def iteration_print(agents, plan, alg_name, alg_info, runtime, iteration):
    plan_lngths = [len(path) for path in plan.values()]
    if 0 in plan_lngths:
        raise RuntimeError('0 in plan_lngths')
    cost = sum([len(path) for path in plan.values()])
    print(f'\r---\n'
          f'ITERATION PRINT: '
          f'[{alg_name}][{len(agents)} agents][A* calls: {alg_info["a_star_calls_counter"]}][time: {runtime:0.2f}s]'
          f'[step {iteration}]\n'
          f'cost: {cost}\n'
          f'---\n')
    return cost


def check_plan(agents, plan, alg_name, alg_info, runtime, iteration, immediate=False):
    cost = iteration_print(agents, plan, alg_name, alg_info, runtime, iteration)
    # cost = sum([len(path) for path in plan.values()])
    there_is_col, c_v, c_e = check_for_collisions(plan, immediate=False)
    return there_is_col, c_v, c_e, cost


def just_check_plans(plans):
    cost = sum([len(path) for path in plans.values()])
    there_is_col, c_v, c_e = check_for_collisions(plans, immediate=False)
    # there_is_col, c_v, c_e = check_for_collisions(plans, immediate=True)
    return there_is_col, c_v, c_e, cost


def get_final_t(path_1, plans, k):
    len_list = [len(path_1)]
    len_list.extend([len(v) for v in plans.values()])
    max_len = max(len_list)
    return min(max_len, k)


def check_single_agent_k_step_c_v(agent_1, path_1, plans, k, immediate=False):
    final_t = get_final_t(path_1, plans, k)
    c_v_list = []
    if len(path_1) == 0:
        return c_v_list
    for agent_2, path_2 in plans.items():
        if len(path_2) == 0:
            continue
        if agent_1 != agent_2:
            for t in range(final_t):
                node_1 = path_1[min(t, len(path_1) - 1)]
                node_2 = path_2[min(t, len(path_2) - 1)]
                if (node_1.x, node_1.y) == (node_2.x, node_2.y):
                    c_v_list.append((agent_1, agent_2, node_1.x, node_1.y, t))
                    if immediate:
                        return c_v_list
    return c_v_list


def check_k_step_c_v(plans, k, immediate=False):
    c_v_list = []
    for agent_1, path_1 in plans.items():
        single_agent_c_v_list = check_single_agent_k_step_c_v(agent_1, path_1, plans, k, immediate)
        c_v_list.extend(single_agent_c_v_list)
        if len(c_v_list) > 0 and immediate:
            return c_v_list
    return c_v_list


def check_single_agent_k_step_c_e(agent_1, path_1, plans, k, immediate=False):
    # final_t = get_final_t(path_1, plans, k)
    c_e_list = []
    if len(path_1) <= 1:
        return c_e_list
    for agent_2, path_2 in plans.items():
        if len(path_1) <= 1:
            return c_e_list
        if agent_1 != agent_2:
            prev_node_1 = path_1[0]
            prev_node_2 = path_2[0]
            for t in range(1, min(k, len(path_1), len(path_2))):
                node_1 = path_1[t]
                node_2 = path_2[t]
                if (prev_node_1.x, prev_node_1.y, node_1.x, node_1.y) == (
                        node_2.x, node_2.y, prev_node_2.x, prev_node_2.y):
                    c_e_list.append(
                        (agent_1, agent_2, prev_node_1.x, prev_node_1.y, node_1.x, node_1.y, t))
                    if immediate:
                        return c_e_list
                prev_node_1 = node_1
                prev_node_2 = node_2
    return c_e_list


def check_k_step_c_e(plans, k, immediate=False):
    c_e_list = []
    for agent_1, path_1 in plans.items():
        single_agent_c_e_list = check_single_agent_k_step_c_e(agent_1, path_1, plans, k, immediate)
        c_e_list.extend(single_agent_c_e_list)
        if len(c_e_list) > 0 and immediate:
            return c_e_list
    return c_e_list


def just_check_k_step_plans(plans, k, immediate=False):
    # k_step_plans = {agent_name: path[:k] for agent_name, path in plans.items()}
    # there_is_col, c_v, c_e = check_for_collisions(k_step_plans, immediate=False)
    c_v_list = check_k_step_c_v(plans, k, immediate)
    c_e_list = check_k_step_c_e(plans, k, immediate)
    there_is_col = len(c_v_list) > 0 or len(c_e_list) > 0
    return there_is_col, c_v_list, c_e_list


def build_k_step_perm_constr_dict(nodes, other_paths, k):
    perm_constr_dict = {node.xy_name: [] for node in nodes}
    for agent_name, path in other_paths.items():
        if 0 < len(path) <= k:
            final_node = path[-1]
            # perm_constr_dict[final_node.xy_name].append(final_node.t)
            perm_constr_dict[final_node.xy_name].append(len(path) - 1)
            perm_constr_dict[final_node.xy_name] = [max(perm_constr_dict[final_node.xy_name])]

    return perm_constr_dict


def main():
    pass


if __name__ == '__main__':
    main()


