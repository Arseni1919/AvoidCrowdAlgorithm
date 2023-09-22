from globals import *

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


def manhattan_distance_nodes(node1, node2):
    return abs(node1.x-node2.x) + abs(node1.y-node2.y)


@lru_cache(maxsize=128)
def euclidean_distance_nodes(node1, node2):
    # p = [node1.x, node1.y]
    # q = [node2.x, node2.y]
    return math.dist([node1.x, node1.y], [node2.x, node2.y])
    # return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


def rename_nodes_in_path(path):
    for t, node in enumerate(path):
        node.t = t
        node.ID = f'{node.x}_{node.y}_{t}'

