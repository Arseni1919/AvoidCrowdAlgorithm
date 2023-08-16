from globals import *


def get_line_or_marker(index, kind):
    if kind == 'l':
        lines = ['--', '-', '-.', ':']
        index = index % len(lines)
        return lines[index]
    elif kind == 'm':
        markers = ['^', '1', '2', 'X', 'd', 'v', 'o']
        index = index % len(markers)
        return markers[index]
    else:
        raise RuntimeError('no such kind')


def set_plot_title(ax, title):
    ax.set_title(f'{title}', fontweight="bold")


def set_log(ax):
    # log = True
    log = False
    if log:
        ax.set_yscale('log')
    return log


def plot_text_in_cactus(ax, l_x, l_y):
    if len(l_x) > 0:
        ax.text(l_x[-1] - 2, l_y[-1], f'{l_x[-1] + 1}', bbox=dict(facecolor='yellow', alpha=0.75))


def set_legend(ax, framealpha=None):
    # to_put_legend = True
    to_put_legend = False
    if to_put_legend:
        if not framealpha:
            framealpha = 0
        legend_properties = {'weight': 'bold', 'size': 9}
        # legend_properties = {}
        if framealpha is not None:
            ax.legend(prop=legend_properties, framealpha=framealpha)
        else:
            ax.legend(prop=legend_properties)


def sub_plot_cactus_big_lines(ax, index, l_x, l_y, alg_name, alg_info):
    line_style = get_line_or_marker(index, 'l')
    linewidth = 2
    alpha = 0.75
    if 'color' in alg_info:
        ax.plot(l_x, l_y, line_style, label=f'{alg_name}', alpha=alpha, color=alg_info['color'], linewidth=linewidth)
    else:
        raise RuntimeError('no color! ufff')
        # ax.plot(l_x, l_y, line_style, label=f'{alg_name}', alpha=alpha, linewidth=linewidth)
    plot_text_in_cactus(ax, l_x, l_y)


def sub_plot_cactus_dist_lines(ax, index, l_x, l_y, alg_name, alg_info):
    line_style = f"-{get_line_or_marker(index, 'm')}"
    # line_style = get_line_or_marker(index, 'l')
    linewidth = 1
    alpha = 0.43
    if 'color' in alg_info:
        ax.plot(l_x, l_y, line_style, label=f'{alg_name}', alpha=alpha, color=alg_info['color'], linewidth=linewidth)
        # ax.plot(l_x, l_y, line_style, alpha=alpha, color=alg_info['color'], linewidth=linewidth)
    else:
        raise RuntimeError('no color! ufff')
        # ax.plot(l_x, l_y, line_style, label=f'{alg_name} (dist)', alpha=alpha, linewidth=linewidth)
    plot_text_in_cactus(ax, l_x, l_y)


# --------------------------------------------------------------------------------- #
# ----------------------------------getting lists---------------------------------- #
# --------------------------------------------------------------------------------- #


def get_list_n_run(statistics_dict, alg_name, n_agents, list_type, runs_per_n_agents, is_json=False):
    """
    {
        alg_name: {
            n_agents: {
                'success_rate': {run: None for run in range(runs_per_n_agents)},
                'sol_quality': {run: None for run in range(runs_per_n_agents)},
                'runtime': {run: None for run in range(runs_per_n_agents)},
            } for n_agents in n_agents_list
        } for alg_name, _ in algs_to_test_dict.items()
    }
    """
    curr_list = []
    for i_run in range(runs_per_n_agents):
        if is_json:
            n_agents = str(n_agents)
            i_run = str(i_run)
        curr_element = statistics_dict[alg_name][n_agents][list_type][i_run]
        if curr_element is not None:
            curr_list.append(curr_element)
    return curr_list


def get_list_sol_q_style(statistics_dict, alg_name, n_agents, list_type, runs_per_n_agents, algs_to_test_list,
                         is_json=False):
    """
    {
        alg_name: {
            n_agents: {
                'success_rate': {run: None for run in range(runs_per_n_agents)},
                'sol_quality': {run: None for run in range(runs_per_n_agents)},
                'runtime': {run: None for run in range(runs_per_n_agents)},
            } for n_agents in n_agents_list
        } for alg_name, _ in algs_to_test_dict.items()
    }
    """
    curr_list = []
    for i_run in range(runs_per_n_agents):
        if is_json:
            n_agents = str(n_agents)
            i_run = str(i_run)
        curr_element = statistics_dict[alg_name][n_agents][list_type][i_run]
        if curr_element is not None:
            to_insert = True
            for another_alg in algs_to_test_list:
                another_element = statistics_dict[another_alg][n_agents][list_type][i_run]
                if another_element is None:
                    to_insert = False
                    break
            if to_insert:
                curr_list.append(curr_element)
    return curr_list


def get_list_runtime(statistics_dict, alg_name, n_agents_list, list_type, runs_per_n_agents, is_json=False):
    """
    {
        alg_name: {
            n_agents: {
                'success_rate': {run: None for run in range(runs_per_n_agents)},
                'sol_quality': {run: None for run in range(runs_per_n_agents)},
                'runtime': {run: None for run in range(runs_per_n_agents)},
            } for n_agents in n_agents_list
        } for alg_name, _ in algs_to_test_dict.items()
    }
    """
    curr_list = []
    for n_agents in n_agents_list:
        for i_run in range(runs_per_n_agents):
            if is_json:
                n_agents = str(n_agents)
                i_run = str(i_run)
            curr_element = statistics_dict[alg_name][n_agents][list_type][i_run]
            if curr_element is not None:
                curr_list.append(curr_element)
    return curr_list


def get_list_a_star(statistics_dict, alg_name, n_agents_list, list_type, is_json=False):
    """
    {
        alg_name: {
            n_agents: {
                'success_rate': {run: None for run in range(runs_per_n_agents)},
                'sol_quality': {run: None for run in range(runs_per_n_agents)},
                'runtime': {run: None for run in range(runs_per_n_agents)},
                'iterations_time': {run: None for run in range(runs_per_n_agents)},
                'a_star_calls_counter': {run: None for run in range(runs_per_n_agents)},
                'a_star_calls_dist_counter': {run: None for run in range(runs_per_n_agents)},
                'a_star_runtimes': [],
                'a_star_n_closed': [],
            } for n_agents in n_agents_list
        } for alg_name, _ in algs_to_test_dict.items()
    }
    """
    curr_list = []
    for n_agents in n_agents_list:
        if is_json:
            n_agents = str(n_agents)
        curr_element = statistics_dict[alg_name][n_agents][list_type]
        if curr_element is not None:
            curr_list.extend(curr_element)
    return curr_list


# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #

def plot_step_in_mapf_paths(ax, info):
    ax.cla()
    paths_dict = info['paths_dict']
    nodes = info['nodes']
    side_x = info['side_x']
    side_y = info['side_y']
    t = info['t']
    img_dir = info['img_dir']

    field = np.zeros((side_x, side_y))

    if nodes:
        for node in nodes:
            field[node.x, node.y] = -1

    n = len(list(paths_dict.keys()))
    color_map = plt.cm.get_cmap('hsv', n)
    i = 0
    for agent_name, path in paths_dict.items():
        t_path = path[:t + 1]
        # for node in t_path:
        #     field[node.x, node.y] = 3
        if agent_name == 'agent_0':
            ax.scatter(t_path[-1].y, t_path[-1].x, s=200, c='white')
            ax.scatter(t_path[-1].y, t_path[-1].x, s=100, c='k')
        else:
            ax.scatter(t_path[-1].y, t_path[-1].x, s=100, c='k')
            ax.scatter(t_path[-1].y, t_path[-1].x, s=50, c=np.array([color_map(i)]))
        # ax.text(t_path[-1].y - 0.4, t_path[-1].x - 0.4, agent_name[6:])
        i += 1

    # for agent_name, path in paths_dict.items():
    #     # field[path[0].x, path[0].y] = 4
    #     field[path[-1].x, path[-1].y] = 5

    ax.imshow(field, origin='lower')
    ax.set_title(f'Paths in <{img_dir}> (time: {t})')


def plot_success_rate(ax, info):
    ax.cla()
    statistics_dict = info['statistics_dict']
    runs_per_n_agents = info['runs_per_n_agents']
    algs_to_test_dict = info['algs_to_test_dict']
    n_agents_list = info['n_agents_list']
    is_json = info['is_json']

    index = -1
    for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
        index += 1
        marker = f"-{get_line_or_marker(index, 'm')}"
        # success_rate
        sr_x = []
        sr_y = []
        for n_agents in n_agents_list:
            sr_list = get_list_n_run(statistics_dict, alg_name, n_agents, 'success_rate', runs_per_n_agents,
                                     is_json)
            if len(sr_list) > 0:
                sr_x.append(n_agents)
                sr_y.append(sum(sr_list) / len(sr_list))
        if 'color' in alg_info:
            ax.plot(sr_x, sr_y, marker, label=f'{alg_name}', alpha=0.9, color=alg_info['color'])
        else:
            ax.plot(sr_x, sr_y, marker, label=f'{alg_name}', alpha=0.9)

    set_plot_title(ax, 'success rate')
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim([0, 1.5])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents', labelpad=-1)
    set_legend(ax)


def plot_sol_quality(ax, info):
    ax.cla()
    statistics_dict = info['statistics_dict']
    runs_per_n_agents = info['runs_per_n_agents']
    algs_to_test_dict = info['algs_to_test_dict']
    n_agents_list = info['n_agents_list']
    is_json = info['is_json']

    index = -1
    for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
        index += 1
        # line = get_line_or_marker(index, 'l')
        marker = f"-{get_line_or_marker(index, 'm')}"
        # sol_quality
        sq_x = []
        sq_y = []
        for n_agents in n_agents_list:
            sq_list = get_list_sol_q_style(statistics_dict, alg_name, n_agents, 'sol_quality', runs_per_n_agents,
                                           list(algs_to_test_dict.keys()), is_json)
            if len(sq_list) > 0:
                sq_x.append(n_agents)
                sq_y.append(np.mean(sq_list))
        if 'color' in alg_info:
            ax.plot(sq_x, sq_y, marker, label=f'{alg_name}', alpha=0.8, color=alg_info['color'])
        else:
            ax.plot(sq_x, sq_y, marker, label=f'{alg_name}', alpha=0.8)

    set_plot_title(ax, 'solution quality')
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents', labelpad=-1)
    # ax.set_ylabel('sol_quality')
    set_legend(ax)


def plot_runtime_cactus(ax, info):
    ax.cla()
    statistics_dict = info['statistics_dict']
    runs_per_n_agents = info['runs_per_n_agents']
    algs_to_test_dict = info['algs_to_test_dict']
    n_agents_list = info['n_agents_list']
    is_json = info['is_json']

    max_instances = 0
    index = -1
    for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
        index += 1
        rt_y = get_list_runtime(statistics_dict, alg_name, n_agents_list, 'runtime', runs_per_n_agents, is_json)
        rt_y.sort()
        rt_x = list(range(len(rt_y)))
        max_instances = max(max_instances, len(rt_x))

        # dist_runtime
        if alg_info['dist']:
            it_y = get_list_runtime(statistics_dict, alg_name, n_agents_list, 'dist_runtime', runs_per_n_agents,
                                    is_json)
            it_y.sort()
            it_x = list(range(len(it_y)))
            max_instances = max(max_instances, len(it_x))
            sub_plot_cactus_dist_lines(ax, index, it_x, it_y, alg_name, alg_info)
        else:
            sub_plot_cactus_big_lines(ax, index, rt_x, rt_y, alg_name, alg_info)


    ax.set_xlim([0, max_instances + 2])
    # ax.set_xticks(rt_x)
    ax.set_xlabel('solved instances', labelpad=-1)
    ax.set_ylabel('seconds', labelpad=-1)
    is_log = set_log(ax)
    set_plot_title(ax, f'runtime (cactus{" - log scale" if is_log else ""})')
    # ax.set_ylim([1, 3000])
    # ax.set_ylim([0, 3000])
    set_legend(ax)


def plot_a_star_calls_counters(ax, info):
    ax.cla()
    statistics_dict = info['statistics_dict']
    runs_per_n_agents = info['runs_per_n_agents']
    algs_to_test_dict = info['algs_to_test_dict']
    n_agents_list = info['n_agents_list']
    is_json = info['is_json']

    max_instances = 0
    index = -1
    for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
        index += 1
        # A* calls
        ac_y = get_list_runtime(statistics_dict, alg_name, n_agents_list, 'a_star_calls_counter', runs_per_n_agents,
                                is_json)
        ac_y.sort()
        ac_x = list(range(len(ac_y)))
        max_instances = max(max_instances, len(ac_x))

        if alg_info['dist']:
            # get_list_a_star(statistics_dict, alg_name, n_agents_list, list_type, is_json=False)
            acd_y = get_list_a_star(statistics_dict, alg_name, n_agents_list, 'a_star_calls_counter_dist', is_json)
            acd_y.sort()
            acd_x = list(range(len(acd_y)))
            max_instances = max(max_instances, len(acd_x))
            sub_plot_cactus_dist_lines(ax, index, acd_x, acd_y, alg_name, alg_info)
        else:
            sub_plot_cactus_big_lines(ax, index, ac_x, ac_y, alg_name, alg_info)


    ax.set_xlim([0, max_instances + 2])
    ax.set_xlabel('solved instances', labelpad=-1)
    is_log = set_log(ax)
    # ax.set_ylim([0, 3e7])
    set_plot_title(ax, f'# of A* calls (cactus{" - log scale" if is_log else ""})')
    set_legend(ax, framealpha=0)


def plot_n_messages(ax, info):
    # stats_dict[alg_name][n_agents]['n_messages_per_agent'].extend(alg_info['n_messages_per_agent'])
    ax.cla()
    statistics_dict = info['statistics_dict']
    runs_per_n_agents = info['runs_per_n_agents']
    algs_to_test_dict = info['algs_to_test_dict']
    n_agents_list = info['n_agents_list']
    is_json = info['is_json']

    showfliers = False
    # showfliers = True
    big_table = []
    counter = 0
    for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
        line_style = f"-{get_line_or_marker(counter, 'm')}"
        counter += 1
        if alg_info['dist']:
            x_list = []
            y_list = []
            y2_list = []
            for n_agents in n_agents_list:
                x_list.append(n_agents)
                n_agents_number = n_agents
                if is_json:
                    n_agents = str(n_agents)
                y_list.append(np.mean(statistics_dict[alg_name][n_agents]['n_messages']))
                y2_list.append(np.mean(statistics_dict[alg_name][n_agents]['m_per_step'])/n_agents_number)

            if len(y_list) > 0:
                if 'color' in alg_info:
                    # ax.plot(x_list, y_list, '-o', label=f'{alg_name}', alpha=0.75, color=alg_info['color'])
                    ax.plot(x_list, y2_list, line_style, label=f'{alg_name}', alpha=0.55, color=alg_info['color'])
                else:
                    ax.plot(x_list, y_list, '-o', label=f'{alg_name}', alpha=0.75)

    # ax.set_ylabel('n_messages_per_agent')
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents', labelpad=-1)
    set_plot_title(ax, f'# of messages per step')
    set_legend(ax)


def plot_n_steps_iters(ax, info):
    # n_steps, n_small_iters
    ax.cla()
    statistics_dict = info['statistics_dict']
    runs_per_n_agents = info['runs_per_n_agents']
    algs_to_test_dict = info['algs_to_test_dict']
    n_agents_list = info['n_agents_list']
    is_json = info['is_json']

    counter = 0
    for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
        line_style = f"-{get_line_or_marker(counter, 'm')}"
        counter += 1
        if alg_info['dist']:
            x_list = []
            steps_list = []
            iters_list = []
            for n_agents in n_agents_list:
                x_list.append(n_agents)
                if is_json:
                    n_agents = str(n_agents)
                steps_list.append(np.mean(statistics_dict[alg_name][n_agents]['n_steps']))
                iters_list.append(np.mean(statistics_dict[alg_name][n_agents]['n_small_iters']))

            if len(steps_list) > 0:
                if 'color' in alg_info:
                    ax.plot(x_list, steps_list, line_style, label=f'{alg_name}', alpha=0.75, color=alg_info['color'])  # steps
                    # ax.plot(x_list, iters_list, line_style, label=f'{alg_name}', alpha=0.55, color=alg_info['color'])  # iters
    set_plot_title(ax, f'# of steps')
    # set_plot_title(ax, f'# of iterations')
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents', labelpad=-1)
    set_legend(ax)


def plot_n_expanded_cactus(ax, info):
    ax.cla()
    statistics_dict = info['statistics_dict']
    runs_per_n_agents = info['runs_per_n_agents']
    algs_to_test_dict = info['algs_to_test_dict']
    n_agents_list = info['n_agents_list']
    is_json = info['is_json']

    max_instances = 0
    index = -1
    for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
        index += 1
        # runtime
        rt_y = get_list_a_star(statistics_dict, alg_name, n_agents_list, 'n_closed_per_run', is_json)
        rt_y.sort()
        rt_x = list(range(len(rt_y)))
        max_instances = max(max_instances, len(rt_x))

        if alg_info['dist']:
            l_y = get_list_a_star(statistics_dict, alg_name, n_agents_list, 'a_star_n_closed_dist', is_json)
            l_y.sort()
            l_x = list(range(len(l_y)))
            max_instances = max(max_instances, len(l_x))
            sub_plot_cactus_dist_lines(ax, index, l_x, l_y, alg_name, alg_info)
        else:
            sub_plot_cactus_big_lines(ax, index, rt_x, rt_y, alg_name, alg_info)

    ax.set_xlim([0, max_instances + 2])
    # ax.set_xticks(rt_x)
    # ax.set_ylabel('n_closed')
    ax.set_xlabel('solved instances', labelpad=-1)
    # ax.set_xlabel('y: N expanded nodes (cactus - log scale)')
    is_log = set_log(ax)
    # ax.set_ylim([0, 3e7])
    set_plot_title(ax, f'# of expanded nodes (cactus{" - log scale" if is_log else ""})')
    set_legend(ax)


def plot_n_agents_conf(ax, info):
    ax.cla()
    statistics_dict = info['statistics_dict']
    runs_per_n_agents = info['runs_per_n_agents']
    algs_to_test_dict = info['algs_to_test_dict']
    n_agents_list = info['n_agents_list']
    is_json = info['is_json']

    for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
        l_x = []
        l_y = []
        for n_agents in n_agents_list:
            if is_json:
                n_agents = str(n_agents)
            if 'n_agents_conf' in statistics_dict[alg_name][n_agents]:
                n_agents_conf = statistics_dict[alg_name][n_agents]['n_agents_conf']
                if len(n_agents_conf) > 0:
                    l_y.append(np.mean(statistics_dict[alg_name][n_agents]['n_agents_conf']))
                    l_x.append(n_agents)

        if len(l_y) > 0:
            if 'color' in alg_info:
                ax.plot(l_x, l_y, '-o', label=f'{alg_name}', alpha=0.75, color=alg_info['color'])
            else:
                ax.plot(l_x, l_y, '-o', label=f'{alg_name}', alpha=0.75)

    ax.set_ylabel('n_agents_conf')
    ax.set_xlim([min(n_agents_list) - 1, max(n_agents_list) + 1])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents')
    set_legend(ax)


def plot_a_star_calls_boxplot(ax, info):
    ax.cla()
    statistics_dict = info['statistics_dict']
    runs_per_n_agents = info['runs_per_n_agents']
    algs_to_test_dict = info['algs_to_test_dict']
    n_agents_list = info['n_agents_list']
    is_json = info['is_json']

    showfliers = False
    # showfliers = True
    big_table = []
    algs_names = []
    for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
        if alg_info['dist']:
            algs_names.append(alg_name)
            a_star_calls = []
            for n_agents in n_agents_list:
                if is_json:
                    n_agents = str(n_agents)
                a_star_calls.extend(statistics_dict[alg_name][n_agents]['a_star_calls_per_agent'])
            big_table.append(a_star_calls)
    ax.boxplot(big_table,
               # notch=True,  # notch shape
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               # labels=algs_to_test_list,  # linewidth=0.5, edgecolor="white"
               showfliers=showfliers)

    # ax.set_title('a_star_runtimes (boxplot)')
    # ax.set_xlim([0, max_instances + 2])
    # ax.set_xticks()
    ax.set_xticklabels(algs_names, rotation=15)
    # ax.set_ylabel(f'A * calls average per agent')
    # ax.set_xlabel(f'{"with" if showfliers else "no"} outliers')
    ax.set_xlabel(f'A * calls average per agent')
    # ax.legend()


def plot_conf_per_iter(ax, info, **kwargs):
    ax.cla()
    statistics_dict = info['statistics_dict']
    runs_per_n_agents = info['runs_per_n_agents']
    algs_to_test_dict = info['algs_to_test_dict']
    n_agents_list = info['n_agents_list']
    is_json = info['is_json']

    index = -1
    for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
        index += 1
        # line = get_line_or_marker(index, 'l')
        marker = f"-{get_line_or_marker(index, 'm')}"
        if alg_info['dist'] and 'n_agents' in kwargs:
            l_y = statistics_dict[alg_name][kwargs['n_agents']]['confs_per_iter']
            l_x = list(range(len(l_y)))

            if len(l_y) > 0:
                if 'color' in alg_info:
                    ax.plot(l_x, l_y, marker, label=f'{alg_name}', alpha=0.75, color=alg_info['color'])
                else:
                    ax.plot(l_x, l_y, marker, label=f'{alg_name}', alpha=0.75)

    # ax.set_ylabel('Conflicts per Iteration')
    # ax.set_xlim([min_x - 1, max_x + 1])
    # ax.set_xticks(n_agents_list)
    ax.set_xlabel('Conflicts per Iteration')
    set_legend(ax)


def plot_n_nei(ax, info):
    # n_steps, n_small_iters
    ax.cla()
    statistics_dict = info['statistics_dict']
    runs_per_n_agents = info['runs_per_n_agents']
    algs_to_test_dict = info['algs_to_test_dict']
    n_agents_list = info['n_agents_list']
    is_json = info['is_json']

    counter = 0
    for alg_name, (alg_func, alg_info) in algs_to_test_dict.items():
        line_style = f"-{get_line_or_marker(counter, 'm')}"
        counter += 1
        if alg_info['dist']:
            x_list = []
            nei_list = []
            for n_agents in n_agents_list:
                x_list.append(n_agents)
                n_agents_number = n_agents
                if is_json:
                    n_agents = str(n_agents)
                nei_list.append(np.mean(statistics_dict[alg_name][n_agents]['n_nei'])/n_agents_number)

            if len(nei_list) > 0:
                if 'color' in alg_info:
                    ax.plot(x_list, nei_list, line_style, label=f'{alg_name}', alpha=0.55, color=alg_info['color'])

    # ax.set_ylabel('sum of neighbours', labelpad=-1)
    set_plot_title(ax, f'# of neighbours per agent')
    ax.set_xlim([min(n_agents_list) - 20, max(n_agents_list) + 20])
    ax.set_xticks(n_agents_list)
    ax.set_xlabel('N agents', labelpad=-1)
    set_legend(ax)
