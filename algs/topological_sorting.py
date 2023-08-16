def visit(node, sorted_list, nodes_without_perm_mark, nodes_with_perm_mark, nodes_with_temp_mark, sorting_rules):
    print('\rFUNC: visit', end='')
    if node in nodes_with_perm_mark:
        return True
    if node in nodes_with_temp_mark:
        return False  # not a DAG

    nodes_with_temp_mark.append(node)

    for h_node, l_node in sorting_rules:
        if h_node == node:
            is_dag = visit(l_node, sorted_list, nodes_without_perm_mark, nodes_with_perm_mark, nodes_with_temp_mark, sorting_rules)
            if not is_dag:
                return False
    nodes_with_temp_mark.remove(node)
    nodes_with_perm_mark.append(node)
    sorted_list.insert(0, node)
    return True


def topological_sorting(nodes: list, sorting_rules: list):
    print('\rFUNC: topological_sorting', end='')
    sorted_list = []
    nodes_without_perm_mark = nodes
    nodes_with_perm_mark = []
    nodes_with_temp_mark = []

    while len(nodes_without_perm_mark) > 0:
        unmarked_node = nodes_without_perm_mark.pop()
        is_dag = visit(unmarked_node, sorted_list, nodes_without_perm_mark, nodes_with_perm_mark, nodes_with_temp_mark, sorting_rules)
        if not is_dag:
            return None

    return sorted_list

