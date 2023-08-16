import random
import heapq


class Node:
    def __init__(self, x, y, t=0, neighbours=None, new_ID=None):
        if new_ID:
            self.ID = new_ID
        else:
            self.ID = f'{x}_{y}_{t}'
        self.xy_name = f'{x}_{y}'
        self.x = x
        self.y = y
        self.t = t
        if neighbours is None:
            self.neighbours = []
        else:
            self.neighbours = neighbours
        # self.neighbours = neighbours

        self.h = 0
        self.g = t
        self.parent = None
        self.g_dict = {}

    def f(self):
        return self.t + self.h
        # return self.g + self.h

    def reset(self, target_nodes=None, **kwargs):
        if 'start_time' in kwargs:
            self.t = kwargs['start_time']
        else:
            self.t = 0
        self.h = 0
        self.g = self.t
        self.ID = f'{self.x}_{self.y}_{self.t}'
        self.parent = None
        if target_nodes is not None:
            self.g_dict = {target_node.xy_name: 0 for target_node in target_nodes}
        else:
            self.g_dict = {}


class ListNodes:
    def __init__(self, target_name=None):
        self.heap_list = []
        # self.nodes_list = []
        self.dict = {}
        self.h_func_bool = False
        if target_name:
            self.h_func_bool = True
            self.target_name = target_name

    def __len__(self):
        return len(self.heap_list)

    def remove(self, node):
        if self.h_func_bool:
            self.heap_list.remove((node.g_dict[self.target_name], node.xy_name))
            del self.dict[node.xy_name]
        else:
            if node.ID not in self.dict:
                raise RuntimeError('node.ID not in self.dict')
            self.heap_list.remove(((node.f(), node.h), node.ID))
            del self.dict[node.ID]
        # self.nodes_list.remove(node)

    def add(self, node):
        if self.h_func_bool:
            heapq.heappush(self.heap_list, (node.g_dict[self.target_name], node.xy_name))
            self.dict[node.xy_name] = node
        else:
            heapq.heappush(self.heap_list, ((node.f(), node.h), node.ID))
            self.dict[node.ID] = node
        # self.nodes_list.append(node)

    def pop(self):
        heap_tuple = heapq.heappop(self.heap_list)
        node = self.dict[heap_tuple[1]]
        if self.h_func_bool:
            del self.dict[node.xy_name]
        else:
            del self.dict[node.ID]
        # self.nodes_list.remove(node)
        return node

    def get(self, ID):
        return self.dict[ID]

    def get_nodes_list(self):
        return [self.dict[item[1]] for item in self.heap_list]


# def decision_rule(self, other):
#     # return bool(random.getrandbits(1))
#     return self.x > other.x or self.y > other.y
#
# def __lt__(self, other):
#     return self.decision_rule(other)
#
# def __le__(self, other):
#     return self.decision_rule(other)
#
# def __eq__(self, other):
#     return self.decision_rule(other)
#
# def __ne__(self, other):
#     return self.decision_rule(other)
#
# def __gt__(self, other):
#     return self.decision_rule(other)
#
# def __ge__(self, other):
#     return self.decision_rule(other)




