import numpy as np
import copy
import networkx as nx
import random
from itertools import islice


def calculate_adjacent_bw(graph, u, kind='bw'):
    """计算一个节点的相邻链路带宽和，默认为总带宽和，若计算剩余带宽资源和，需指定kind属性为bw-remain"""

    bw_sum = 0
    for v in graph.neighbors(u):
        bw_sum += graph[u][v][kind]
    return bw_sum


# k最短路径
def k_shortest_path(G, source, target, k=5):
    return list(islice(nx.shortest_simple_paths(G, source, target), k))


class GRC:
    def __init__(self, damping_factor, sigma):
        self.damping_factor = damping_factor
        self.sigma = sigma
        # 调整操作是否成功
        self.adjust_success = False
        # 迁移后所有相邻链路是否重映射成功
        self.remap_success = False
        self.migrate_success = False
        self.no_solution = False

    def run(self, sub, vnr):
        node_map = {}
        sub_grc_vector = self.calculate_grc(sub.net)
        vnr_grc_vector = self.calculate_grc(vnr, category='vnr')
        sub_copy=copy.deepcopy(sub.net)
        for v_node in vnr_grc_vector:
            v_id = v_node[0]
            for s_node in sub_grc_vector:
                s_id = s_node[0]
                if s_id not in node_map.values() and \
                        sub.net.nodes[s_id]['cpu_remain'] > vnr.nodes[v_id]['cpu']:
                    node_map.update({v_id: s_id})
                    tmp=sub_copy.nodes[s_id]['cpu_remain']-vnr.nodes[v_id]['cpu']
                    sub_copy.nodes[s_id]['cpu_remain']=round(tmp,6)
                    break
        return node_map


    def calculate_grc(self, graph, category='substrate'):
        """calculate grc vector of a substrate network or a virtual network"""

        if category == 'vnr':
            cpu_type = 'cpu'
            bw_type = 'bw'
        else:
            cpu_type = 'cpu_remain'
            bw_type = 'bw_remain'

        cpu_vector, m_matrix = [], []
        n = graph.number_of_nodes()
        for u in range(n):
            cpu_vector.append(graph.nodes[u][cpu_type])
            sum_bw = calculate_adjacent_bw(graph, u, bw_type)
            for v in range(n):
                if v in graph.neighbors(u):
                    m_matrix.append(graph[u][v][bw_type] / sum_bw)
                else:
                    m_matrix.append(0)
        cpu_vector = np.array(cpu_vector) / np.sum(cpu_vector)
        m_matrix = np.array(m_matrix).reshape(n, n)
        cpu_vector *= (1 - self.damping_factor)
        current_r = cpu_vector
        delta = float('inf')
        while delta >= self.sigma:
            mr_vector = np.dot(m_matrix, current_r)
            mr_vector *= self.damping_factor
            next_r = cpu_vector + mr_vector
            delta = np.linalg.norm(next_r - current_r, ord=2)
            current_r = next_r

        output = []
        for i in range(n):
            output.append((i, current_r[i]))
        output.sort(key=lambda element: element[1], reverse=True)
        return output
