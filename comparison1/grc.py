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
        for v_node in vnr_grc_vector:
            v_id = v_node[0]
            for s_node in sub_grc_vector:
                s_id = s_node[0]
                if s_id not in node_map.values() and \
                        sub.net.nodes[s_id]['cpu_remain'] >= vnr.nodes[v_id]['cpu']:
                    node_map.update({v_id: s_id})
        return node_map

    def run_run(self, sub, upper_req, infrastructure):
        node_map = {}
        sub_grc_vector = self.calculate_grc(sub.net)
        vnr_grc_vector = self.calculate_grc(upper_req, category='vnr')

        for v_node in vnr_grc_vector:
            v_id = v_node[0]
            candidates = []
            self.no_solution = True
            for s_node in sub_grc_vector:
                s_id = s_node[0]
                if s_id not in node_map.values():
                    candidates.append(s_id)
                    if sub.net.nodes[s_id]['cpu_remain'] >= upper_req.nodes[v_id]['cpu']:
                        node_map.update({v_id: s_id})
                        self.no_solution = False
                        break

            # if self.no_solution:
            #
            #     # 调整操作是否成功
            #     self.adjust_success = False
            #     # 迁移后所有相邻链路是否重映射成功
            #     self.remap_success = False
            #     # 整体迁移是否成功
            #     self.migrate_success = False
            #
            #     rand_id = candidates[random.randint(0, len(candidates) - 1)]
            #     print(rand_id)
            #
            #     extra = upper_req.nodes[v_id]['cpu'] - sub.net.nodes[rand_id]['cpu']
            #     # 找到接受该下层虚拟节点映射的底层物理节点
            #     physical_node = infrastructure.mapped_info[sub.net.graph['id']][0][rand_id]
            #
            #     # 第1种情形
            #     if infrastructure.net.nodes[physical_node]['cpu_remain'] >= extra:
            #         # 为下层虚拟节点分配增加的资源
            #         infrastructure.net.nodes[physical_node]['cpu_remain'] -= extra
            #         # 更新下层虚拟网络请求的资源
            #         sub.net.nodes[rand_id]['cpu'] = upper_req.nodes[v_id]['cpu']
            #         self.adjust_success = True
            #
            #     # 第2种情形
            #     else:
            #
            #         # 迁移操作
            #         for new_one in infrastructure.net.neighbors(physical_node):
            #             tmp = copy.deepcopy(infrastructure)
            #             if tmp.net.nodes[new_one]['cpu_remain'] >= upper_req.nodes[v_id]['cpu']:
            #
            #                 # 更新请求的资源
            #                 sub.net.nodes[rand_id]['cpu'] = upper_req.nodes[v_id]['cpu']
            #                 # 更新映射的位置
            #                 tmp.mapped_info[sub.net.graph['id']][0][rand_id] = new_one
            #                 # 释放原资源
            #                 tmp.net.nodes[physical_node]['cpu_remain'] += sub.net.nodes[rand_id]['cpu']
            #                 # 占用新资源
            #                 tmp.net.nodes[new_one]['cpu_remain'] -= sub.net.nodes[rand_id]['cpu']
            #
            #                 links = []
            #                 for under_link in sub.net.edges:
            #                     if under_link[0] == rand_id or under_link[1] == rand_id:
            #                         links.append(under_link)
            #
            #                 # 所有相邻链路的调整
            #                 for under_link in links:
            #                     if under_link[0] == rand_id:
            #                         neighbor = under_link[1]
            #                     else:
            #                         neighbor = under_link[0]
            #                     link_resource = sub.net[under_link[0]][under_link[1]]['bw']
            #
            #                     # 释放资源
            #                     path = tmp.mapped_info[sub.net.graph['id']][1][under_link]
            #                     start = path[0]
            #                     for end in path[1:]:
            #                         tmp.net[start][end]['bw_remain'] += link_resource
            #                         start = end
            #
            #                     # 重映射
            #                     if nx.has_path(tmp.net, source=new_one,
            #                                        target=tmp.mapped_info[sub.net.graph['id']][0][neighbor]):
            #                         for path in k_shortest_path(tmp.net, new_one, tmp.mapped_info[sub.net.graph['id']][0][neighbor]):
            #                             if tmp.get_path_capacity(path) >= sub.net[rand_id][neighbor]['bw']:
            #                                 # 更新该上层虚拟链路的映射
            #                                 tmp.mapped_info[sub.net.graph['id']][1][under_link] = path
            #                                 # 分配相应的资源
            #                                 start = path[0]
            #                                 for end in path[1:]:
            #                                     tmp.net[start][end]['bw_remain'] -= link_resource
            #                                     start = end
            #                                 self.remap_success = True
            #                                 break
            #                             else:
            #                                 continue
            #
            #                     # 如果有相邻链路重映射失败，那么就需要尝试其他的可迁移节点
            #                     if not self.remap_success:
            #                         break
            #
            #                 if self.remap_success:
            #                     self.migrate_success = True
            #                     infrastructure = tmp
            #                     break
            #
            #             if self.migrate_success:
            #                 self.adjust_success = True
            #                 break
            #
            #     if self.adjust_success:
            #         node_map.update({v_id: rand_id})
            #         continue
            #     else:
            #         break

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
