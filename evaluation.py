class Evaluation:
    def __init__(self, graph):
        self.graph = graph
        # 到达的虚拟网络请求数
        self.total_arrived = 0
        # 成功接受的虚拟网络请求数
        self.total_accepted = 0
        # 总收益
        self.total_revenue = 0
        # 总成本
        self.total_cost = 0
        # 平均节点利用率
        self.average_node_stress = 0
        # 平均链路利用率
        self.average_link_stress = 0
        # 每个时刻对应的性能指标元组（请求接受率、平均收益、平均成本、收益成本比、平均节点利用率、平均链路利用率）
        self.metrics = {}

    def add(self, req, link_map):
        """增加对应的评估指标值"""
        self.total_accepted += 1
        self.total_revenue += self.calcualte_revenue(req)
        self.total_cost += self.calculate_cost(req, link_map)
        self.average_node_stress += self.calculate_ans()
        self.average_link_stress += self.calculate_als()
        self.metrics.update({req.graph['time']: (self.total_accepted / self.total_arrived,
                                                 self.total_revenue,
                                                 self.total_cost,
                                                 self.total_revenue / self.total_cost,
                                                 self.average_node_stress / self.total_arrived,
                                                 self.average_link_stress / self.total_arrived)})

    def calcualte_revenue(self, req):
        """"映射收益"""
        revenue = 0
        for vn in range(req.number_of_nodes()):
            revenue += req.nodes[vn]['cpu']
        for vl in req.edges:
            revenue += req[vl[0]][vl[1]]['bw']
        return revenue

    def calculate_cost(self, req, link_map):
        """映射成本"""
        cost = 0
        for vn in range(req.number_of_nodes()):
            cost += req.nodes[vn]['cpu']

        for vl, path in link_map.items():
            link_resource = req[vl[0]][vl[1]]['bw']
            cost += link_resource * (len(path) - 1)
        return cost

    def calculate_ans(self):
        """节点资源利用率"""
        node_stress = 0
        for i in range(self.graph.number_of_nodes()):
            node_stress += 1 - self.graph.nodes[i]['cpu_remain'] / self.graph.nodes[i]['cpu']
        node_stress /= self.graph.number_of_nodes()
        return node_stress

    def calculate_als(self):
        """链路资源利用率"""
        link_stress = 0
        for vl in self.graph.edges:
            link_stress += 1 - self.graph[vl[0]][vl[1]]['bw_remain'] / self.graph[vl[0]][vl[1]]['bw']
        link_stress /= self.graph.number_of_edges()
        return link_stress
