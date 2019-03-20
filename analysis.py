import os
import networkx as nx
import matplotlib.pyplot as plt


class Analysis:

    def __init__(self):
        self.result_dir = 'results/'
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.algorithms = ['grc-VNE', 'mcts-VNE', 'rl-VNE', 'ml-VNE-single']
        self.line_types = ['b:', 'r--', 'y-.', 'g-']
        self.metric_names = {'acceptance ratio': 'Acceptance Ratio',
                             'average revenue': 'Long Term Average Revenue',
                             'average cost': 'Long Term Average Cost',
                             'R_C': 'Long Term Revenue/Cost Ratio',
                             'node utilization': 'Average Node Utilization',
                             'link utilization': 'Average Link Utilization'}

    def save_result(self, sub, filename):
        """将一段时间内底层网络的性能指标输出到指定文件内"""

        filename = self.result_dir + filename
        with open(filename, 'w') as f:
            for time, evaluation in sub.evaluation.metrics.items():
                f.write("%-10s\t" % time)
                f.write("%-20s\t%-20s\t%-20s\t%-20s\t%-20s\t%-20s\n" % evaluation)

    def read_result(self, filename):
        """读取结果文件"""

        with open(self.result_dir + filename) as f:
            lines = f.readlines()

        t, acceptance, revenue, cost, r_to_c, node_stress, link_stress = [], [], [], [], [], [], []
        count = 0
        for line in lines:
            count = count + 1
            a, b, c, d, e, f, g = [float(x) for x in line.split()]
            t.append(a)
            acceptance.append(b)
            revenue.append(c / a)
            cost.append(d / a)
            r_to_c.append(e)
            node_stress.append(f)
            link_stress.append(g)

        return t, acceptance, revenue, cost, r_to_c, node_stress, link_stress

    def draw_result(self):
        """绘制实验结果图"""

        results = []
        for alg in self.algorithms:
            results.append(self.read_result(self.result_dir + alg + '-new.txt'))

        index = 0
        for metric, title in self.metric_names.items():
            index += 1
            plt.figure()
            for alg_id in range(len(self.algorithms)):
                x = results[alg_id][0]
                y = results[alg_id][index]
                plt.plot(x, y, self.line_types[alg_id], label=self.algorithms[alg_id])
            plt.xlim([0, 50000])
            if metric == 'acceptance ratio' or metric == 'node utilization' or metric == 'link utilization':
                plt.ylim([0, 1])
            plt.xlabel("time", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.title(title, fontsize=15)
            plt.legend(loc='best', fontsize=12)
            # plt.savefig(self.result_dir + metric + '.png')
        plt.show()

    def draw_topology(self, graph, filename):
        """绘制拓扑图"""

        nx.draw(graph, with_labels=False, node_color='black', edge_color='gray', node_size=50)
        plt.savefig(self.result_dir + filename + '.png')
        plt.close()


if __name__ == '__main__':
    analysis = Analysis()
    analysis.draw_result()
