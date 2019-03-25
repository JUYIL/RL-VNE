import os
import networkx as nx
import matplotlib.pyplot as plt


class Analysis:

    def __init__(self):
        self.result_dir = 'results/'
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.algorithms = ['grc-ts', 'mcts-ts', 'rl-ts',  'RIM-ts', 'RLN-ts','RLNL-ts']
        # self.algorithms = ['rl-ts', 'RLN-t5-ts', 'RLN-t6-ts', 'RLN-t7-ts']
        self.line_types = ['b:', 'r--', 'y-.', 'g-','c:','m:']
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

        if filename == "RIM-ts.txt":
            with open('results/RIM-ts.dat') as file_object:
                lines = file_object.readlines()
            t, acceptance, revenue, cost, r_to_c, nodeuti, linkuti = [], [], [], [], [], [], []
            for line in lines[1:]:
                time, _, _, ar, _, _, ac, _, anut, _, alut, acp, _, _, rc = [float(x) for x in line.split('	')]
                t.append(time)
                acceptance.append(acp)
                revenue.append(ar)
                cost.append(ac)
                r_to_c.append(rc)
                nodeuti.append(anut)
                linkuti.append(alut)
        else:
            with open(self.result_dir + filename) as f:
                lines = f.readlines()

            t, acceptance, revenue, cost, r_to_c, nodeuti, linkuti = [], [], [], [], [], [], []
            for line in lines:
                a, b, c, d, e, f, g = [float(x) for x in line.split()]
                t.append(a)
                acceptance.append(b)
                revenue.append(c / a)
                cost.append(d / a)
                r_to_c.append(e)
                nodeuti.append(f)
                linkuti.append(g)

        return t, acceptance, revenue, cost, r_to_c, nodeuti, linkuti


    def draw_result(self):
        """绘制实验结果图"""

        results = []
        for alg in self.algorithms:
            results.append(self.read_result( alg + '.txt'))

        index = 0
        for metric, title in self.metric_names.items():
            index += 1
            plt.figure()
            for alg_id in range(len(self.algorithms)):
                x = results[alg_id][0]
                y = results[alg_id][index]
                plt.plot(x, y, self.line_types[alg_id], label=self.algorithms[alg_id])
            plt.xlim([0, 50000])
            if metric == 'acceptance ratio' :
                # or metric == 'node utilization' or metric == 'link utilization':
                plt.ylim([0, 1])
            plt.xlabel("time", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.title(title, fontsize=15)
            plt.legend(loc='best', fontsize=12)
            plt.savefig(self.result_dir + metric + '.png')
        plt.show()

    def draw_topology(self, graph, filename):
        """绘制拓扑图"""

        nx.draw(graph, with_labels=False, node_color='black', edge_color='gray', node_size=50)
        plt.savefig(self.result_dir + filename + '.png')
        plt.close()


if __name__ == '__main__':
    analysis = Analysis()
    analysis.draw_result()
