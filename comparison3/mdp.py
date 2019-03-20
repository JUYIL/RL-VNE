import gym
from gym import spaces
import copy
import networkx as nx
import numpy as np


def calculate_adjacent_bw(graph, u, kind='bw'):
    """计算一个节点的相邻链路带宽和，默认为总带宽和，若计算剩余带宽资源和，需指定kind属性为bw-remain"""

    bw_sum = 0
    for v in graph.neighbors(u):
        bw_sum += graph[u][v][kind]
    return bw_sum


class Env(gym.Env):

    def render(self, mode='human'):
        pass

    def __init__(self, sub):
        self.count = -1
        self.n_action = sub.number_of_nodes()
        self.sub = copy.deepcopy(sub)
        self.action_space = spaces.Discrete(self.n_action)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_action, 4), dtype=np.float32)
        self.state = None
        self.actions = []
        self.degree = []
        for i in nx.degree_centrality(sub).values():
            self.degree.append(i)
        self.vnr = None

    def set_sub(self, sub):
        self.sub = copy.deepcopy(sub)

    def set_vnr(self, vnr):
        self.vnr = vnr

    def step(self, action):
        self.actions.append(action)
        self.count = self.count + 1
        cpu_remain, bw_all_remain, avg_dst = [], [], []
        for u in range(self.n_action):
            adjacent_bw = calculate_adjacent_bw(self.sub, u, 'bw_remain')
            if u == action:
                self.sub.nodes[action]['cpu_remain'] -= self.vnr.nodes[self.count]['cpu']
                adjacent_bw -= calculate_adjacent_bw(self.vnr, self.count)
            cpu_remain.append(self.sub.nodes[u]['cpu_remain'])
            bw_all_remain.append(adjacent_bw)

            sum_dst = 0
            for v in self.actions:
                sum_dst += nx.shortest_path_length(self.sub, source=u, target=v)
            sum_dst /= (len(self.actions) + 1)
            avg_dst.append(sum_dst)

        cpu_remain = (cpu_remain - np.min(cpu_remain)) / (np.max(cpu_remain) - np.min(cpu_remain))
        bw_all_remain = (bw_all_remain - np.min(bw_all_remain)) / (np.max(bw_all_remain) - np.min(bw_all_remain))
        avg_dst = (avg_dst - np.min(avg_dst)) / (np.max(avg_dst)-np.min(avg_dst))

        self.state = (cpu_remain,
                      bw_all_remain,
                      self.degree,
                      avg_dst)
        return np.vstack(self.state).transpose(), 0.0, False, {}

    def reset(self):
        """获得底层网络当前最新的状态"""
        self.count = -1
        self.actions = []
        cpu_remain, bw_all_remain = [], []
        for u in range(self.n_action):
            cpu_remain.append(self.sub.nodes[u]['cpu_remain'])
            bw_all_remain.append(calculate_adjacent_bw(self.sub, u, 'bw_remain'))

        cpu_remain = (cpu_remain - np.min(cpu_remain)) / (np.max(cpu_remain) - np.min(cpu_remain))
        bw_all_remain = (bw_all_remain - np.min(bw_all_remain)) / (np.max(bw_all_remain) - np.min(bw_all_remain))
        avg_dst = np.zeros(self.n_action).tolist()
        self.state = (cpu_remain,
                      bw_all_remain,
                      self.degree,
                      avg_dst)
        return np.vstack(self.state).transpose()
