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

def minbw(sub, path):
    """找到一条路径中带宽资源最小的链路并返回其带宽资源值"""

    bandwidth = 1000
    head = path[0]
    for tail in path[1:]:
        if sub[head][tail]['bw_remain'] <= bandwidth:
            bandwidth = sub[head][tail]['bw_remain']
        head = tail
    return bandwidth

def getbtns():
    btns=[]
    for i in range(1,7):
        path = 'Mine/btns/btn%s.txt' % i
        with open(path) as file_object:
            lines = file_object.readlines()
        for line in lines:
            btns.append(float(line))
    return btns

def getallpath(sub):
    i = 0
    k=0
    linkaction = {}
    while i < 100:
        j = 0
        while j < 100:
            path = nx.shortest_simple_paths(sub, i, j)
            for p in path:
                if 1 < len(p) < 6:
                    linkaction.update({k:{(i,j):p}})
                    k+=1
                else:
                    break
            j += 1
        i += 1

    return linkaction

btns=getbtns()


class LinkEnv(gym.Env):

    def render(self, mode='human'):
        pass

    def __init__(self, sub):
        self.count = -1
        self.linkpath = getallpath(sub)
        self.n_action = len(self.linkpath)
        self.sub = copy.deepcopy(sub)
        self.action_space = spaces.Discrete(self.n_action)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_action, 2), dtype=np.float32)
        self.state = None
        self.vnr = None
        btn = btns
        self.btn = []
        self.btn = (btn - np.min(btn)) / (np.max(btn) - np.min(btn))

    def set_sub(self, sub):
        self.sub = copy.deepcopy(sub)

    def set_vnr(self, vnr):
        self.vnr = vnr

    def set_link(self,link):
        self.link=link

    def step(self, action):
        mbw_remain = []
        thepath = list(self.linkpath[action].values())[0]

        i = 0
        while i < len(thepath) - 1:
            fr = thepath[i]
            to = thepath[i + 1]
            self.sub[fr][to]['bw_remain'] -= self.vnr[self.link[0]][self.link[1]]['bw']
            i += 1

        for paths in self.linkpath.values():
            path = list(paths.values())[0]
            mbw_remain.append(minbw(self.sub, path))
        mbw_remain = (mbw_remain - np.min(mbw_remain)) / (
                np.max(mbw_remain) - np.min(mbw_remain))

        self.state = (mbw_remain, self.btn)

        return np.vstack(self.state).transpose(), 0.0, False, {}

    def reset(self):
        """获得底层网络当前最新的状态"""
        self.count = -1
        self.actions = []
        mbw = []
        for paths in self.linkpath.values():
            path = list(paths.values())[0]
            mbw.append(minbw(self.sub, path))

        # normalization
        mbw = (mbw - np.min(mbw)) / (np.max(mbw) - np.min(mbw))
        mbw_remain = mbw

        self.state = (mbw_remain, self.btn)


        return np.vstack(self.state).transpose()
