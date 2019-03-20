import math
import random
import os
import copy
import numpy as np
import networkx as nx

# 仿真时间
TOTAL_TIME = 50000

# 网络节点坐标范围
SCALE = 100

# 仅与虚拟网络请求相关的参数
DURATION_MEAN = 1000
MIN_DURATION = 250
MAX_DISTANCE = 20

# 生成文件的存放目录
spec_dir = 'generated/spec/'
alt_dir = 'generated/alt/'
save_path = 'networks-tmp/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


def calculate_dis(coordinate1, coordinate2):
    """给定两个节点坐标，求解它们之间的欧氏距离"""
    return math.sqrt(pow(coordinate1[0] - coordinate2[0], 2) + pow(coordinate1[1] - coordinate2[1], 2))


def generate_network(network_name, node_num, min_res, max_res, time=0, duration=0, transit_nodes=0):
    """生成网络文件"""

    # # Step1: 执行itm指令生成.gb文件
    # spec_filename = 'spec-%s' % network_name
    # cmd = "%s %s" % (itm, spec_dir + spec_filename)
    # os.system(cmd)
    #
    # # Step2: 执行sgb2alt指令将刚刚生成的gb文件转换为alt文件
    # gb_filename = spec_filename + '-0.gb'
    # alt_filename = '%s.alt' % network_name
    # cmd = "%s %s %s" % (sgb2alt, spec_dir + gb_filename, alt_dir + alt_filename)
    # os.system(cmd)

    # Step3: 读取alt文件
    if network_name == 'sub-ts':
        alt_filename = 'ts100.alt'
    else:
        alt_filename = '%s.alt' % node_num

    with open(alt_dir + alt_filename) as f:
        lines = f.readlines()

    # Step4: 生成网络文件
    print("generate %s" % network_name)
    network_filename = '%s.txt' % network_name
    with open(save_path + network_filename, 'w') as network_file:

        coordinates = []

        # Step4-1: 写入网络整体信息
        edge_num = len(lines)-node_num-6
        if network_name == 'sub-wm' or network_name == 'sub-ts':
            # 物理网络信息包括：节点数量、链路数量
            network_file.write("%d %d\n" % (node_num, edge_num))
        else:
            # 虚拟网络信息包括：节点数量、链路数量、到达时间、持续时间、可映射范围
            network_file.write("%d %d %d %d %d\n" % (node_num, edge_num, time, duration, MAX_DISTANCE))

        # Step4-2: 依次写入节点信息（x坐标，y坐标，节点资源）
        for line in lines[4:4 + node_num]:
            blocks = line.split()
            x = int(blocks[2])
            y = int(blocks[3])
            coordinates.append((x, y))
            resource = random.uniform(min_res, max_res)

            # 属于transit-stub模型网络的特殊操作
            if network_name == 'sub-ts' and len(coordinates) <= transit_nodes:
                network_file.write("%d %d %f\n" % (x, y, 100 + resource))
                continue

            network_file.write("%d %d %f\n" % (x, y, resource))

        # Step4-3: 依次写入链路信息（起始节点，终止节点，带宽资源，时延）
        for line in lines[6 + node_num:]:
            from_id, to_id, length, a = [int(x) for x in line.split()]
            distance = calculate_dis(coordinates[from_id], coordinates[to_id])
            resource = random.uniform(min_res, max_res)

            # 属于transit-stub模型网络的特殊操作
            if network_name == 'sub-ts':
                if from_id < transit_nodes and to_id < transit_nodes:
                    network_file.write("%d %d %f %f\n" % (from_id, to_id, 200 + resource, distance))
                    continue
                if from_id < transit_nodes or to_id < transit_nodes:
                    network_file.write("%d %d %f %f\n" % (from_id, to_id, 100 + resource, distance))
                    continue

            network_file.write("%d %d %f %f\n" % (from_id, to_id, resource, distance))


def make_sub_wm(node_num, min_res, max_res):
    """生成物理网络文件（基于waxman随机型网络模型）"""

    network_name = 'sub-wm'

    # # 生成GT-ITM配置文件
    # spec_filename = 'spec-%s' % network_name
    # with open(spec_dir + spec_filename, 'w') as f:
    #     f.write("geo 1\n")
    #     f.write("%d %d 2 %f 0.2\n" % (node_num, SCALE, connect_prob))

    # 生成基于Waxman随机模型的物理网络文件
    generate_network(network_name, node_num, min_res, max_res)


# transits： transit域数量
# stubs: 每个transit节点连接的stub域数量
# transit_nodes: 每个transit域中节点数量
# transit_p: transit域内的连通性
# stub_nodes: 每个stub域中节点数量
# stub_p: stub域内的连通性
def make_sub_ts(transits, stubs, transit_nodes, stub_nodes, min_res, max_res):
    """生成物理网络文件（基于Transit-Stub模型）"""

    network_name = 'sub-ts'

    # # 生成GT-ITM配置文件
    # spec_filename = 'spec-%s' % network_name
    # with open(spec_dir + spec_filename, 'w') as f:
    #     f.write("ts 1 47\n")
    #     f.write("%d 0 0\n" % stubs)
    #     f.write("%d 2 3 1.0\n" % transits)
    #     f.write("%d 5 3 %f\n" % (transit_nodes, transit_p))
    #     f.write("%d 5 3 %f\n" % (stub_nodes, stub_p))

    # 生成基于transit-stub模型的物理网络文件
    node_num = transits * transit_nodes * (1 + stubs * stub_nodes)
    generate_network(network_name, node_num, min_res, max_res, transit_nodes=transit_nodes)


def make_req(index, min_res, max_res, node_num, time, duration):
    """生成虚拟网络请求文件"""

    network_name = 'req%s' % index
    generate_network(network_name, node_num, min_res, max_res, time=time, duration=duration)


# possion_mean：虚拟网络请求的到达服从泊松分布，且平均每1000个时间单位内到达的数量为possion_mean个
# 虚拟节点数量服从[min_num_nodes, max_num_nodes]的均匀分布
def make_batch_req(possion_mean, min_num_nodes, max_num_nodes, min_res, max_res):
    """生成多个虚拟网络请求文件"""

    # 时间间隔
    interval = 1000
    # 虚拟网络请求数量
    req_num = int(possion_mean / interval * TOTAL_TIME)
    # 在一个时间间隔内到达的VNR数量
    req_num_interval = 0
    # 记录该时间间隔内已到达的VNR数量
    count = 0
    # 记录已经经历了多少个时间间隔
    p = 0
    # 每个时间间隔的起始时间
    start = 0

    # 按照以下步骤分别生成req_num个虚拟网络请求文件
    for i in range(req_num):

        if count == req_num_interval:
            req_num_interval = 0
            while req_num_interval == 0:
                req_num_interval = np.random.poisson(possion_mean)
            count = 0
            start = p * interval
            p += 1
        count += 1
        time = start + ((count + 1) / (req_num_interval + 1)) * interval
        duration = MIN_DURATION + int(-math.log(random.random()) * (DURATION_MEAN - MIN_DURATION))

        if random.random() <= 0.4:
            node_amount = random.randint(min_num_nodes, max_num_nodes)
        else:
            node_amount = random.randint(min_num_nodes, (min_num_nodes + max_num_nodes)/2)

        make_req(i, min_res, max_res, node_amount, time, duration)

        # 生成子虚拟网络文件
        for j in range(4):

            j_node_amount = random.randint(2, min_num_nodes-1)
            index = "%d-%d" % (i, j)
            make_req(index, 0, max_res * 0.5, j_node_amount, time, duration)


def extract_network(path, filename):
    """读取网络文件并生成networkx.Graph实例"""

    node_id, link_id = 0, 0

    with open(path + filename) as f:
        lines = f.readlines()

    if len(lines[0].split()) == 2:
        """create a substrate network"""

        node_num, link_num = [int(x) for x in lines[0].split()]
        graph = nx.Graph()
        for line in lines[1: node_num + 1]:
            x, y, c = [float(x) for x in line.split()]
            graph.add_node(node_id, x_coordinate=x, y_coordinate=y, cpu=c, cpu_remain=c)
            node_id = node_id + 1

        for line in lines[-link_num:]:
            src, dst, bw, dis = [float(x) for x in line.split()]
            graph.add_edge(int(src), int(dst), link_id=link_id, bw=bw, bw_remain=bw, distance=dis)
            link_id = link_id + 1
    else:
        """create a virtual network"""

        node_num, link_num, time, duration, max_dis = [int(x) for x in lines[0].split()]
        graph = nx.Graph(type=0, time=time, duration=duration)
        for line in lines[1:node_num + 1]:
            x, y, c = [float(x) for x in line.split()]
            graph.add_node(node_id, x_coordinate=x, y_coordinate=y, cpu=c, cpu_remain=c)
            node_id = node_id + 1

        for line in lines[-link_num:]:
            src, dst, bw, dis = [float(x) for x in line.split()]
            graph.add_edge(int(src), int(dst), link_id=link_id, bw=bw, bw_remain=bw, distance=dis)
            link_id = link_id + 1

    return graph


def simulate_events(path, number):
    """读取number个虚拟网络及其自虚拟网络请求，构成底层虚拟网络请求事件队列和子虚拟网络请求事件队列"""

    # 第1层虚拟网络请求
    queue = []
    # 第2层虚拟网络请求
    queue_second = []

    for i in range(number):
        filename = 'req%d.txt' % i
        req_arrive = extract_network(path, filename)
        req_arrive.graph['id'] = i
        req_leave = copy.deepcopy(req_arrive)
        req_leave.graph['type'] = 1
        req_leave.graph['time'] = req_arrive.graph['time'] + req_arrive.graph['duration']
        queue.append(req_arrive)
        queue.append(req_leave)

        children = []
        for j in range(4):
            second_filename = 'req%d-%d.txt' % (i, j)
            second_req = extract_network(path, second_filename)
            children.append(second_req)
        queue_second.append(children)
    # 按照时间（到达时间或离开时间）对这些虚拟网络请求从小到大进行排序
    queue.sort(key=lambda r: r.graph['time'])
    return queue, queue_second


def simulate_events_one(path, number):
    """读取number个虚拟网络，构成虚拟网络请求事件队列"""

    queue = []
    for i in range(number):
        filename = 'req%d.txt' % i
        req_arrive = extract_network(path, filename)
        req_arrive.graph['id'] = i
        req_leave = copy.deepcopy(req_arrive)
        req_leave.graph['type'] = 1
        req_leave.graph['time'] = req_arrive.graph['time'] + req_arrive.graph['duration']
        queue.append(req_arrive)
        queue.append(req_leave)
    # 按照时间（到达时间或离开时间）对这些虚拟网络请求从小到大进行排序
    queue.sort(key=lambda r: r.graph['time'])
    return queue


if __name__ == '__main__':
    pass

    # 生成节点数为100，连通率为0.5的随机型物理网络
    # make_sub_wm(100, 50, 100)

    # # 生成节点数为1×4×(1+3×8)=100，连通率为0.5的Transit-Stub型物理网络
    # make_sub_ts(1, 3, 4, 8, 50, 100)
    #
    # # 平均每1000个时间单位内到达40个虚拟网络请求， 且虚拟节点数服从10~20的均匀分布，请求资源服从50~100的均匀分布
    # make_batch_req(40, 10, 20, 0, 50)
