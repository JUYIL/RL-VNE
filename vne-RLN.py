import time
from substrate import Substrate
from maker import simulate_events_one
from analysis import Analysis


def main():

    # Step1: 读取底层网络和虚拟网络请求文件
    network_files_dir = 'networks/'
    sub_filename = 'subts.txt'
    sub = Substrate(network_files_dir, sub_filename)
    event_queue1 = simulate_events_one('VNRequest/', 2000)

    # Step2: 选择映射算法
    algorithm = 'RLN'
    arg = 10

    # Step3: 处理虚拟网络请求事件
    start = time.time()
    sub.handle(event_queue1, algorithm, arg)
    time_cost = time.time()-start
    print(time_cost)

    # Step4: 输出映射结果文件
    tool = Analysis()
    tool.save_result(sub, '%s-VNE-0321-%s-t7-2.txt' % (algorithm, arg))


if __name__ == '__main__':
    main()
