import tensorflow as tf
import numpy as np
import copy
import time
from comparison3.mdp import Env


class RL:

    def __init__(self, sub, n_actions, n_features, learning_rate, num_epoch, batch_size):
        self.n_actions = n_actions  # 动作空间大小
        self.n_features = n_features  # 节点向量维度
        self.lr = learning_rate  # 学习速率
        self.num_epoch = num_epoch  # 训练轮数
        self.batch_size = batch_size  # 批处理的批次大小

        self.sub = copy.deepcopy(sub)
        self._build_model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, training_set):

        loss_average = []
        iteration = 0
        start = time.time()
        # 训练开始
        while iteration < self.num_epoch:
            values = []
            print("Iteration %s" % iteration)
            # 每轮训练开始前，都需要重置底层网络和相关的强化学习环境
            sub_copy = copy.deepcopy(self.sub)
            env = Env(self.sub.net)
            # 创建存储参数梯度的缓冲器
            grad_buffer = self.sess.run(self.tvars)
            # 初始化为0
            for ix, grad in enumerate(grad_buffer):
                grad_buffer[ix] = grad * 0
            # 记录已经处理的虚拟网络请求数量
            counter = 0
            for req in training_set:
                # 当前待映射的虚拟网络请求ID
                req_id = req.graph['id']
                print("\nHandling req%s..." % req_id)

                if req.graph['type'] == 0:

                    print("\tIt's a newly arrived request, try to map it...")
                    counter += 1
                    sub_copy.total_arrived = counter
                    # 向环境传入当前的待映射虚拟网络
                    env.set_vnr(req)
                    # 获得底层网络的状态
                    observation = env.reset()

                    node_map = {}
                    xs, acts = [], []
                    for vn_id in range(req.number_of_nodes()):
                        x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])

                        sn_id = self.choose_action(observation, sub_copy.net, req.nodes[vn_id]['cpu'], acts)

                        if sn_id == -1:
                            break
                        else:
                            # 输入的环境信息添加到xs列表中
                            xs.append(x)
                            # 将选择的动作添加到acts列表中
                            acts.append(sn_id)
                            # 执行一次action，获取返回的四个数据
                            observation, _, done, info = env.step(sn_id)
                            node_map.update({vn_id: sn_id})
                    # end for,即一个VNR的全部节点映射全部尝试完毕

                    if len(node_map) == req.number_of_nodes():

                        reward, link_map = self.calculate_reward(sub_copy, req, node_map)

                        if reward != -1:
                            epx = np.vstack(xs)
                            epy = np.eye(self.n_actions)[acts]
                            # 返回损失函数值
                            loss_value = self.sess.run(self.loss,
                                                       feed_dict={self.tf_obs: epx,
                                                                  self.input_y: epy})

                            print("Success! The loss value is: %s" % loss_value)
                            values.append(loss_value)

                            # 返回求解梯度
                            tf_grad = self.sess.run(self.newGrads,
                                                    feed_dict={self.tf_obs: epx,
                                                               self.input_y: epy})
                            # 将获得的梯度累加到gradBuffer中
                            for ix, grad in enumerate(tf_grad):
                                grad_buffer[ix] += grad
                            grad_buffer[0] *= reward
                            grad_buffer[1] *= reward

                            # 更新底层网络
                            sub_copy.mapped_info.update({req.graph['id']: (node_map, link_map)})
                            sub_copy.change_resource(req, 'allocate')
                        else:
                            print("Failure!")

                    # 当实验次数达到batch size整倍数，累积的梯度更新一次参数
                    if counter % self.batch_size == 0:
                        self.sess.run(self.update_grads,
                                      feed_dict={self.kernel_grad: grad_buffer[0],
                                                 self.biases_grad: grad_buffer[1]})

                        # 清空gradBuffer
                        for ix, grad in enumerate(grad_buffer):
                            grad_buffer[ix] = grad * 0

                if req.graph['type'] == 1:

                    print("\tIt's time is out, release the occupied resources")
                    if req_id in sub_copy.mapped_info.keys():
                        sub_copy.change_resource(req, 'release')

                env.set_sub(sub_copy.net)

            loss_average.append(np.mean(values))
            iteration = iteration + 1

        end = (time.time() - start) / 3600
        with open('results/loss-%s.txt' % self.num_epoch, 'w') as f:
            f.write("Training time: %s hours\n" % end)
            for value in loss_average:
                f.write(str(value))
                f.write('\n')

    def run(self, sub, req):
        """基于训练后的策略网络，直接得到每个虚拟网络请求的节点映射集合"""

        node_map = {}
        env = Env(sub.net)
        env.set_vnr(req)
        observation = env.reset()
        acts = []
        for vn_id in range(req.number_of_nodes()):
            sn_id = self.choose_max_action(observation, sub.net, req.nodes[vn_id]['cpu'], acts)
            if sn_id == -1:
                break
            else:
                acts.append(sn_id)
                observation, _, done, info = env.step(sn_id)
                node_map.update({vn_id: sn_id})
        return node_map

    def _build_model(self):
        """搭建策略网络"""

        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.n_actions, self.n_features, 1],
                                         name="observations")

            self.tf_acts = tf.placeholder(dtype=tf.int32,
                                          shape=[None, ],
                                          name="actions_num")

            self.tf_vt = tf.placeholder(dtype=tf.float32,
                                        shape=[None, ],
                                        name="action_value")

        with tf.name_scope("conv"):
            self.kernel = tf.Variable(tf.truncated_normal([1, self.n_features, 1, 1],
                                                          dtype=tf.float32,
                                                          stddev=0.1),
                                      name="weights")
            conv = tf.nn.conv2d(input=self.tf_obs,
                                filter=self.kernel,
                                strides=[1, 1, self.n_features, 1],
                                padding="VALID")
            self.bias = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                                    name="bias")
            conv1 = tf.nn.relu(tf.nn.bias_add(conv, self.bias))
            self.scores = tf.reshape(conv1, [-1, self.n_actions])

        with tf.name_scope("output"):
            self.probability = tf.nn.softmax(self.scores)

        # 损失函数
        with tf.name_scope('loss'):
            # 获取策略网络中全部可训练的参数
            self.tvars = tf.trainable_variables()
            # 设置虚拟label的placeholder
            self.input_y = tf.placeholder(tf.float32, [None, self.n_actions], name="input_y")
            # 计算损失函数loss(当前Action对应的概率的对数)
            self.loglik = -tf.reduce_sum(tf.log(self.probability) * self.input_y, axis=1)
            self.loss = tf.reduce_mean(self.loglik)
            # 计算损失函数梯度
            self.newGrads = tf.gradients(self.loss, self.tvars)

        # 批量梯度更新
        with tf.name_scope('update'):
            # 权重参数梯度
            self.kernel_grad = tf.placeholder(tf.float32, name="batch_grad1")
            # 偏置参数梯度
            self.biases_grad = tf.placeholder(tf.float32, name="batch_grad2")
            # 整合两个梯度
            self.batch_grad = [self.kernel_grad, self.biases_grad]
            # 优化器
            adam = tf.train.AdamOptimizer(learning_rate=self.lr)
            # 累计到一定样本梯度，执行updateGrads更新参数
            self.update_grads = adam.apply_gradients(zip(self.batch_grad, self.tvars))

    def choose_action(self, observation, sub, current_node_cpu, acts):
        """在给定状态observation下，根据策略网络输出的概率分布选择动作，供训练阶段使用，兼顾了探索和利用"""

        # 规范化网络输入格式
        x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])

        tf_score = self.sess.run(self.scores, feed_dict={self.tf_obs: x})
        candidate_action = []
        candidate_score = []
        for index, score in enumerate(tf_score.ravel()):
            if index not in acts and sub.nodes[index]['cpu_remain'] >= current_node_cpu:
                candidate_action.append(index)
                candidate_score.append(score)

        if len(candidate_action) == 0:
            return -1
        else:
            candidate_prob = np.exp(candidate_score) / np.sum(np.exp(candidate_score))
            # 选择动作
            action = np.random.choice(candidate_action, p=candidate_prob)
            return action

    def choose_max_action(self, observation, sub, current_node_cpu, acts):
        """在给定状态observation下，根据策略网络输出的概率分布选择概率最大的动作，仅利用"""

        x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
        tf_prob = self.sess.run(self.probability, feed_dict={self.tf_obs: x})
        filter_prob = tf_prob.ravel()
        for index, score in enumerate(filter_prob):
            if index in acts or sub.nodes[index]['cpu_remain'] < current_node_cpu:
                filter_prob[index] = 0.0
        action = np.argmax(filter_prob)
        if filter_prob[action] == 0.0:
            return -1
        else:
            return action

    def calculate_reward(self, sub, req, node_map):

        link_map = sub.link_mapping(req, node_map)
        if len(link_map) == req.number_of_edges():
            requested, occupied = 0, 0

            # node resource
            for vn_id, sn_id in node_map.items():
                node_resource = req.nodes[vn_id]['cpu']
                occupied += node_resource
                requested += node_resource

            # link resource
            for vl, path in link_map.items():
                link_resource = req[vl[0]][vl[1]]['bw']
                requested += link_resource
                occupied += link_resource * (len(path) - 1)

            reward = requested / occupied

            return reward, link_map
        else:
            return -1, link_map
