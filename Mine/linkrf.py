import tensorflow as tf
import time
from Mine.nodemdp import NodeEnv
from Mine.linkmdp import *



class nodepolicy:
    def __init__(self,
                 n_actions,
                 n_features,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self._build_net()
        self.req_as = []
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features[0], self.n_features[1], 1],
                                         name="observations")

        with tf.name_scope('conv'):
            read=tf.train.NewCheckpointReader('./Mine/nodemodel/nodemodel.ckpt')

            kernel=read.get_tensor('conv/weights')
            conv = tf.nn.conv2d(input=self.tf_obs,
                                filter=kernel,
                                strides=(1, 1,self.n_features[1],1),
                                padding='VALID')
            biases=read.get_tensor('conv/bias')
            conv1=tf.nn.relu(tf.nn.bias_add(conv,biases))
            self.scores=tf.reshape(conv1,[-1,self.n_features[0]])

        with tf.name_scope('output'):
            self.probs=tf.nn.softmax(self.scores)

    def choose_max_action(self,observation,sub,current_node_cpu,curreqnum):
        x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
        prob_weights = self.sess.run(self.scores,
                                     feed_dict={self.tf_obs: x})

        candidate_action = []
        candidate_score = []
        for index, score in enumerate(prob_weights.ravel()):
            if index not in self.req_as and sub.nodes[index]['cpu_remain'] >= current_node_cpu:
                candidate_action.append(index)
                candidate_score.append(score)
        if len(candidate_action) == 0:
            return -1
        else:
            action_index = candidate_score.index(np.max(candidate_score))
            action = candidate_action[action_index]

        self.req_as.append(action)
        if len(self.req_as) == curreqnum:
            self.req_as = []

        return action

class LinkPolicy:

    def __init__(self,sub, n_actions, n_features, learning_rate, num_epoch, batch_size):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.sub=copy.deepcopy(sub)
        self.linkpath = getallpath(self.sub.net)
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_actions, self.n_features, 1],
                                         name="observations")
            self.tf_acts = tf.placeholder(tf.int32,  name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="action_value")


        with tf.name_scope('conv'):
            self.kernel=tf.Variable(tf.truncated_normal([1,self.n_features,1,1],dtype=tf.float32,stddev=0.1),
                               name="weights")
            conv = tf.nn.conv2d(input=self.tf_obs,
                                filter=self.kernel,
                                strides=(1, 1,self.n_features,1),
                                padding='VALID')
            self.bias=tf.Variable(tf.constant(0.0,shape=[1],dtype=tf.float32),name="bias")
            conv1=tf.nn.relu(tf.nn.bias_add(conv,self.bias))
            self.scores=tf.reshape(conv1,[-1,self.n_actions])

        with tf.name_scope('output'):
            self.probs=tf.nn.softmax(self.scores)

        # loss function
        with tf.name_scope('loss'):
            # 获取策略网络中全部可训练的参数
            self.tvars = tf.trainable_variables()
            # 设置虚拟label的placeholder
            self.input_y = tf.placeholder(tf.float32, [None, self.n_actions], name="input_y")
            # 计算损失函数loss(当前Action对应的概率的对数)
            self.loglik = -tf.reduce_sum(tf.log(self.probs) * self.input_y, axis=1)
            self.loss = tf.reduce_mean(self.loglik)
            # 计算损失函数梯度
            self.newGrads = tf.gradients(self.loss, self.tvars)


        # Optimizer
        with tf.name_scope('update'):
            # 权重参数梯度
            self.kernel_grad = tf.placeholder(tf.float32, name="batch_grad1")
            # 偏置参数梯度
            self.biases_grad = tf.placeholder(tf.float32, name="batch_grad2")
            # 整合两个梯度
            self.batch_grad = [self.kernel_grad, self.biases_grad]
            # 优化器
            adam = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.update_grads = adam.apply_gradients(zip(self.batch_grad, self.tvars))


    def choose_action(self, observation,sub, linkbw, linkpath,vfr,vto):
        x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
        prob_weights = self.sess.run(self.scores,
                                     feed_dict={self.tf_obs: x})

        candidate_action = []
        candidate_score = []
        for index, score in enumerate(prob_weights.ravel()):
            s_fr = list(linkpath[index].keys())[0][0]
            s_to = list(linkpath[index].keys())[0][1]
            v_fr = vfr
            v_to = vto
            if s_fr==v_fr and s_to==v_to and minbw(sub,list(linkpath[index].values())[0]) >= linkbw:
                candidate_action.append(index)
                candidate_score.append(score)
        if len(candidate_action) == 0:
            return -1
        else:
            candidate_prob = np.exp(candidate_score) / np.sum(np.exp(candidate_score))
            action = np.random.choice(candidate_action, p=candidate_prob)

        return action

    def choose_max_action(self, observation,sub, linkbw, linkpath,vfr,vto):
        x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
        prob_weights = self.sess.run(self.scores,
                                     feed_dict={self.tf_obs: x})

        candidate_action = []
        candidate_score = []
        for index, score in enumerate(prob_weights.ravel()):
            s_fr = list(linkpath[index].keys())[0][0]
            s_to = list(linkpath[index].keys())[0][1]
            v_fr = vfr
            v_to = vto
            if s_fr==v_fr and s_to==v_to and minbw(sub,list(linkpath[index].values())[0]) >= linkbw:
                candidate_action.append(index)
                candidate_score.append(score)
        if len(candidate_action) == 0:
            return -1
        else:
            # candidate_prob = np.exp(candidate_score) / np.sum(np.exp(candidate_score))
            # action = np.random.choice(candidate_action, p=candidate_prob)
            action_index = candidate_score.index(np.max(candidate_score))
            action = candidate_action[action_index]

        return action

    #train model
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
            nodeenv = NodeEnv(self.sub.net)
            nodep = nodepolicy(nodeenv.action_space.n,nodeenv.observation_space.shape)
            linkenv = LinkEnv(self.sub.net)
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
                    nodeenv.set_vnr(req)
                    # 获得底层网络的状态
                    nodeobservation = nodeenv.reset()

                    node_map = {}
                    for vn_id in range(req.number_of_nodes()):

                        sn_id = nodep.choose_max_action(nodeobservation,nodeenv.sub,req.nodes[vn_id]['cpu'],req.number_of_nodes())
                        if sn_id == -1:
                            break
                        else:
                            # 执行一次action，获取返回的四个数据
                            nodeobservation, _, done, info = nodeenv.step(sn_id)
                            node_map.update({vn_id: sn_id})
                    # end for,即一个VNR的全部节点映射全部尝试完毕

                    if len(node_map) == req.number_of_nodes():
                        print('link mapping...')
                        linkenv.set_vnr(req)
                        linkob=linkenv.reset()
                        link_map = {}
                        xs, acts = [], []
                        for link in req.edges:
                            linkenv.set_link(link)
                            vn_from = link[0]
                            vn_to = link[1]
                            sn_from = node_map[vn_from]
                            sn_to = node_map[vn_to]
                            bw = req[vn_from][vn_to]['bw']
                            if nx.has_path(linkenv.sub, sn_from, sn_to):

                                x = np.reshape(linkob, [1, linkob.shape[0], linkob.shape[1], 1])
                                linkaction = self.choose_action(linkob, linkenv.sub, bw, self.linkpath, sn_from, sn_to)
                                if linkaction == -1:
                                    break
                                else:
                                    # 输入的环境信息添加到xs列表中
                                    xs.append(x)
                                    # 将选择的动作添加到acts列表中
                                    acts.append(linkaction)
                                    # 执行一次action，获取返回的四个数据
                                    linkob, _, done, info = linkenv.step(linkaction)
                                    path = list(self.linkpath[linkaction].values())[0]
                                    link_map.update({link: path})


                        if len(link_map) == req.number_of_edges():

                            reward=self.calculate_reward(req,node_map,link_map)

                            ys = tf.one_hot(acts, self.n_actions)
                            epx = np.vstack(xs)
                            epy = tf.Session().run(ys)

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
                    else:
                        print("Failure!")

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
                nodeenv.set_sub(sub_copy.net)
                linkenv.set_sub(sub_copy.net)

            loss_average.append(np.mean(values))
            iteration = iteration + 1

        end = (time.time() - start) / 3600
        with open('results/linklosslog-%s.txt' % self.num_epoch, 'w') as f:
            f.write("Training time: %s hours\n" % end)
            for value in loss_average:
                f.write(str(value))
                f.write('\n')

    def run(self, sub, req,node_map):
        """基于训练后的策略网络，直接得到每个虚拟网络请求的节点映射集合"""
        linkenv=LinkEnv(sub)
        linkenv.set_vnr(req)
        linkob = linkenv.reset()
        link_map = {}
        xs, acts = [], []
        for link in req.edges:
            linkenv.set_link(link)
            vn_from = link[0]
            vn_to = link[1]
            sn_from = node_map[vn_from]
            sn_to = node_map[vn_to]
            bw = req[vn_from][vn_to]['bw']
            if nx.has_path(linkenv.sub, sn_from, sn_to):

                x = np.reshape(linkob, [1, linkob.shape[0], linkob.shape[1], 1])
                linkaction = self.choose_action(linkob, linkenv.sub, bw, self.linkpath, sn_from, sn_to)
                if linkaction == -1:
                    break
                else:
                    # 输入的环境信息添加到xs列表中
                    xs.append(x)
                    # 将选择的动作添加到acts列表中
                    acts.append(linkaction)
                    # 执行一次action，获取返回的四个数据
                    linkob, _, done, info = linkenv.step(linkaction)
                    path = list(self.linkpath[linkaction].values())[0]
                    link_map.update({link: path})
        return link_map

    def calculate_reward(self, req, node_map,link_map):

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

        return reward
