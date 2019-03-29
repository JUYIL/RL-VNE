from maker import simulate_events_one
from comparison1.grc import GRC
from comparison2.mcts import MCTS
from comparison3.reinforce import RL
from Mine.reinforce import RLN
from Mine.linkrf import LinkPolicy
import tensorflow as tf


def configure(sub, name, arg):

    if name == 'grc':
        grc = GRC(damping_factor=0.9, sigma=1e-6)
        return grc

    elif name == 'mcts':
        mcts = MCTS(computation_budget=5, exploration_constant=0.5)
        return mcts

    elif name == 'rl':
        training_set_path = 'comparison3/training_set/'
        training_set = simulate_events_one(training_set_path, 1000)
        rl = RL(sub=sub,
                n_actions=sub.net.number_of_nodes(),
                n_features=4,
                learning_rate=0.05,
                num_epoch=arg,
                batch_size=100)
        rl.train(training_set)
        return rl

    elif name == 'RLN':
        training_set_path = 'Mine/training_set/'
        training_set = simulate_events_one(training_set_path, 1000)
        rln = RLN(sub=sub,
                n_actions=sub.net.number_of_nodes(),
                n_features=6,
                learning_rate=0.05,
                num_epoch=arg,
                batch_size=100)
        rln.train(training_set)
        nodesaver = tf.train.Saver()
        nodesaver.save(rln.sess, "./Mine/nodemodel/nodemodel.ckpt")
        return rln

    else:
        training_set_path = 'Mine/linktrset/'
        training_set = simulate_events_one(training_set_path, 1000)
        rlnl = LinkPolicy(sub=sub,
                n_actions=59614,
                n_features=2,
                learning_rate=0.05,
                num_epoch=arg,
                batch_size=100)
        rlnl.train(training_set)
        linksaver = tf.train.Saver()
        linksaver.save(rlnl.sess, "./Mine/linkmodel/linkmodel.ckpt")
        return rlnl

# grc 53    0.3
# mcts 4336 0.628
# RLN 10 T6 2289 0.7375
# RLN 50 T6 3668 0.71
# RLN 100 15241
# rl 10 1679 0.67
# rl 50 8359 0.69
# rl 100 9655 0.71
# RLN 10 t5 1650 0.7255
# RLN 10 t6 2137 0.7425
# RLN 10 T7 2289 0.7375
# RLNL 14:00 29507-5000 0.8
# RLNL t6 newtrset 10000 0.8

# import networkx as nx
# import matplotlib.pyplot as plt
# from maker import extract_network
# network_files_dir = 'networks/'
# sub_filename = 'subts.txt'
# sub = extract_network(network_files_dir, sub_filename)
# nx.draw(sub, with_labels=False, node_color='black', edge_color='gray', node_size=50)
# plt.savefig('resultsj/subts.png')
# plt.show()