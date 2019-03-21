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
                n_features=7,
                learning_rate=0.05,
                num_epoch=arg,
                batch_size=100)
        rln.train(training_set)
        nodesaver = tf.train.Saver()
        nodesaver.save(rln.sess, "./Mine/nodemodel/nodemodel.ckpt")
        return rln

    else:
        training_set_path = 'Mine/training_set/'
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
