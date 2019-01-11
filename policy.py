# analytical tools
import tensorflow as tf
import numpy as np
import cv2 as cv

# python libraries
from time import sleep
from pdb import set_trace as bp

# local dependencies
from networks import NetworkBuilder
# from networks import build_network

class Policy(object):

    def __init__(self, shared_network, actor_network, value_network, num_actions,
                 state_size, num_steps):

        self.num_actions = num_actions
        self.state_size = state_size

        # set up placeholders for hyperparams and data
        self.learning_rate = tf.placeholder(tf.float32, [], name='pl_lr')
        self.grad_clip = tf.placeholder(tf.float32, [], name='pl_gc')
        self.c1 = tf.placeholder(tf.float32, [], name='pl_c1')
        self.c2 = tf.placeholder(tf.float32, [], name='pl_c2')
        self.eps = tf.placeholder(tf.float32, [], name='pl_eps')
        self.states = tf.placeholder(tf.float32, [None] + self.state_size, name='pl_st')
        self.actions = tf.placeholder(tf.int32, [None], name='pl_act')
        self.returns = tf.placeholder(tf.float32, [None], name='pl_ret')
        self.advantages = tf.placeholder(tf.float32, [None], name='pl_adv')
        self.old_a = tf.placeholder(tf.float32, [None], name='pl_oldp')
        self.old_v = tf.placeholder(tf.float32, [None], name='pl_oldv')
        g = tf.get_default_graph()
        placeholder_names = [n.name for n in g.as_graph_def().node\
                             if n.name.startswith('policy/pl')]
        self.placeholders = {n: g.get_tensor_by_name(n+':0') for n in placeholder_names}
        # create policy graph
        NB = NetworkBuilder()
        shared, actor, critic = [getattr(NB, n) for n in [shared_network, actor_network,\
                                 value_network]]
        with tf.variable_scope('shared_network'):
            if 'lstm' in shared_network:
                self.shared_val, rec_info = shared(self.states, steps=num_steps)
                self.rec_masks = rec_info['rec_masks']
                self.rec_state = rec_info['rec_state']
                self.cur_rec_state = rec_info['cur_rec_state']
            else:
                self.shared_val = shared(self.states)
        with tf.variable_scope('actor'):
            self.actor_logits = actor(self.shared_val, self.num_actions)
            self.neglog_a_probs = self._l_to_nlp(self.actor_logits)
            self.entropy = self._calc_entropy(self.actor_logits)
            self.take_action = self._sample(self.actor_logits)
        with tf.variable_scope('critic'):
            self.value = critic(self.shared_val)

    def _l_to_nlp(self, logits):
        '''
        Calculates the negative log (nlp) of actions taken given logits (l).
        Uses tf softmax cross entropy function because it has failsafes to avoid numerical issues.
        '''
        actions_one_hot = tf.one_hot(
            indices=self.actions, depth=self.num_actions, name='a_one_hot')
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=actions_one_hot, name='neglog_a_probs')

    def _calc_entropy(self, actor_logits):
        '''
        Calculates entropy with more numerical stability. From OpenAI Baselines.
        '''
        a0 = actor_logits - tf.reduce_max(actor_logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def _sample(self, actor_logits):
        '''
        Samples random action. From OpenAI Baselines.
        '''
        u = tf.random_uniform(tf.shape(actor_logits), dtype=actor_logits.dtype)
        return tf.argmax(actor_logits - tf.log(-tf.log(u)), axis=-1)
