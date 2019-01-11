# analytical tools
import tensorflow as tf
import numpy as np

# python libraries
import time
import itertools

# local dependencies
from env.environment import build_env
from algorithm.policy import Policy
from algorithm.worker import Worker
from algorithm.parts.optimizers import build_optimizer

class Model(object):
    '''
    This class has methods for training, testing, logging and saving.
    '''
    def __init__(self, env_id, policy_settings, hyperparams):
        self.hyperparams = hyperparams
        # Build environments
        self.envs = [build_env(env_id, env_num) for env_num in\
                     range(self.hyperparams.num_envs)]
        # Build policy which includes methods to access logits, value, entropy etc...
        # Two policies (with shared parameters) are built because the custom LSTM
        # cell in the recurrent network does not support dynamic tensor shapes
        with tf.variable_scope('policy', reuse=tf.AUTO_REUSE):
            self.policy = Policy(*policy_settings, self.envs[0].num_actions,
                                 self.envs[0].state_shape, self.hyperparams.env_steps)
            self.play_policy = Policy(*policy_settings, self.envs[0].num_actions,
                                 self.envs[0].state_shape, 1)
        # Build optimizer, which will train the policy
        self.trainer = build_optimizer('adam')()(
            self.policy.neglog_a_probs, self.policy.value, self.policy.entropy,
            **self.policy.placeholders)
        # tensorboard logger
        with tf.variable_scope('stats'):
            self.summarizers = self._build_summarizers()
        # Build saver and start session
        self.saver = tf.train.Saver(max_to_keep=10)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # build worker, which will be used to collect data from envs
        self.worker = Worker(self.play_policy, self.policy, self.sess,
                             self.hyperparams.env_steps, self.hyperparams.gamma,
                             self.hyperparams.lam)

    def train(self, num_batches, log_every, save_every, save_path, tb_path):
        '''
        Trains model for specified number of batches. Logs and saves model as specified.
        '''
        if tb_path:
            tb_logger = tf.summary.FileWriter(tb_path, graph=self.sess.graph)
        # anneal lr or create function that keeps lr constant because of max
        if callable(eval(self.hyperparams.learning_rate)):
            lr_func = eval(self.hyperparams.learning_rate)
        else:
            lr_func = lambda x: max(x, 1) * float(self.hyperparams.learning_rate)
        for global_step in range(num_batches):
            self.hyperparams = self.hyperparams._replace(
                learning_rate=lr_func(1 - global_step / num_batches))
            play_results, avg_stats = self._collect_batch_data()
            avg_loss_vals = self._train_for_batch(play_results)
            summaries = self._summarize(*avg_loss_vals, *avg_stats)
            if (global_step + 1) % log_every == 0 and tb_path:
                self._log_to_tb(summaries, tb_logger)
            if (global_step + 1) % save_every == 0 and save_path:
                self._save(save_path)

    def test(self, num_test_games):
        '''
        Allows user to evaluate model by watching it play game(s).
        '''
        for _ in range(num_test_games):
            game_length = self.worker.play(self.envs[0], num_steps=100000,
                                           end_condition='game_end', render=True)
            print('GAME OVER, length: {}'.format(game_length))

    def _train_for_batch(self, play_results):
        '''
        Perform minibatch stochastic gradient descent. Also trains each batch over
        multiple updates so order the model is trained on for each env is random.
        Each minibatch has a size equal to the number of steps taken in each env.
        So, batch size = num_envs * env_steps
        '''
        loss_vals = []
        env_nums = np.arange(self.hyperparams.num_envs)
        for _ in range(self.hyperparams.batch_updates):
            np.random.shuffle(env_nums)
            for i in range(len(env_nums)):
                env_num = env_nums[i]
                start = env_num * self.hyperparams.env_steps
                end = start + self.hyperparams.env_steps
                mb_play_results = [r[start:end] for r in play_results]
                loss_vals.append(self._update_params(*mb_play_results, env_num)[1:])
        return [np.mean(v) for v in zip(*loss_vals)]

    def _collect_batch_data(self):
        '''
        Unpack results from workers.
        '''
        worker_results = [self.worker.play(e, self.hyperparams.env_steps) for e in\
                          self.envs]
        env_play_results, env_play_stats = [list(r) for r in zip(*worker_results)]
        play_results_as_lists = [list(r) for r in zip(*env_play_results)]
        play_results = [list(itertools.chain.from_iterable(l)) for l in\
                        play_results_as_lists]
        play_stats = [list(s) for s in zip(*env_play_stats)]
        avg_stats = [np.mean(s) for s in play_stats[:-1]]
        self.games_played += np.sum(play_stats[-1])
        self.global_step += 1
        return play_results, avg_stats

    def _update_params(self, states, actions, returns, advs, old_a, old_v, dones, env_num):
        '''
        Updates params.
        '''
        # normalize all advantages
        advs = np.asarray(advs)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        # create feed dict
        feed_dict = {
            self.policy.states: states,
            self.policy.actions: actions,
            self.policy.returns: returns,
            self.policy.advantages: advs,
            self.policy.old_a: old_a,
            self.policy.old_v: old_v,
            self.policy.learning_rate: self.hyperparams.learning_rate,
            self.policy.grad_clip: self.hyperparams.grad_clip,
            self.policy.c1: self.hyperparams.c1,
            self.policy.c2: self.hyperparams.c2,
            self.policy.eps: self.hyperparams.eps
        }
        if 'rec_masks' in self.policy.__dict__:
            feed_dict[self.policy.rec_masks] = dones
            cur_env = [env for env in self.envs if env.env_num == env_num][0]
            feed_dict[self.policy.rec_state] = cur_env.rec_state
        # update params
        return self.sess.run(self.trainer, feed_dict=feed_dict)

    
    def _log_to_tb(self, summaries, tb_logger):
        '''
        Add summaries to logger.
        '''
        tb_logger.add_summary(summaries[0], self.global_step)
        tb_logger.add_summary(summaries[1], self.games_played)
        tb_logger.flush()

    def _summarize(self, policy_loss, value_loss, entropy, approxkl, clipfrac,
                   avg_reward, avg_game_score):
        '''
        Logs to tensorboard.
        '''
        return self.sess.run(self.summarizers, feed_dict={
            self.policy_loss: policy_loss,
            self.value_loss: value_loss,
            self.entropy: entropy,
            self.avg_reward: avg_reward,
            self.avg_game_score: avg_game_score,
            self.approxkl: approxkl,
            self.clipfrac: clipfrac
        })

    def _build_summarizers(self):
        '''
        Logs to summaries to tensorboard. These placeholders are only used for logging
        purposes.
        '''
        self.policy_loss = tf.placeholder(tf.float32, [], name='pl_pl')
        self.value_loss = tf.placeholder(tf.float32, [], name='pl_vl')
        self.entropy = tf.placeholder(tf.float32, [], name='pl_e')
        self.avg_reward = tf.placeholder(tf.float32, [], name='pl_r')
        self.avg_game_score = tf.placeholder(tf.float32, [], name='pl_gsc')
        self.approxkl = tf.placeholder(tf.float32, [], name='pl_akl')
        self.clipfrac = tf.placeholder(tf.float32, [], name='pl_cf')
        # summaries using global step as x axis
        step_summaries = tf.summary.merge([
            tf.summary.scalar('global_step/policy_loss', self.policy_loss),
            tf.summary.scalar('global_step/value_loss', self.value_loss),
            tf.summary.scalar('global_step/entropy', self.entropy),
            tf.summary.scalar('global_step/avg_reward', self.avg_reward),
            tf.summary.scalar('global_step/avg_game_score', self.avg_game_score),
            tf.summary.scalar('global_step/approxkl', self.approxkl),
            tf.summary.scalar('global_step/clipfrac', self.clipfrac)
        ])
        # summaries using games as x axis
        game_summaries = tf.summary.merge([
            tf.summary.scalar('games/policy_loss', self.policy_loss),
            tf.summary.scalar('games/value_loss', self.value_loss),
            tf.summary.scalar('games/entropy', self.entropy),
            tf.summary.scalar('games/avg_reward', self.avg_reward),
            tf.summary.scalar('games/avg_game_score', self.avg_game_score),
            tf.summary.scalar('games/approxkl', self.approxkl),
            tf.summary.scalar('games/clipfrac', self.clipfrac)
        ])
        # these will be used as the x axes in tb
        self.global_step = 0
        self.games_played = 0
        return [step_summaries, game_summaries]

    def _save(self, save_path):
        '''
        Saves model.
        '''
        self.saver.save(self.sess, save_path, global_step=self.global_step)
        print('Saved model to: {}-{}'.format(save_path, self.global_step))
