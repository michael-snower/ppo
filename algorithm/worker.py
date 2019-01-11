# analytical tools
import tensorflow as tf
import numpy as np
import cv2 as cv

# python libraries
from time import sleep

class Worker(object):

    def __init__(self, play_policy, train_policy, sess, env_steps, gamma, lam):
        self.play_policy = play_policy
        self.train_policy = train_policy
        self.env_steps = env_steps
        self.sess = sess
        self.gamma = gamma
        self.lam = lam

    def play(self, env, num_steps, end_condition='batch', render=False):
        '''
        Collects training data for a batch.
        End condition can be 'batch' or 'game_end'
        '''
        batch_data = [env.state] # initalize this batch to start at the state the last batch ended on
        for i in range(num_steps):
            if render:
                env.render()
                sleep(0.05)
            batch_data.append(self._take_step(env))
            if env.done == True and end_condition == 'game_end':
                return i
        return self._process_play_data(env, batch_data)

    def _take_step(self, env):
        '''
        Takes step in the env.
        '''
        feed_dict = {self.play_policy.states: env.state[np.newaxis, ...]}
        if 'rec_masks' in self.play_policy.__dict__:
            feed_dict[self.play_policy.rec_masks] = [env.done]
            feed_dict[self.play_policy.rec_state] = env.rec_state
            action, env.rec_state = self.sess.run([self.play_policy.take_action,
                                                        self.play_policy.cur_rec_state],
                                                        feed_dict=feed_dict)
        else:
            action = self.sess.run(self.play_policy.take_action, feed_dict=feed_dict)
        return env.step(action[0])

    def _process_play_data(self, env, batch_data):
        '''
        Processes the batch data, calculates old neglog probabilites for actor, and returns.
        '''
        states, actions, rewards, dones, infos = [list(z) for z in zip(*batch_data[1:])]
        states.insert(0, batch_data[0]) # add extra state, which is required in returns calculation
        old_a, old_v, extra_val = self._calc_old(env, actions, states, dones)
        returns, advs = self._calc_returns(old_v, extra_val, rewards, dones)
        stats = self._calc_stats(env, dones, infos, rewards)
        return [states[:-1], actions, returns, advs, old_a, old_v, dones], stats

    def _calc_old(self, env, actions, states, dones):
        '''
        Calc old states and values.
        '''
        feed_dict_train = {self.train_policy.actions: actions,
                           self.train_policy.states: states[:-1]}
        if 'rec_masks' in self.train_policy.__dict__:
            feed_dict_train[self.train_policy.rec_masks] = dones
            feed_dict_train[self.train_policy.rec_state] = env.rec_state
        old_a, old_v = self.sess.run([self.train_policy.neglog_a_probs,
                                      self.train_policy.value],
                                      feed_dict=feed_dict_train)
        feed_dict_play = {self.play_policy.states: states[-1][np.newaxis, ...]}
        if 'rec_masks' in self.play_policy.__dict__:
            feed_dict_play[self.play_policy.rec_masks] = [dones[-1]]
            feed_dict_play[self.play_policy.rec_state] = env.rec_state
        extra_val = self.sess.run(self.play_policy.value,
                                  feed_dict=feed_dict_play)
        return old_a, old_v, extra_val

    def _calc_returns(self, old_v, extra_val, rewards, dones):
        '''
        Computes GAE (Generalized Advantage Estimation).
        This is from Shulman et al's paper - https://arxiv.org/pdf/1506.02438.pdf
        '''
        values = old_v.tolist()
        values.append(extra_val)
        advs = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0
        for i in reversed(range(len(rewards))):
            terminal_val = 1.0 - dones[i]
            next_value = values[i+1]
            delta = rewards[i] + self.gamma * next_value * terminal_val - values[i]
            advs[i] = last_gae = delta + self.gamma * self.lam * terminal_val * last_gae
        return advs + old_v, advs

    def _calc_stats(self, env, dones, infos, rewards):
        '''
        Calculates stats for a batch.
        '''
        game_end_indices = [j for j, d in enumerate(dones) if d == True]
        games_played = len(game_end_indices)
        # no games finish - update rewards by adding sum of rewards accumulated over batch
        if games_played == 0:
            env.cur_game_reward += np.sum(rewards)
        # cases where game ends
        else:
            first_game_end_index = dones[game_end_indices[0]]
            last_game_end_index = dones[game_end_indices[-1]]
            # done on last move - there's an off by 1 here, but I don't think it's a big deal
            if game_end_indices[-1] == len(rewards) - 1:
                env.last_game_reward = env.cur_game_reward +\
                                            np.sum(rewards[:first_game_end_index])
                env.cur_game_reward = np.sum(rewards[first_game_end_index:])
            # done(s) somewhere else
            else:
                env.last_game_reward = env.cur_game_reward +\
                                            np.sum(rewards[:first_game_end_index + 1])
                env.cur_game_reward = np.sum(rewards[first_game_end_index + 1:])
            # update score
            if 'score' in infos[0]:
                env.game_score = infos[first_game_end_index]['score']
        return [env.last_game_reward, env.game_score, games_played]
