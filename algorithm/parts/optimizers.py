# analytical tools
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

optimizers = {}

def register(opt_name):
    def _map(func):
        optimizers[opt_name] = func
    return _map

@register('adam')
def adam():
    def opt_fn(neglog_params, values, entropies, **kwargs):
        '''
        This is the PPO Loss Function from the Schulman et al paper with
        the addition of value loss clipping.
        '''
        learning_rate = kwargs.get('policy/pl_lr')
        grad_clip = kwargs.get('policy/pl_gc')
        c1 = kwargs.get('policy/pl_c1')
        c2 = kwargs.get('policy/pl_c2')
        eps = kwargs.get('policy/pl_eps')
        actions = kwargs.get('policy/pl_act')
        returns = kwargs.get('policy/pl_ret')
        advantages = kwargs.get('policy/pl_adv')
        old_A = kwargs.get('policy/pl_oldp')
        old_V = kwargs.get('policy/pl_oldv')

        with tf.name_scope('optimizer'):
            # Clip ratio to calc policy loss
            pr = tf.exp(old_A - neglog_params)
            pr_clipped = tf.clip_by_value(pr, 1-eps, 1+eps)
            policy_loss = tf.reduce_mean(tf.maximum(-pr * advantages,
                                                    -pr_clipped * advantages))
            approx_kldiv = 0.5 * tf.reduce_mean(tf.square(neglog_params - old_A))
            clip_frac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(pr - 1.0), eps)))
            # Clip value to calc value loss
            v_clip = old_V + tf.clip_by_value(values - old_V, -eps, eps)
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(tf.square(values - returns), 
                                                         tf.square(v_clip - returns)))
            # Calculate the entropy bonus
            entropy = tf.reduce_mean(entropies)
            # sum loss components
            L = policy_loss + c1 * value_loss - c2 * entropy

            # train
            params = tf.trainable_variables('policy')
            adam = tf.train.AdamOptimizer(learning_rate, epsilon=1e-5)
            grads_and_var = adam.compute_gradients(L, params)
            grads, var = zip(*grads_and_var)
            clipped_grads, _grad_norm = tf.clip_by_global_norm(grads, grad_clip)
            grads_and_var = list(zip(clipped_grads, var))
            trainer = adam.apply_gradients(grads_and_var)
        return [trainer, policy_loss, value_loss, entropy, approx_kldiv, clip_frac]
    return opt_fn

def build_optimizer(name):
    if name in optimizers:
        return optimizers[name]
    else:
        raise ValueError('{} is not a valid optimizer'.format(name))
