# python libraries
from argparse import ArgumentParser
from collections import namedtuple
import os

# local dependencies
from algorithm.model import Model

PolicySettings = namedtuple('PolicySettings', '''shared_network actor_network
                                                 critic_network''')
RuntimeSettings = namedtuple('RuntimeSettings', '''mode num_batches num_test_games
                                                   restore_path save_path
                                                   save_every log_every tb_path''')
Hyperparams = namedtuple('Hyperparams', '''learning_rate env_steps num_envs batch_updates
                                            gamma lam grad_clip c1 c2 eps''')

def main(env_id, runtime_settings, policy_settings, hyperparams):
    '''
    Initializes model, loads params if necessary, then trains or tests as specified.
    '''
    # create models and tensorboard directories, if necessary
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./tbs'):
        os.makedirs('./tbs')
    model = Model(env_id, policy_settings, hyperparams)
    if runtime_settings.restore_path:
        model.saver.restore(model.sess, runtime_settings.restore_path)
        print('Checkpoint restored from {}'.format(runtime_settings.restore_path))
    if runtime_settings.mode == 'train':
        model.train(runtime_settings.num_batches, runtime_settings.log_every,
                    runtime_settings.save_every, runtime_settings.save_path,
                    runtime_settings.tb_path)
    elif runtime_settings.mode == 'test':
        model.test(runtime_settings.num_test_games)

def parse_args(args):
    '''
    Maps arguments to data structures.
    '''
    env_id = args.env_id

    runtime_settings = RuntimeSettings(args.mode, args.num_batches, args.num_test_games,
                                       args.restore_path, args.save_path, args.save_every,
                                       args.log_every, args.tb_path)

    policy_settings = PolicySettings(args.shared_network, args.actor_network,
                                     args.critic_network)

    # note that c1, c2 will only be used in a2c and ppo and epsilon is only used in ppo
    # their values will be None, if they are not used
    hyperparams = Hyperparams(args.learning_rate, args.env_steps, args.num_envs,
                              args.batch_updates, args.gamma, args.lam, args.grad_clip,
                              args.c1, args.c2, args.eps)

    if runtime_settings.mode == 'train':
        print('''\nRUNTIME SETTINGS
        \n mode: {}
        \n num batches: {}
        \n restore path: {}
        \n save path: {}
        \n save every: {}
        \n log every: {}
        \n tb path: {} \n'''\
        .format(runtime_settings.mode, runtime_settings.num_batches,
                runtime_settings.restore_path, runtime_settings.save_path,
                runtime_settings.save_every, runtime_settings.log_every,
                runtime_settings.tb_path))
    elif runtime_settings.mode == 'test':
        print('''\nRUNTIME SETTINGS
        \n mode: {}
        \n num test games: {} \n'''\
        .format(runtime_settings.mode, runtime_settings.num_test_games))

    print('''POLICY SETTINGS
            \n shared net: {}
            \n actor net: {}
            \n critic net: {} \n'''\
            .format(policy_settings.shared_network, policy_settings.actor_network,
                    policy_settings.critic_network))

    print('''HYPERPARAMS
             \n learning rate: {}
             \n grad clipping: {} 
             \n env steps: {}
             \n num envs: {}
             \n batch updates: {}
             \n gamma: {}
             \n lambda: {}
             \n c1: {}
             \n c2: {}
             \n eps: {} \n'''
             .format(hyperparams.learning_rate, hyperparams.grad_clip,
                     hyperparams.env_steps, hyperparams.num_envs,
                     hyperparams.batch_updates, hyperparams.gamma, hyperparams.lam,
                     hyperparams.c1, hyperparams.c2, hyperparams.eps))

    return env_id, runtime_settings, policy_settings, hyperparams

if __name__ == '__main__':
    parser = ArgumentParser()
    # environment id which defines reward shaping
    parser.add_argument('--env-id', default='CartPole-v1',
                        help='Type of env. Check snek.py file for all options.')

    # policy settings
    parser.add_argument('--shared-network', default='fc3',
                        help='Type of network. Options can be found in networks.py')
    parser.add_argument('--actor-network', default='actor_fc1',
                        help='Type of network. Options can be found in networks.py')
    parser.add_argument('--critic-network', default='critic_fc1',
                        help='Type of network. Options can be found in networks.py')

    # runtime settings
    parser.add_argument('--mode', default='train',
                        help='Define whether to train or test the model.')
    parser.add_argument('--num-batches', default=1000, type=int,
                        help='Num batches to train for.')
    parser.add_argument('--num-test-games', default=10, type=int,
                        help='Number of games to test for.')
    parser.add_argument('--restore-path', default=None,
                        help='''Path from which to restore a model from.
                                Must specify this when in test mode.''')
    parser.add_argument('--save-path', default=None,
                        help='Save path for model.')
    parser.add_argument('--save-every', default=100, type=int,
                        help='Frequency of which to save model - in units of batches.')
    parser.add_argument('--log-every', default=10, type=int,
                        help='How often to log training stats to console.')
    parser.add_argument('--tb-path', default=None, help='Path to save tensorboard.')

    # hyperparameters
    parser.add_argument('--learning-rate', default=1e-4,
                        help='Value for learning rate. Set to a lambda to anneal.')
    parser.add_argument('--grad-clip', default=0.5, type=float,
                        help='Value at which to clip the gradient.')
    parser.add_argument('--env-steps', default=32, type=int,
                        help='''Steps to take in each environment.
                        Batch size will be env-steps * num-envs.''')
    parser.add_argument('--num-envs', default=32, type=int,
                        help='''Number of envs to concurrently train on.''')
    parser.add_argument('--batch-updates', default=4, type=int,
                        help='''Number of envs to concurrently train on.''')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='Discount factor.')
    parser.add_argument('--lam', default=0.95, type=float,
                        help='''Lambda - Generalized Advantage Estimation parameter.''')
    parser.add_argument('--c1', default=0.5, type=float,
                        help='Coefficient for critic loss term.')
    parser.add_argument('--c2', default=0.01, type=float,
                        help='Coefficient for entropy loss term.')
    parser.add_argument('--eps', default=0.2, type=float,
                        help='Clipping hyperparameter.')

    args = parser.parse_args()
    env_id, runtime_settings, policy_settings, hyperparams = parse_args(args)
    main(env_id, runtime_settings, policy_settings, hyperparams)
