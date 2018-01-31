#!/usr/bin/env python
"""
Run DQN to train multiple traffic lights in SUMO environment.

"""


import argparse
import os

import tensorflow as tf
from keras.layers import (Convolution2D, Dense, Flatten, Input, Permute)
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

from DQNTL.dqn import DQNAgent, DQNAgents
from DQNTL.preprocessors import TLStatePreprocessor, TLMAPPreprocessor
from DQNTL.utils import ReplayMemory
from DQNTL.policy import GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy
from DQNTL.objectives import mean_huber_loss

from simulator import Simulator

import pdb ##TODO delete after debugging


def conv_layers(permute):
    with tf.name_scope('conv1'):
        conv1 = Convolution2D(32, 8, 8, activation='relu', subsample=(4, 4))(permute)
    with tf.name_scope('conv2'):
        conv2 = Convolution2D(64, 4, 4, activation='relu', subsample=(2, 2))(conv1)
    with tf.name_scope('conv3'):
        conv3 = Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1))(conv2)
    with tf.name_scope('conv3_flat'):
        conv3_flat = Flatten()(conv3)

    return conv3_flat


def dense_layers(permute): # TODO number of layers and units
    with tf.name_scope('flat'):
        flat = Flatten()(permute)
    with tf.name_scope('dense1'):
        dense1 = Dense(512, activation='relu')(flat)
    #with tf.name_scope('dense2'):
    #     dense2 = Dense(512, activation='relu')(dense1)
   # with tf.name_scope('dense3'):
   #      dense3 = Dense(512, activation='relu')(dense2)

    return dense1


def create_model(window, input_shape, num_actions,
                 model_name='q_network'):
    """Create the Q-network model.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    input_size = (window, input_shape[0], input_shape[1])

    input = Input(shape=input_size, name='input')
    with tf.name_scope('permute'):
        permute = Permute((2, 3, 1))(input)

    # fc_in = conv_layers(permute)
    fc_in = dense_layers(permute)

    with tf.name_scope('fc'):
        fc = Dense(512, activation='relu')(fc_in)
    with tf.name_scope('output'):
        output = Dense(num_actions, activation='linear')(fc)
        
    with tf.name_scope(model_name):   
        model = Model(input=input, output=output)
    
    return model
        

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    # python 3
    #os.makedirs(parent_dir, exist_ok=True)

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():
    parser = argparse.ArgumentParser(description='Run DQN on Traffic Lights')
    # parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    # parser.add_argument(
    #     '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--gpu', default=0, help='comma separated list of GPU(s) to use.')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    parser.add_argument('--load', help='load model.')
    # parser.add_argument('--render', action='store_true', default=False, help='render while testing')
    parser.add_argument('--pysumo', action='store_true', help='use pysumo')
    parser.add_argument('--dir_name', default = 'TL',help = 'directory name')
    parser.add_argument('--no_counter', action = 'store_true', default = False, help = 'no counter in saving files, note it might overwrite previous results')
    parser.add_argument('--penetration_rate', type = float, default = 1., help = 'specify penetration rate')
    
    args = parser.parse_args()

    ## PARAMS ##

    num_episodes = 150
    episode_time = 3000  # must be less than 3600
    num_iterations = num_episodes * episode_time
    memory_size = 100000
    decay_steps = 100000
    target_update_freq = 3000
    lr = 0.0001
    num_burn_in = 10000
    train_freq = 1
    tl_state = 1
    
    window = 1  # total size of the state
    stride = 0  # stride/skip of states
    #pdb.set_trace()
    if args.pysumo:
        import pysumo
        env = Simulator(episode_time=episode_time,
                        penetration_rate = args.penetration_rate,
                        map_file='map/5-intersections/traffic.net.xml',
                        route_file='map/5-intersections/traffic.rou.xml')
    else:
        import traci
        env = Simulator(visual=True,
                        episode_time=episode_time,
                        penetration_rate = args.penetration_rate,
                        map_file='map/5-intersections/traffic.net.xml',
                        route_file='map/5-intersections/traffic.rou.xml')
        
    id_list = env.tl_id_list
    num_agents = len(id_list)
    #pdb.set_trace() #TODO delete after debugging
    if tl_state == 1:
        input_shape = (1, env.num_traffic_state)
        buffer_input_shape = (num_agents, 1, env.num_traffic_state)
        preprocessor = TLStatePreprocessor()
    elif tl_state == 2:
        input_shape = (4, 250)
        window = 1
        preprocessor = TLMAPPreprocessor()
    else:
        print 'invalid state'
        return

    network = 'DQN'
    
    # choose device
    device = '/gpu:{}'.format(args.gpu)
    if args.cpu:
        device = '/cpu:0'

    env_name = 'SUMO'
    seed = args.seed
    # env.seed(seed)

    num_actions = env.action_space.n
    # print 'num_actions', num_actions
    
    # memory grows as it requires
    #This will assign the computation to CPU automatically whenever GPU is not available    
    config = tf.ConfigProto(allow_soft_placement=True)
    
    #config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    with tf.device(device):
    
        # model
        #model = create_model(window=window, input_shape=input_shape, num_actions=num_actions)
        # memory
        memory = ReplayMemory(max_size=memory_size, window_length=window, stride=stride, state_input=buffer_input_shape)
        # policy
        policy = LinearDecayGreedyEpsilonPolicy(start_value=1.0, end_value=0.05, num_steps=decay_steps, num_actions=num_actions)
        # optimizer
        adam = Adam(lr=lr)
        agent_list = []
        index = 0
        for id in id_list:
            # agent
            model = create_model(window=window, input_shape=input_shape, num_actions=num_actions)
            agent = DQNAgent(
                model=model,
                preprocessor=preprocessor,
                memory=memory,
                policy=policy,
                gamma=0.9,
                target_update_freq=target_update_freq,
                num_burn_in=num_burn_in,
                train_freq=train_freq,
                batch_size=32,
                window_length=window,
                start_random_steps=20,
                num_actions=num_actions,
                env_name=env_name,
                network=network,
                name=id,
                index = index,
                input_shape=input_shape,
                stride=stride)
            index+=1
            
            # compile
            agent.compile(optimizer=adam, loss_func=mean_huber_loss, metrics=['mae'])
            agent_list.append(agent)
        
        agents = DQNAgents(
                agent_list,
                #model=model,
                preprocessor=preprocessor,
                memory=memory,
                #policy=policy,
                #gamma=0.9,
                #target_update_freq=target_update_freq,
                num_burn_in=num_burn_in,
                train_freq=train_freq,
                batch_size=32,
                #window_length=window,
                start_random_steps=20,
                #num_actions=num_actions,
                env_name=env_name,
                network=network,
                #name=id,
                input_shape=input_shape,
                stride=stride)
    
        if args.mode == 'train':
            # log file
            logfile_name = network + '_log/'
            if args.no_counter == False:
                logfile = get_output_folder(logfile_name, args.dir_name)
            else:
                logfile = logfile_name+args.dir_name
            writer = tf.summary.FileWriter(logfile, sess.graph)
            # weights file
            weights_file_name = network + '_weights/'
            if args.no_counter == False:
                weights_file = get_output_folder(weights_file_name, args.dir_name)
            else:
                weights_file = weights_file_name+args.dir_name
            
            os.makedirs(weights_file)
            weights_file += '/'
            
            save_interval = num_iterations / 30  # save model every 1/3
            # print 'start training....'
            agents.fit(env=env, num_iterations=num_iterations, save_interval=save_interval, writer=writer, weights_file=weights_file)
            
            # save weights
            for agent in agents.agents:
                file_name = '{}_{}_{}_weights_{}.hdf5'.format(network, env_name, num_iterations, agent.name)
                file_path = weights_file + file_name
                agent.model.save_weights(file_path)
            
        else:  # test
            if not args.load:
                print 'please load a model'
                return
            for i, agent in enumerate(agents.agents):
                # currently, remove after weights....
                #weight_name = args.load + '_' + str(i) + '.hdf5'
                #weight_name = "DQN_weights/TL-run23/DQN_SUMO_450000_weights_"+agent.name+".hdf5"
                weight_name = "DQN_weights/TL-run25/DQN_SUMO_best_weights_"+agent.name+".hdf5"
                agent.model.load_weights(weight_name)
            
            #print model.layers[3].get_weights()
            #print 'number of layers',len(model.layers)
            num_episodes = 10
            avg_reward,overall_waiting_time,equipped_waiting_time,unequipped_waiting_time = agents.evaluate(env=env, num_episodes=num_episodes)
            print 'Evaluation Result for average of {} episodes'.format(num_episodes)
            print 'average total reward: {} \noverall waiting time: {} \nequipped waiting time: {} \nunequipped waiting time: {}'\
                .format(avg_reward,overall_waiting_time,equipped_waiting_time,unequipped_waiting_time)
            env.stop()
    

if __name__ == '__main__':
    main()
