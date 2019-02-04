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
import numpy as np
import sys

from DQNTL.dqn import DQNAgent, DQNAgents
from DQNTL.preprocessors import TLStatePreprocessor, TLMAPPreprocessor
from DQNTL.utils import ReplayMemory
from DQNTL.policy import GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy
from DQNTL.objectives import mean_huber_loss

from simulator import Simulator, TrafficLightLuxembourg, SimpleTrafficLight

import gym
import gym_trafficlight
from gym_trafficlight.wrappers import  TrafficParameterSetWrapper
from gym_trafficlight import TrafficEnv, PenetrationRateManager
#from simulator_osm import Simulator as Simulator_osm

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

def process_observation(obs):
    occupations, feature = obs
    result = np.concatenate(occupations)
    result = np.concatenate([result, np.array(feature)])
    return result

def create_model(window, input_shape, num_actions,
                 model_name='q_network', input_type = 'simple', num_lanes = 4, num_feature = 10):
    if input_type == 'simple':
        return create_model_simple(window, input_shape, num_actions, model_name= model_name)
    elif input_type == 'full':
        feature_input_shape = input_shape
        return create_model_full(window, input_shape, num_actions, model_name=model_name,  num_lanes = num_lanes, num_feature = num_feature)

def create_model_full(window, input_shape, num_actions,
                 model_name='q_network', num_lanes = 4, num_features = 10, lane_length = 125):
    input = [Input(shape=lane_input_shape) for _ in range(num_lanes)]
    input.append(Input(shape=feature_input_shape))


def create_model_simple(window, input_shape, num_actions,
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
    #fc = fc_in
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
    parser.add_argument('--record', action = 'store_true', help='record results when loading.')
    # parser.add_argument('--render', action='store_true', default=False, help='render while testing')
    parser.add_argument('--pysumo', action='store_true', help='use pysumo')
    parser.add_argument('--dir_name', default = 'TL',help = 'directory name')
    parser.add_argument('--no_counter', action = 'store_true', default = False, help = 'no counter in saving files, note it might overwrite previous results')
    parser.add_argument('--penetration_rate', type = float, default = 1., help = 'specify penetration rate')
    parser.add_argument('--sumo', action='store_true', help='force to use non-gui sumo')
    parser.add_argument('--whole_day', action = 'store_true', help = 'specify the time of the day when training')
    parser.add_argument('--day_time', type = int, help = 'specify day time')
    parser.add_argument('--phase_representation', default = 'original', help = 'specify representation')
    parser.add_argument('--shared_model', action='store_true', help='use a common model between all agents')
    parser.add_argument('--simulator', choices=['original', 'osm'], default='original')
    parser.add_argument('--simple_inputs', action='store_true', help='use simplified inputs with fixed number of states (12)')
    parser.add_argument('--map', choices=['osm_3_intersections', 'osm_13_intersections', 'manhattan_small','manhattan'], default='osm_13_intersections')
    #parser.add_argument('--route', choices=['osm_3_intersections', 'osm_13_intersections', 'manhattan_small','manhattan'], default='osm_13_intersections')
    #parser.add_argument('--aggregated_reward', action='store_true', help='choose to combine waiting times to optimize waiting time on entire network instead of individually at each TL')
    parser.add_argument('--arrival_rate', default='1', help='arrival rate of cars')
    parser.add_argument('--unstationary_flow', action = 'store_true', help='use when the training flow is unstationary')
    parser.add_argument('--dynamic_penetration', action = 'store_true', help='dynamic penetration rate')
    parser.add_argument('--reward_type', default = 'local', help='dynamic penetration rate')
    parser.add_argument('--evaluation_interval', default = 5, type = int, help='how many episodes per evaluation')
    parser.add_argument('--env', default = 'TrafficLight-simple-medium-v0', help='env name')
    parser.add_argument('--test_while_learn', action = 'store_true', help='set agent mode to test while learn, this will disable epsilon greedy')
    parser.add_argument('--log_waiting_time', action='store_true', help='set env def logger on')

    args = parser.parse_args()

    ## PARAMS ##

    num_episodes = 150
    episode_time = 3000  # must be less than 3600
    num_iterations = num_episodes * episode_time
    memory_size = 200000
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
        import libsumo
        visual = False
    else:
        import traci
        visual = True
    #
    # if args.simulator == 'original':
    #     env = Simulator(visual=visual,
    #                     episode_time=episode_time,
    #                     num_traffic_state = 26,
    #                     penetration_rate = args.penetration_rate,
    #                     #config_file= './map/OneIntersectionLuSTScenario-12408/traffic.sumocfg',
    #                     #standard_file_name ='./map/OneIntersectionLuSTScenario-12408/traffic-standard.rou.xml',
    #                     map_file='./map/OneIntersectionLuST-12408-stationary/8/traffic.net.xml',
    #                     route_file='./map/OneIntersectionLuST-12408-stationary/8/traffic.rou.xml',
    #                     #map_file='./map/LuxembougDetailed-DUE-12408/traffic.net.xml',
    #                     #route_file='./map/LuxembougDetailed-DUE-12408/traffic-8.rou.xml',
    #                     whole_day = args.whole_day,
    #                     flow_manager_file_prefix='./map/LuxembougDetailed-DUE-12408/traffic',
    #                     state_representation = args.phase_representation,
    #                     unstationary_flow = args.unstationary_flow,
    #                     traffic_light_module = TrafficLightLuxembourg,
    #                     tl_list = ['0'],
    #                     force_sumo = args.sumo)
    # elif args.simulator == 'osm':
    #     env = Simulator_osm(visual=visual,
    #                     episode_time=episode_time,
    #                     penetration_rate = args.penetration_rate,
    #                     map_file= args.map,
    #                     arrival_rate= args.arrival_rate,
    #                     simple = args.simple_inputs,
    #                     aggregated_reward = args.aggregated_reward)

    env = gym.make(args.env)
    #env = gym.make('TrafficLight-Lust12408-regular-time-v0')
    #env_args = TrafficEnv.get_default_init_parameters()
    env_args    = {
        'visual':                 visual,
        'episode_time':           episode_time,
        #'num_traffic_state':      10,
        'penetration_rate':       args.penetration_rate,
        #config_file= './map/OneIntersectionLuSTScenario-12408/traffic.sumocfg',
        #standard_file_name ='./map/OneIntersectionLuSTScenario-12408/traffic-standard.rou.xml',
        #map_file='./map/OneIntersectionLuST-12408-stationary/8/traffic.net.xml',
        #route_file='./map/OneIntersectionLuST-12408-stationary/8/traffic.rou.xml',
        #map_file='./map/LuxembougDetailed-DUE-12408/traffic.net.xml',
        #route_file='./map/LuxembougDetailed-DUE-12408/traffic-8.rou.xml',
        'whole_day':              args.whole_day,
        #flow_manager_file_prefix:'./map/LuxembougDetailed-DUE-12408/traffic',
        #state_representation:   args.phase_representation,
        #unstationary_flow:      args.unstationary_flow,
        #traffic_light_module:   TrafficLightLuxembourg,
        #tl_list:                ['0'],
        'force_sumo':             args.sumo,
        'reward_type':            args.reward_type,
        'reward_present_form':    'penalty',
        'log_waiting_time':       args.log_waiting_time
        #'observation_processor':  process_observation
    }
    if args.dynamic_penetration:
        prm = PenetrationRateManager(
              trend = 'linear',
              transition_time = 3*365, #3 years
              pr_start = 0.1,
              pr_end = 1
              )
        env_args['reset_manager'] = prm
        args.test_while_learn = True ## in dynamic penetration rate, we normally want to disable epsilon greedy, so we do it in default here
        args.evaluation_interval = sys.maxsize
    # env_args    = {
    #     'visual':                 visual,
    #     'episode_time':           episode_time,
    #     #'num_traffic_state':      10,
    #     'penetration_rate':       args.penetration_rate,
    #     #config_file= './map/OneIntersectionLuSTScenario-12408/traffic.sumocfg',
    #     #standard_file_name ='./map/OneIntersectionLuSTScenario-12408/traffic-standard.rou.xml',
    #     #map_file='./map/OneIntersectionLuST-12408-stationary/8/traffic.net.xml',
    #     #route_file='./map/OneIntersectionLuST-12408-stationary/8/traffic.rou.xml',
    #     #map_file='./map/LuxembougDetailed-DUE-12408/traffic.net.xml',
    #     #route_file='./map/LuxembougDetailed-DUE-12408/traffic-8.rou.xml',
    #     'whole_day':              args.whole_day,
    #     #flow_manager_file_prefix:'./map/LuxembougDetailed-DUE-12408/traffic',
    #     #state_representation:   args.phase_representation,
    #     #unstationary_flow:      args.unstationary_flow,
    #     #traffic_light_module:   TrafficLightLuxembourg,
    #     #tl_list:                ['0'],
    #     'force_sumo':             args.sumo,
    #     'reward_type':            args.reward_type,
    #     'reward_present_form':    'penalty',
    #     #'observation_processor':  process_observation
    # }
    ##This wrapper is used to pass the parameter into the env, the wrapper will re-init the whole\
    ## env with the new parameters, so by wrapping it and unwrap it, we get a new #!/usr/bin/env python
    ## This is obviously not the best way to init env, TODO: more intuitional way to init env
    env = TrafficParameterSetWrapper(env, env_args).unwrapped
    print(env.num_traffic_state)

    id_list = env.tl_id_list

    num_agents = len(id_list)
    #print num_agents
    #os.system("pause")
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
        print('invalid state')
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
        model = create_model(window=window, input_shape=input_shape, num_actions=num_actions)
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
            if not args.shared_model:
                model = create_model(window=window, input_shape=input_shape, num_actions=num_actions)
            agent = DQNAgent(
                model=model,
                preprocessor=preprocessor,
                memory=memory,
                policy=policy,
                gamma=0.99,
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
                stride=stride,
                test_while_learn = args.test_while_learn)
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
                stride=stride,
                evaluation_interval = args.evaluation_interval)

        if args.load:
            for agent in agents.agents:

                weight_name = args.load + '_' + agent.name + '.hdf5'
                agent.model.load_weights(weight_name)

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
                print('please load a model')
                return
            num_episodes = 5
            if args.whole_day:
                env.flow_manager.travel_to_time(args.day_time)
                num_episodes = 5
                env.reset_to_same_time = True
            avg_reward,overall_waiting_time,equipped_waiting_time,unequipped_waiting_time = agents.evaluate(env=env, num_episodes=num_episodes)
            print('Evaluation Result for average of {} episodes'.format(num_episodes))
            print('average total reward: {} \noverall waiting time: {} \nequipped waiting time: {} \nunequipped waiting time: {}'\
                .format(avg_reward,overall_waiting_time,equipped_waiting_time,unequipped_waiting_time))

            if args.record:
                record_file_name = 'record.txt'
                f = open(record_file_name,'a')
                f.write('{}\t{}\t{}\n'.format(overall_waiting_time,equipped_waiting_time,unequipped_waiting_time))
                f.close()
            env.stop()


if __name__ == '__main__':
    main()