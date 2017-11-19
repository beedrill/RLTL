#!/usr/bin/env python
"""
Main DQN agent.
"""

import tensorflow as tf
from keras.models import Model
from collections import deque
import numpy as np
from DQNTL.policy import UniformRandomPolicy, GreedyEpsilonPolicy
from keras.layers import Input, Lambda, Dense, concatenate
import keras.backend as K
# import gym
import time
# from PIL import Image


class DQNAgent:
    """Class implementing DQN.

    Parameters
    ----------
    model: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    window_length: int
        number of frames for a state
    start_random_steps: int
        number of random steps at the beggining
    num_actions: int
        number of possible actions
    env_name: string
        name of the environment
    network: string
        type of network; DQN, doubleDQN, duelDQN

    """
    def __init__(self,
                 model,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 window_length,
                 start_random_steps,
                 num_actions,
                 env_name,
                 network,
                 input_shape = (1,9)):
                    
        self.model = model
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.window_length = window_length
        self.start_random_steps = start_random_steps
        self.num_actions = num_actions
        self.env_name = env_name
        self.network = network
        self.input_shape = input_shape        

        # store recent states for selecting action according to the current state
        self.recent_states = deque(maxlen=self.window_length) # store in float32
        self.recent_states_test = deque(maxlen=self.window_length) # store in float32
        
        self.uniform_policy = UniformRandomPolicy(self.num_actions)
        self.test_policy = GreedyEpsilonPolicy(epsilon=0.05, num_actions=self.num_actions)
        
        # counters
        self.steps = 0  # number of total steps
        self.episodes = 0  # number of episodes
        self.episode_steps = 0  # number of steps per episode
        # parameter
        # TODO evaluation
        self.evaluation_interval = 100000  # interval to run evaluation and get average reward
        self.log_interval = 100  # interval of logging data

    def compile(self, optimizer, loss_func, metrics):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        # change the model for dueling DQN
        # create fully-connected layer for value and advantange estimation, 
        # and combine both with average method
        # if self.network == 'duelDQN':
        #     with tf.name_scope('V'):
        #         V = Dense(512, activation='relu')(self.model.layers[-3].output)
        #     with tf.name_scope('A'):
        #         A = Dense(512, activation='relu')(self.model.layers[-3].output)
        #     with tf.name_scope('Vout'):
        #         Vout = Dense(1, activation='linear')(V)
        #     with tf.name_scope('Aout'):
        #         Aout = Dense(self.num_actions, activation='linear')(A)
        #     with tf.name_scope('concatenate'):
        #         con = concatenate([Vout, Aout])
        #     with tf.name_scope('combine'):
        #         output = Lambda(lambda x : K.expand_dims(x[: , 0], -1) + x[:, 1:] - K.mean(x[: , 1:], keepdims=True), output_shape=(self.num_actions, ))(con)
        #     self.model = Model(input=self.model.input, output=output)

        # get target model
        config = self.model.get_config()
        with tf.name_scope('target_model'):
            self.target_model = Model.from_config(config)
        self.target_model.set_weights(self.model.get_weights()) # copy weights
        
        # compile online and target models		
        self.model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics)
        self.target_model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
            
        # loss
        def custom_loss(args):
            y_true, y_pred, action_mask = args
            loss = loss_func(y_true*action_mask, y_pred*action_mask)
            return loss
        
        losses = [lambda y_true, y_pred: y_pred, lambda y_true, y_pred: tf.zeros_like(y_pred),]
        
        # inputs to the custom loss layer
        with tf.name_scope('y_pred'):
            y_pred = self.model.output
        y_true = Input(shape=(self.num_actions, ), name='y_true')
        action_mask = Input(shape=(self.num_actions, ), name='action_mask')
        
        # custom loss layer for updating loss only for a specific action
        loss_out = Lambda(custom_loss, output_shape=(1, ), name='loss_layer')([y_pred, y_true, action_mask])
        
        # model for training
        model_train = Model(input=[self.model.input, y_true, action_mask], output=[loss_out, y_pred])
        
        # compile model	
        model_train.compile(loss=losses, optimizer=optimizer, metrics=metrics)
        self.model_train = model_train
        
        # print out summary
        print 'model summary'
        print(self.model.summary())
        print 'target model summary'
        print(self.target_model.summary())

    def calc_q_values_model(self, states):
        """Given a state (or batch of states) calculate the Q-values from model

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        q_values = self.model.predict_on_batch(states)
        return q_values

    def select_action(self, mode='train', **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        if mode == 'test':
            states = list(self.recent_states_test)
            # if not enough states, append zero
            # note that at the begging of each episode buffer is cleared, thus no need to take care of states across episodes
            while len(states) < self.window_length:
                states.insert(0, np.zeros(states[0].shape))

            states = self.preprocessor.process_batch(states)
            q_values = self.calc_q_values_model(np.array([states])).flatten()
            action = self.test_policy.select_action(q_values)
        else:
            if self.steps < self.num_burn_in:
                action = self.uniform_policy.select_action()
            else:
                states = list(self.recent_states)
                # if not enough states, append 0
                if not type(states[0])==np.ndarray or np.isnan(states).any():
                #if np.isnan(states[0]):
                    print 'state is nan'
                    print type(states[0])
                    states = [np.zeros(self.input_shape)]

                try:
                    while len(states) < self.window_length:
                        states.insert(0, np.zeros(self.input_shape))
                    # print 'state', states[0].shape
                    states1 = self.preprocessor.process_batch(states)
                    # print 'states', states.shape, np.array([states]).shape
                    q_values = self.calc_q_values_model(np.array([states1])).flatten()
                    action = self.policy.select_action(q_values)
                except:
                    print states
                    print states1
                    #time.sleep()
                    #print len(states)


        return action
        
    def select_greedy_actions(self, batch_state):
        """ 
        select greedy action to according to the state input 
        mainly used for double DQN
        """
        q_values = self.calc_q_values_model(batch_state)
        actions = np.argmin(q_values,1).flatten()
        return actions

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        huber_loss = None
        mae_metric = None
        
        if self.steps > self.num_burn_in and self.steps % self.train_freq == 0:
            # sample a mini batch
            batch_state, batch_action, batch_reward, batch_next_state, batch_terminal = self.memory.sample(self.batch_size)
            #print 'state', batch_state
            batch_state = self.preprocessor.process_batch(batch_state)
            batch_next_state = self.preprocessor.process_batch(batch_next_state)
            # compute target q values
            target_q_values = self.target_model.predict_on_batch(batch_next_state) # return 32x6
            #print target_q_values

            if self.network == 'DQN' or self.network == 'duelDQN':
                target_q_values = np.min(target_q_values, 1).flatten()
            
            elif self.network == 'doubleDQN':
                actions = self.select_greedy_actions(batch_next_state)
                target_q_values = target_q_values[range(self.batch_size), actions]

            # target discounted reward
            #print 'batch terminal',batch_terminal
            target_reward = self.gamma*target_q_values*batch_terminal + batch_reward
            
            target = np.zeros((self.batch_size, self.num_actions))
            dummy_target = np.zeros((self.batch_size, ))
            action_mask = np.zeros((self.batch_size, self.num_actions))

            for idx, (each_target, each_mask, each_reward, each_action) in enumerate(zip(target, action_mask, target_reward, batch_action)):
                each_target[each_action] = each_reward
                dummy_target[idx] = each_reward
                each_mask[each_action] = 1.0
            

            # update model
            loss_metric= self.model_train.train_on_batch([batch_state, np.array(target, dtype='float32'), np.array(action_mask, dtype='float32')], [dummy_target, np.array(target, dtype='float32')])
            huber_loss = loss_metric[0]
            mae_metric = loss_metric[3]
            
        # update target model
        if self.steps % self.target_update_freq == 0:
            self.hard_target_model_updates()
        
        return huber_loss, mae_metric
    
    def reset_environment(self, env):
        """ 
            reset environment 
            initialize the game and run random actions specified by 
            random number between 0 to start_random_steps
        """		
        
        env.reset()
        state = None
        for _ in range(np.random.randint(self.start_random_steps)):
            action = env.action_space.sample() # sample random action
            next_state, _, _, _ = env.step(action)
            next_state = next_state[0]
            next_state = np.expand_dims(next_state, axis=0)
            state = next_state
        
        print state
        return state
        
    def log_tb_value(self, name, value):
        """ 
            helper function for logging files
        """
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        return summary        

    def fit(self, env, env_eval, num_iterations, save_interval, writer, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: Simulator()
          This is pysumo environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        env_eval: Simulator()
          This is test env
        num_iterations: int
          How many samples/updates to perform.
         save_interval: int
            how frequently to save the model
         writer: tf.summary.FileWriter
            writer for logging data
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
    
        state = None
        self.steps = 0
        self.episodes = 0
        self.episode_steps = 0

        test_eval_steps = 1

        while self.steps < num_iterations:
            
            if state is None: # beginning of an episode
                state = self.reset_environment(env)
                self.recent_states.clear() # reset the recent states buffer
                
                # TODO debug
                states = list(self.recent_states)
                if len(states) != 0  and (not type(states[0])==np.ndarray or np.isnan(states).any()):
                    print 'state is nan after clear'
                    print type(states[0])
    
                self.episode_steps = 0
                episode_reward = 0
                # add states to recent states
                self.recent_states.append(self.preprocessor.process_state_for_memory(state))
                # TODO debug
                states = list(self.recent_states)
                if len(states) != 0  and (not type(states[0])==np.ndarray or np.isnan(states).any()):
                    print 'state is nan after append from reset'
                    print type(states[0])

            # select action 
            action = self.select_action()
            action = [action]
            next_state, reward, terminal, _ = env.step(action)
            next_state = next_state[0]
            next_state = np.expand_dims(next_state, axis=0)
            reward = reward[0]
            #print type(next_state)
            if type(next_state) is not np.ndarray:
                print 'wrong'
            elif np.isnan(next_state).any():
                print 'some nan in state'
            
            # add sample to replay memory
            self.memory.append(
                self.preprocessor.process_state_for_memory(state),
                action,
                self.preprocessor.process_reward(reward),
                self.preprocessor.process_state_for_memory(next_state),
                terminal)
            
            state = next_state
            
            # add states to recent states
            self.recent_states.append(self.preprocessor.process_state_for_memory(state))
            # TODO debug
            states = list(self.recent_states)
            if len(states) != 0  and (not type(states[0])==np.ndarray or np.isnan(states).any()):
                print 'state is nan after append from step'
                print type(states[0])

            # update policy -- update Q network and update target network
            huber_loss, mae_metric = self.update_policy()
            
            episode_reward += reward
            # print 'steps: {} loss: {}'.format(self.steps, huber_loss)
            # log info
            if self.steps % self.log_interval == 0 and huber_loss and mae_metric:
                writer.add_summary(self.log_tb_value('loss', huber_loss), self.steps)
                writer.add_summary(self.log_tb_value('mae_metric', mae_metric), self.steps)
                writer.add_summary(self.log_tb_value('reward', reward), self.steps)
                
            # save weights
            if self.steps % save_interval == 0:
                file_name = '{}_{}_{}_weights.hdf5'.format(self.network, self.env_name, self.steps)
                file_path = 'weights/' + file_name
                self.model.save_weights(file_path)
                
            # evaluation	
# =============================================================================
#             if self.steps > self.num_burn_in and self.steps % self.evaluation_interval == 0:
#                 # env_eval = gym.make(self.env_name)
#                 avg_reward = self.evaluate(env_eval, test_eval_steps)
#                 print 'steps: {}, average reward: {}'.format(self.steps, avg_reward)
#                 writer.add_summary(self.log_tb_value('performance', avg_reward), self.steps)	
#                 env_eval.close()
# =============================================================================
                
            # episode terminal condition
            if terminal or (max_episode_length and self.episode_steps % max_episode_length == 0):
                print 'episode {} reward {}'.format(self.episodes, episode_reward)
                self.episodes += 1
                state = None
                writer.add_summary(self.log_tb_value('episode_reward', episode_reward), self.episodes)
                writer.add_summary(self.log_tb_value('episode_length', self.episode_steps), self.episodes)
                
            # counter update
            self.steps += 1
            self.episode_steps += 1
        
        # evaluate performance
        # env_eval = gym.make(self.env_name)
        avg_reward = self.evaluate(env_eval, test_eval_steps)
        print 'steps: {}, average reward: {}'.format(self.steps, avg_reward)
        writer.add_summary(self.log_tb_value('performance', avg_reward), self.steps)	
        # env_eval.close()
        env.stop()

    def evaluate(self, env, num_episodes, render=False, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        cumulative_reward = 0.0
        # env.start()
        
        for episode in range(num_episodes):

            test_episode_steps = 0
            state = self.reset_environment(env)
            self.recent_states_test.clear()
            if render:
                env.render()
            # add states to recent states for test
            self.recent_states_test.append(self.preprocessor.process_state_for_memory(state))

            while True:
                action = self.select_action('test')
                action = [action]
                next_state, reward, terminal, _ = env.step(action)
                next_state = next_state[0]
                next_state = np.expand_dims(next_state, axis=0)
                reward = reward[0]
                if render:
                    env.render()
                state = next_state
                self.recent_states_test.append(self.preprocessor.process_state_for_memory(state))
                cumulative_reward += reward

                # episode terminal condition
                if terminal or (max_episode_length and test_episode_steps % max_episode_length == 0):
                        break

                test_episode_steps += 1
                
        avg_total_reward = float(cumulative_reward)/num_episodes
        return avg_total_reward 		       	

    def hard_target_model_updates(self):
        """
        These are hard target updates. The source weights are copied
        directly to the target network.
        """
        self.target_model.set_weights(self.model.get_weights())

