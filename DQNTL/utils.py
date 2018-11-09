"""
Core classes.

"""

import numpy as np


class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    def __init__(self, state, action, reward, next_state, terminal):
        self.sample = (state, action, reward, next_state, terminal)


class Preprocessor:
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. You may
    implement any of these functions. Feel free to add or change the
    interface to suit your needs.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.

        Should be called just before the action is selected.

        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in anyway.

        """
        return state

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.

        """
        return state

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass


class ReplayMemory:
    """Interface for replay memories.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw samples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size, window_length, state_input, stride=0):
        """Setup memory.

        You should specify the maximum size of the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.
        """
        
        self.max_size = max_size
        # ring buffer
        self.buffer = [None] * self.max_size
        self.size = 0
        self.start_index = 0
        self.window_length = window_length # number of consecutive windows for input
        self.state_input = state_input
        self.stride = stride
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        if index < 0 or index >= self.size:
            raise KeyError('Invalid Key')
        return self.buffer[(self.start_index + index) % self.max_size]
        
    def __iter__(self):
        for i in range(self.size):
            yield self.buffer[(self.start_index + i) % self.max_size]
    
    def append(self, state, action, reward, next_state, terminal):
        if self.size < self.max_size:
            self.size += 1
        else:
            self.start_index = (self.start_index + 1) % self.max_size
        s = Sample(state, action, reward, next_state, terminal)
        self.buffer[(self.start_index + self.size - 1) % self.max_size] = s.sample

    def subsample(self, batch_size, indexes=None):
        """ 
        subsample a batch from replay memory 
        e.g. window_length = 3, stride = 1
        0 1 2 3 4
        samples O-O-O, so
        024 is the states representation                
        """

        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []
        batch_terminal = []

        # total size of buffer size required to get subsample of size window length with subsample rate (stride is skip)
        sample_size = (self.window_length - 1) * (self.stride + 1) + 1

        #buffer_start = sample_size - 1
        for i in range(batch_size):
            if self.size < self.max_size:
                sample_start_index = sample_size - 1
            else:
                # if the buffer is full, can wrap around
                sample_start_index = 0
            idx = np.random.randint(sample_start_index, self.size)
            # deal with discontinuity at the start index
            while self.start_index <= idx < (self.start_index + sample_size - 1):
                idx = np.random.randint(sample_start_index, self.size)

            state_frame = np.zeros((self.window_length,) + self.state_input)
            next_state_frame = np.zeros((self.window_length,) + self.state_input)
            buffer_index = sample_size - 1
            for j in range(self.window_length - 1, -1, -1):
                # if among window frames, there is a terminal, append zero
                if j != self.window_length - 1 and self.buffer[idx - sample_size + buffer_index + 1][4] == True:
                    while j >= 0:
                        state_frame[j] = np.zeros(self.state_input)
                        next_state_frame[j] = np.zeros(self.state_input)
                        j -= 1
                    break
                else:
                    state_frame[j] = self.buffer[idx - sample_size + buffer_index + 1][0]
                    next_state_frame[j] = self.buffer[idx - sample_size + buffer_index + 1][3]
                buffer_index -= self.stride + 1

            batch_state.append(state_frame)
            batch_next_state.append(next_state_frame)
            batch_action.append(self.buffer[idx][1])
            batch_reward.append(self.buffer[idx][2])
            batch_terminal.append(self.buffer[idx][4])

        terminal_batch = 1.0 - (np.array(batch_terminal) + 0.0)  # True to be 0 and False to be 1
        # print batch_state

        return np.array(batch_state), np.array(batch_action), np.array(batch_reward), np.array(
            batch_next_state), terminal_batch

    def sample(self, batch_size, indexes=None):
        """ sample a batch from replay memory """
        #print self.buffer[0][0]
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []
        batch_terminal = []
        #print 'inside buffer'
        #print self.buffer[0][0].shape
        for i in range(batch_size):
            idx = np.random.randint(self.window_length - 1, self.size)#self.start_index
            #print idx
            #print self.size
            # deal with discontinuity at the start index
            while self.start_index <= idx < (self.start_index + self.window_length - 1):
                idx = np.random.randint(self.window_length - 1, self.size)
            
            state_frame = np.zeros((self.window_length,) + self.state_input)
            next_state_frame = np.zeros((self.window_length,) + self.state_input)
            for j in range(self.window_length-1, -1, -1):
                # if among window frames, there is a terminal, append zero
                if j != self.window_length-1 and self.buffer[idx - self.window_length + j + 1][4] == True:
                    while j >= 0:
                        state_frame[j] = np.zeros(self.state_input)
                        next_state_frame[j] = np.zeros(self.state_input)
                        j -= 1
                    break
                else:
                    state_frame[j] = self.buffer[idx - self.window_length + j + 1][0]
                    next_state_frame[j] = self.buffer[idx - self.window_length + j + 1][3]

            batch_state.append(state_frame)
            batch_next_state.append(next_state_frame)
            batch_action.append(self.buffer[idx][1])
            batch_reward.append(self.buffer[idx][2])
            batch_terminal.append(self.buffer[idx][4])
        
        terminal_batch = 1.0 - (np.array(batch_terminal) + 0.0)  # True to be 0 and False to be 1
        #print batch_state

        return np.array(batch_state), np.array(batch_action), np.array(batch_reward), np.array(batch_next_state), terminal_batch
          
    def clear(self):
        self.buffer = [None] * self.max_size

