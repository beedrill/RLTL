#!/usr/bin/env python
"""
test memory buffer
"""

from DQNTL.utils import ReplayMemory
import numpy as np

# memory
memory = ReplayMemory(max_size=20, window_length=5, stride=1, state_input=(2,2))

iteration = 25
batch_size = 2

for i in range(iteration):
    state = np.full((2,2), i)
    action = i
    reward = i
    next_state = np.full((2,2), i+1)
    terminal = True if i == iteration - 1 else False
    #terminal = True if i % 3 == 0 else False

    # add sample to replay memory
    memory.append(
        state,
        action,
        reward,
        next_state,
        terminal)

# sample
for _ in range(10):
    batch_state, batch_action, batch_reward, batch_next_state, batch_terminal = memory.subsample(batch_size)
    print 'Sampled batch state'
    print batch_state


# for element in memory:
#     print(element)
# print 'start index: ', memory.start_index