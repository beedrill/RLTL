#!/usr/bin/env python

"""Preprocessors."""

import numpy as np
from PIL import Image, ImageOps
from DQNTL.utils import Preprocessor


class TLStatePreprocessor(Preprocessor):
    """Process Traffic State Representation

    You may also want to max over frames to remove flickering.

    Parameters
    ----------
    max_value: float
        specifies maximum value used for normalization
    """

    def __init__(self):
        pass

    # TODO
    def process_state_for_memory(self, state):
        """store as float32
        """
        # # convert to uint8
        state_array = np.array(state, dtype=np.float32)
        return state_array

    def process_state_for_network(self, state):
        """store as float32.
        """
        state_array = np.array(state, dtype=np.float32)
        return state_array

    # TODO normalization?
    def process_batch(self, state_batch):
        """

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.

        Parameters
        ----------
        state_batch: np.array
        """
        return np.asarray(state_batch, dtype=np.float32)

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        # TODO clip reward?
        #return np.clip(reward, -1.0, 1.0)
        return reward


class TLMAPPreprocessor(Preprocessor):
    """Process Map with speed of vehicle at each vehicle location.

    You may also want to max over frames to remove flickering.

    Parameters
    ----------
    max_value: float
        specifies maximum value used for normalization
    """

    def __init__(self, max_value=255.0):
        self.max_value = max_value

    def process_state_for_memory(self, state):
        """store as uint8.
        """
        # convert to uint8
        state_array = np.array(state, dtype=np.uint8)
        return state_array

    def process_state_for_network(self, state):
        """store as float32.
        """
        state_array = np.array(state, dtype=np.float32)
        return state_array

    def process_batch(self, state_batch):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.

        Parameters
        ----------
        state_batch: np.array
        """
        return np.asarray(state_batch, dtype=np.float32) / self.max_value

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return np.clip(reward, -1.0, 1.0)


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size=(84, 84), crop_centering=(0.5, 0.5)):
        self.new_size = new_size
        self.crop_centering = crop_centering

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        # convert to PIL image
        img = Image.fromarray(state)
        # scale to 110x84 and convert to grayscale
        img = img.resize((110, 84)).convert('L')
        # crop and get 84x84
        img = ImageOps.fit(img, self.new_size, centering=self.crop_centering)
        # get numpy array
        state_array = np.array(img, dtype=np.uint8)

        return state_array

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        # convert to PIL image
        img = Image.fromarray(state)
        # scale to 110x84 and convert to grayscale
        img = img.resize((110, 84)).convert('L')
        # crop and get 84x84
        img = ImageOps.fit(img, self.new_size, centering=self.crop_centering)
        # get numpy array
        state_array = np.array(img, dtype=np.float32)
        return state_array

    def process_batch(self, state_batch):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.

        Parameters
        ----------
        state_batch: np.array
        """
        return np.asarray(state_batch, dtype=np.float32)/255.0

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return np.clip(reward, -1.0, 1.0)


# class PreprocessorSequence(Preprocessor):
#     """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).
#
#     You can easily do this by just having a class that calls each preprocessor in succession.
#
#     For example, if you call the process_state_for_network and you
#     have a sequence of AtariPreproccessor followed by
#     HistoryPreprocessor. This this class could implement a
#     process_state_for_network that does something like the following:
#
#     state = atari.process_state_for_network(state)
#     return history.process_state_for_network(state)
#     """
#     def __init__(self, preprocessors):
#         pass
