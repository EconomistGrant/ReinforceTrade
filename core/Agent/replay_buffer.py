# -*- coding: utf-8 -*-
from collections import namedtuple
import random 
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayBuffer(object):
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
