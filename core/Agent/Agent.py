# -*- coding: utf-8 -*-
class Agent(object):
    """
    general agent api
    """

    def act(self,**kwargs):
        raise NotImplementedError
    
    def optimize(self,**kwargs):
        raise NotImplementedError

    def train(self,**kwargs):
        raise NotImplementedError
        