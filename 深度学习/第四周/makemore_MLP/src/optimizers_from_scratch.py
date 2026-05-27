from __future__ import annotations

import torch
from torch import Tensor

import matplotlib.pyplot as plt



def sgd_step(w,g,lr:float=0.1):
    return w-lr*g


def momentum_step(w,g,v,lr:float=0.1,beta:float=0.9):
    v=beta*v+g
    return w-lr*v,v

def rmsprop_step(w,g,s,lr:float=0.1,beta2:float=0.999,eps:float=1e-8):
    s=beta2*s+(1-beta2)*(g**2)
    return w-lr*g/(s.sqrt()+eps),s

def adam_step(w,g,m,v,t,lr:float=0.1,b1:float=0.9,b2:float=0.999,eps:float=1e-8):
    m=b1*m+(1-b1)*g
    v=b2*v+(1-b2)*(g**2)
    m_hat=m/(1-b1**t)
    v_hat=v/(1-b2**t)
    return w-lr*m_hat/(v_hat.sqrt()+eps),m,v




























