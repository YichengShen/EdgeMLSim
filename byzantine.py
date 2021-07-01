import mxnet as mx
import random


# no faulty workers
def no_byz(v, f, grad_example=None):
    pass

# failures that add Gaussian noise
def gaussian_attack(v, f, grad_example=None):
    for i in range(f):
        v[i] = mx.nd.random.normal(0, 200, shape=v[i].shape)

# bit-flipping failure
def bitflip_attack(v, f, grad_example=None):
    for i in range(f):
        if i > 0:
            v[i][:] = -v[0]
    v[0][:] = -v[0]

# sign filpping failure
def signflip_attack(v, f, grad_example):
    MAGNITUDE = 5 # number of times the faulty layer is multipled by
    for i in random.sample(range(len(v)), f):
        idx = 0
        for j, param in enumerate(grad_example):
            if j % 2 == 0:
                v[i][idx:(idx+param.size)] *= -1 * MAGNITUDE
            idx += param.size
