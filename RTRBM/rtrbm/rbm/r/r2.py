import rbm

from mats.bias_mat import bias_mat
from mats.std_mat import std_mat

import data.bouncing_balls as b

res = 20
n_balls = 3
T = 100
dat = lambda : b.bounce_vec(res,n=3,T=100)

v=res**2
h=100

LR = .02

W = .02 * bias_mat(std_mat(v,h))
n_cd = 5
def grad(W, x):
    return rbm.rbm_grad_cd(W, x, n_cd)

def loss(W, x):
    return rbm.rbm_grad_cd(W, x, n_cd)[1]

from pylab import rand


from trainers.std_trainer3 import std_trainer

t = std_trainer(name='r1',
                path='rbm/r_data',
                W = W,
                unnorm_grad_fn = grad,
                unnorm_valid_loss = loss,
                data_fn = dat,
                valid_data_fn = dat,
                num_iter = 10000,
                LR =LR)

