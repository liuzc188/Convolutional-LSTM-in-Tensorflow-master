import rbm

from mats.bias_mat import bias_mat
from mats.std_mat import std_mat

v=4
batch_size=20
W = bias_mat(std_mat(v,5))

def grad(W, x):
    return rbm.rbm_grad_exact(W, x)

def loss(W, x):
    return rbm.rbm_grad_exact(W, x)[1]

from pylab import rand
def data_fn():
    return (rand(batch_size,v)<.5).astype('f')

from trainers.std_trainer3 import std_trainer

t = std_trainer(name='r1',
                path='rbm/r_data',
                W = W,
                unnorm_grad_fn = grad,
                unnorm_valid_loss = loss,
                data_fn = data_fn,
                valid_data_fn = data_fn,
                num_iter = 1000)

