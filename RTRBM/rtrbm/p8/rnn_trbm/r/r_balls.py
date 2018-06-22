import rbm
from mats.bias_mat import bias_mat
from mats.std_mat import std_mat

CD_n_sched = { 100 : 10, 1000 : 25 }
def weights_constraint_iter(W, i):
    print 'W.CD_n = ', W.CD_n
    try:
        W.CD_n = CD_n_sched[i]
    except KeyError:
        pass
    return W

import data.bouncing_balls as b
from p8.rnn_trbm.rnn_trbm import rnn_trbm
from p8.rnn_trbm.trbm import trbm

res = 30
n_balls = 3
T = 100
dat = lambda : b.bounce_vec(res,n=3,T=100)


v=res**2
h=400 

NUM_ITER = 10**5
def LR(x): return .01 * (1- float(x)/NUM_ITER)

VH = bias_mat(std_mat(v,h))
HH = bias_mat(std_mat(h,h))

CD_n = 5
choice = input('press 0 for RNN-TRBM, 1 for plain TRBM\n...')
W0 = .005*rnn_trbm(VH, HH, CD_n = CD_n)
W1 = .005*    trbm(VH, HH, CD_n = CD_n)
W = [W0, W1][choice]

name = 'r_balls_' + ['rnn-trbm', 'plain-trbm'][choice]

def grad(W, x):    return W.grad(x)
def loss(W, x):    return W.grad(x)[1]


from trainers.std_trainer3 import std_trainer

t = std_trainer(name = name,
                path='p8/rnn_trbm/r_data/',
                W = W,
                weights_constraints_iter = weights_constraint_iter,
                unnorm_grad_fn = grad,
                unnorm_valid_loss = loss,
                data_fn = dat,
                valid_data_fn = dat,
                num_iter = NUM_ITER,
                LR =LR,
                WD = 0)

t.W.CD_n = CD_n # 


import rbm.fast_rbm_train
T = rbm.fast_rbm_train.fast_rbm_train(name='hack',RBM = W[0], CD_n=5, 
                                      data_fn = dat, LR = .01, WD =0, 
                                      momentum = .9, num_iter=30)
def init_good_VH():
    T.train()
    t.W[0] = T.W 

def renew_W():
    W.w = t.W.w
    t.W = W

def show_sample(X):
    from pylab import show_seq
    show_seq(X)


