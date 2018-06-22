## This is how we trained the mocap models.
## To sample, we used Y = r.t.W.sample(100,200),
## then r.save_sample(Y, filename)

import rbm
from mats.bias_mat import bias_mat
from mats.std_mat import std_mat

CD_n_sched = { 100 : 10, 1000 : 25 }
CD_n_sched = { 100 : 10, 1000 : 25 }
def weights_constraint_iter(W, i):

    try:
        W.CD_n = CD_n_sched[i]
    except KeyError:
        pass
    return W

import data.mocap as mocap
from p8.rnn_trbm.rnn_trbm import rnn_trbm
from p8.rnn_trbm.trbm import trbm


T_len = 50
dat_tr = lambda : mocap.data_sample_tr(batch_size = T_len)
dat_tst = lambda : mocap.data_sample_tst(batch_size = T_len)


v=49
h=200

NUM_ITER = 10**5
def LR(x): 
    return .001 * (1- float(x)/NUM_ITER)

VH = bias_mat(std_mat(v,h))
HH = bias_mat(std_mat(h,h))

CD_n = 5
choice = input('press 0 for RNN-TRBM, 1 for plain TRBM\n...')
W0 = .005*rnn_trbm(VH, HH, CD_n = CD_n, vis_gauss = True)
W1 = .005*    trbm(VH, HH, CD_n = CD_n, vis_gauss = True)
W = [W0, W1][choice]

name = 'r_mocap_' + ['rnn-trbm', 'plain-trbm'][choice]

def grad(W, x):    return W.grad(x)
def loss(W, x):    return W.grad(x)[1]


from trainers.std_trainer3 import std_trainer

t = std_trainer(name = name,
                path='p8/rnn_trbm/r_data/',
                W = W,
                weights_constraints_iter = weights_constraint_iter,
                unnorm_grad_fn = grad,
                unnorm_valid_loss = loss,
                data_fn = dat_tr,
                valid_data_fn = dat_tst,
                num_iter = NUM_ITER,
                LR =LR,
                WD = .0,
                train_print_sparsity = 5)

t.W.CD_n = CD_n # 


import rbm.fast_rbm_train
T = rbm.fast_rbm_train.fast_rbm_train(name='hack',RBM = W[0], CD_n=5, 
                                      data_fn = dat_tr, LR = .01, WD =0, 
                                      momentum = .9, num_iter=50)
def init_good_VH():
    T.train()
    t.W[0] = T.W 

def renew_W():
    W.w = t.W.w
    t.W = W

def show_sample(X):
    mocap.show_mocap_seq(X)
    

