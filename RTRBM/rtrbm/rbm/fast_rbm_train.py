from trainers.std_trainer3 import std_trainer
import rbm


def fast_rbm_train(name, RBM, CD_n, data_fn, LR, WD, momentum, num_iter):
    def grad(W, x):
        return rbm.rbm_grad_cd(W, x, CD_n)

    def loss(W, x):
        return rbm.rbm_grad_cd(W, x, CD_n)[1]

    return std_trainer(name=name,
                       W = RBM,
                       path = 'rbm/r_data',
                       unnorm_grad_fn = grad,
                       unnorm_valid_loss = loss,
                       data_fn = data_fn,
                       valid_data_fn = data_fn,
                       num_iter = num_iter,
                       LR = LR,
                       WD = WD,
                       momentum = momentum)
    
