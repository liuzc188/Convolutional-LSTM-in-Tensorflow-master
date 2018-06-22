from fns.ff_net import ff_net
from pylab import *
from std import *
import std
from std.loader import *
from trainers.common_functions import *
### TODO:
### ::: more flexible printing of the parameters

class std_trainer:
    """
    This is a sequential, non-parallel, most general version of a trainer of stochastic
    gradient descent problems with momentum.
    (URGENT FUTURE WORK: make use of some second order information, by using stochastic newton's method
    approximating the inverse hessian).

    It is given functions that gives training and validation cases,
    and functions that return unnormalized gradients and unnormalized loss/gain vaules.
    The gradient is combined with the momentum, which is then added to the parameters.
    It is recommended that when the dataset is finite, then the train_function should
    be implemented with a global variable that uses a random permutation estimator.
    This way, there will probably be less variance in the nodes.

    As the class implements grad_check, when used, it assums that the entry 'loss' is
    given in the output of the unnorm_valid_loss function; The gradient function returns
    direction in the steepest descent in loss, thus being its negative derivative.

    The learning rate, L2 weight decay and momentum are given in their most generality as
    functions; however, a dictionary of values (that represent the 'vertices' of a sum of step
    functions or a plain constant can be added as well.

    Given the save path of the simulation and its name, the object automatically loads
    itself provided it can find the file. In addition, the saveing interval parameter
    determines how often the simulation is to be saved.

    There will also be another version, that does not assume sequential structure, so it could
    concatenate datapoints into long vectors and save by doing matrix multiplications.

    This can be used for both finetuning and pretraining, where in pretraining, the gradient-computing
    function accesses the inference weights and the validation could be the reconstruction error
    on the test set, as well as possibly the free energy. 

    """

    def __init__(self,
                 name,  ### of simulation
                 path,  ### path to save/load simulation

                 W,
                 ### ff_net of weight parameters; can be anything.
                 ### if W has few "essential parameters", then by
                 ### providing a member function "W.compress()" that
                 ### returns a -REFERENCE- to the essential parameters,
                 ### this could allow for faster saving and loading.

                 unnorm_grad_fn,
                 ### gradient function:
                 ### input:  whatever data getter gives
                 ### output: (unnormalized) gradient for each W 
                 ### this could allow for more general backpropagations.
                 ### in addition, gard_fn returns an unnormalized dictionary of statistics:
                 ### losses. In particular, there should be a 'loss' which is to be minimized.
                 ### (containing various reconstruction errors)

                 unnorm_valid_loss,
                 ### unnorm_valid_loss is a function that outputs some validation statistics
                 ### which are averaged over all the batches. 
                 ### The results of unnorm_valid_loss are averaged over all the
                 ### test cases, stored and printed out (averaging takes different batch sizes into account)

                 data_fn, 
                 ### data_fn returns some object with data that is fed to unnorm_grad_fn
                 ### as well as the length of this object. This is relevant because oftentimes
                 ### this object is a tuple of inputs and labels. 
                 ### typically, data_fn might be implemented by creating a random permutation
                 ### of the datapoints and using it internally, perhaps by updating a global variable.

                 valid_data_fn,
                 ### valid_data_fn is a function class with 3 member functions:
                 ### init(): sets a new random permutation to 0
                 ### next(): returns the next batch of examples
                 ###         None if nothing is left.

                 num_iter,   ### how many weight updates (update follows after batch)
                 save_iter=10**15,  ### how many iterations between saves?
                 test_iter=None, ### save_iter by default
                 
                 min_D_len=1, ### if the input sequence is any shorter, we ignore it

                 weight_update_freq=1,
                 ### we could update the weights every datapoint (when the data is iid)
                 ### But to reduce variance (in case the data is a sequence, for instance),
                 ### we can collect a gradient over several iterations;
                 ### This can sometimes result in a significant reduction
                 ### in update variance.
                 
                 weights_constraints = lambda x : x,
                 ### Weights_constraints: W = weights_constraints(W) is executed after each weight update.
                 ### Useful for printing out stuff as well as enforcing some hard constraitns, eg
                 ### to upper-bound the norm of W.
                 weights_constraints_iter = lambda x, i : x,
                 ### This is another weights constraints function that also gets an iteration number.
                 ### it is convenient if we want to do things like restricting the amount of hidden
                 ### units that get updated. 


                 grad_constraints_iter = lambda x, i : x,
                 ### While this one is to bound the gradient, in case we wish to freeze some weights
                 ### instead of set them to a prespecified vaule, as done by the weight constraints.


                 LR = .01,   
                 WD =   0, 
                 momentum = .9,
                 ### Thus, the update equations are the following:
                 ### D  = NORMALIZED_GRAD
                 ### V *= momenutm
                 ### V += LR * D
                 ### W += V

                 const_mean_batch_size = None,
                 ### If we wish not to normalize by batch size,
                 ### (eg if the batches are of different sizes) then
                 ### it is better to use the mean batch size.

                 len_fn = None,
                 ### How to compute the length of a batch
                 ### in case we wish to normalize by batch length.


                 plot_iter = None, ### How often to plot
                 plot=True,        ### Whether to plot at all
                 plotter=None,     ### should a specific plotting function be used.?


                 backup_name = None,
                 ### If we save something and wish to resume from there,
                 ### we can use backup_name for initalization.

                 
                 train_stats_sparsity = None,
                 train_print_sparsity = None
                 ):

        self.min_D_len = min_D_len



        if path[-1]!='/':
            path = path + '/'

        self.name, self.path = name, path

        if backup_name == None:
            backup_name = name

        self.W = W

        self.valid_data_fn, self.data_fn = valid_data_fn, data_fn

        self.unnorm_valid_loss, self.unnorm_grad_fn = unnorm_valid_loss, unnorm_grad_fn

        self.LR, self.WD, self.momentum = map(make_parameter, (LR, WD, momentum))
                 
        self.num_iter, self.save_iter =  num_iter, save_iter

        if test_iter==None: test_iter=save_iter

        self.test_iter = test_iter

        if plot_iter == None:
            self.plot_iter = inf
        else:
            self.plot_iter = plot_iter

        self.iter, self.train_stats, self.valid_stats = 1, {}, {}
        if train_stats_sparsity==None:
            self.train_stats_sparsity = 1
        else:
            self.train_stats_sparsity = train_stats_sparsity
        self.train_stats_counter = 0

        if train_print_sparsity==None:
            self.train_print_sparsity = 1
        else:
            self.train_print_sparsity = train_print_sparsity
        self.train_print_counter = 0



        self.weight_update_freq = weight_update_freq
        self.dW_acc = 0 * W
        self.batch_size_acc = 0

        self.const_mean_batch_size = const_mean_batch_size

        def get_essential_W(W):
            try:
                ### compress returns a reference, so by setting it to something
                ### the value of the original array changes.
                return W.compress()
            except:
                return W


        self.get_essential_W = get_essential_W
            
        self.V       = 0*W

        try:
            try:
                print 'attempting to load %s' % (path + name)
                data = load(path + name)
            except IOError:
                print 'file does not exist. loading backup file...'
                try:
                    data = load(path + backup_name)
                    print 'loaded backup name %s' % backup_name
                except IOError:
                    print  'backup file does not exist. '
            WE = get_essential_W(self.W)
            WE.set(data['W'])


            VE = get_essential_W(self.V)
            VE.set(data['V'])
            self.train_stats = data['train_stats']
            self.valid_stats = data['valid_stats']
            self.iter = data['iter']
            self.train_stats_counter = data['train_stats_counter']
            print 'run %s: load successful.' % name
        except:
            print 'run %s: starting a fresh run' % name

        self.weights_constraints = weights_constraints
        self.weights_constraints_iter = weights_constraints_iter
        self.grad_constraints_iter = grad_constraints_iter

        self.plot    = plot
        self.plotter = plotter
        
        if len_fn == None:
            def my_len_fn(x):
                if type(x)==tuple:
                    return len(x[0])
                else:
                    return len(x)
            self.len_fn = my_len_fn
        else:
            self.len_fn = len_fn

    def train(self):
        for self.iter in range(self.iter, self.num_iter):
            W, V, LR, momentum, WD = self.W, self.V, self.LR, self.momentum, self.WD
            W = self.weights_constraints(W)
            W = self.weights_constraints_iter(W, self.iter)
            D = self.data_fn()

            batch_size = self.len_fn(D)
            if batch_size < self.min_D_len:
                continue
            (dW_loc, stats_dict) = self.unnorm_grad_fn(W, D)

            dW_loc = self.grad_constraints_iter(dW_loc, self.iter)

            ### accumulate the gradient's size, as well as the gradient
            ### in case we wish to be in a larger batches mode.
            self.batch_size_acc += batch_size
            self.dW_acc         += dW_loc


            ### normalize the training errors and append the batch size to this thing.
            for k in stats_dict.keys():
                stats_dict[k] /= float(batch_size)
            stats_dict['batch_size'] = batch_size

            if self.train_stats_counter % self.train_stats_sparsity==0:
                self.train_stats[self.iter] = stats_dict
            self.train_stats_counter += 1

            if self.iter % self.weight_update_freq == 0:
                if self.train_print_counter % self.train_print_sparsity==0:
                    print self.name, '   :updating weights', 
                ### UPDATING THE PARAMETERS.
                ### V *= momentum
                V.__imul__(momentum(self.iter))
                ### V += ( dW / batch_size_acc - W * WD ) * LR
                if self.const_mean_batch_size == None:
                    V += ((1/float(self.batch_size_acc))*self.dW_acc -
                          W.__rmul__(WD(self.iter))).__rmul__(LR(self.iter))
                    if self.train_print_counter % self.train_print_sparsity==0:
                        print ('iter=%5d, batch_size=%5d, |W|=%10.5f, |V|=%10.5f' %
                               (self.iter, batch_size, rms(W), rms(V)))

                else: 
                    V += ((1/float(self.const_mean_batch_size))*self.dW_acc -
                          W.__rmul__(WD(self.iter))).__rmul__(LR(self.iter))
                    if self.train_print_counter % self.train_print_sparsity==0:
                        print ('iter=%5d, batch_s=%5d, c_batch_s=%5d, |W|=%10.5f, |V|=%10.5f' %
                               (self.iter, self.batch_size_acc, int(self.const_mean_batch_size), rms(W), rms(V)))

                self.dW_acc *= 0
                self.batch_size_acc = 0
                W += V
            else:
                if self.train_print_counter % self.train_print_sparsity==0:
                    print self.name, '  :collected gradient; no weight update;', 
                    print ('iter=%5d, batch_size=%5d, |dW_acc|=%10.5f, ' %
                           (self.iter, batch_size, rms(self.dW_acc)))

            if self.train_print_counter % self.train_print_sparsity==0:                    
                show_stats('current train statss:', stats_dict)
                print ''

            try:
                last_iter = sort(array(self.valid_stats.keys()))[-1]
                dict_last_iter = self.valid_stats[last_iter]
                if self.train_print_counter % self.train_print_sparsity==0:
                    show_stats('previous (%s) valid stats:' % last_iter, dict_last_iter)
                    print ''
            except:
                if self.train_print_counter % self.train_print_sparsity==0:
                    print 'no previous valid statistics yet.'

            self.train_print_counter += 1
            if self.iter % self.plot_iter==0 and self.plot:
                self.show_W()

            ###### COMPUTING ESTIMATES OF THE VALIDATION LOSSES
            if self.iter % self.test_iter == 0:
                self.valid_stats[self.iter] = self.compute_validation_cost()

            #### SAVING THE DATA.
            if self.iter % self.save_iter == 0:
                print 'saving...'
                self.save()
        ### end of main for loop.

    def check_grad(self, num_tries = 10, eps=1e-6, inds = []):
        """
        This function verifies that the gradient and the loss function
        agree.
        There is a cavet to this function:
        the gradient function returns the direction that minimizes the loss function;
        therefore, the gradient function is actually the negative gradient function.
        """
        W_copy = 1 * self.W
        def get(W):
            for x in inds:
                W = W[x]
            return W

        ### select a random batch from the validation score.
        d = self.valid_data_fn()

        dW, grad_stats = self.unnorm_grad_fn(W_copy, d)

        Actual_W = get(W_copy).flatten()
        Ref_W    = get(W_copy)   ### the reference.
        len_W = len(Actual_W.flatten())

        def set_to_W_copy(Actual_W):
            try:
                Ref_W.set(Ref_W.unpack(Actual_W))
            except AttributeError:
                try:
                    Ref_W[:] = Actual_W.reshape(shape(Ref_W))
                except AttributeError:
                    Ref_W[:] = Actual_W
        for i in range(num_tries):
            k = int(multinomial(len_W))

            Actual_W[k] += eps
            set_to_W_copy(Actual_W)
            l1 = self.unnorm_valid_loss(W_copy , d)['loss']
            print 'l1=',l1

            Actual_W[k] -= 2*eps
            set_to_W_copy(Actual_W)
            l2 = self.unnorm_valid_loss(W_copy, d)['loss']

            Actual_W[k] += eps
            set_to_W_copy(Actual_W)

            estimated_grad = (l1-l2)/(2*eps)
            compute_grad   = - get(dW).flatten()[k]

            print i,'c',compute_grad,'e',estimated_grad,': diff=', compute_grad-estimated_grad


    ### this allows for using other data sources, too!
    def compute_validation_cost(self, valid_data_fn=None):
        W = self.W
        if valid_data_fn == None:
            valid_data_fn = self.valid_data_fn

        print 'testing...'
        valid_tot_num_cases = 0
        unnorm_stat_sum = dict()
        stat_mean       = dict()

        self.valid_data_fn.init()
        i=0
        while 1:
            printf('valid: i=%d      \r' % i)
            i+=1

            d = self.valid_data_fn.next()

            if d==None:
                break  ### then we are truly done.
            
            len_d = self.len_fn(d)

            if len_d < self.min_D_len:
                continue

            valid_tot_num_cases += len_d

            v_stat = self.unnorm_valid_loss(W, d)
            
            #### We update the sum of all the losess.
            for key in v_stat.iterkeys():
                try:
                    unnorm_stat_sum[key] += v_stat[key]
                except KeyError:
                    unnorm_stat_sum[key]  = v_stat[key]

        ### Now we normalize by the total number of examples processed.
        for key in v_stat.iterkeys():
            stat_mean[key] = unnorm_stat_sum[key] / float(valid_tot_num_cases)
            
        stat_mean['batch_size'] = int(valid_tot_num_cases)
        return stat_mean
            
            
    def save(self):
        """
        To save the current state manually.
        """
        to_save = dict(train_stats= self.train_stats,
                       valid_stats =self.valid_stats,
                       iter =self.iter,
                       W    =self.get_essential_W(self.W),
                       V    =self.get_essential_W(self.V),
                       train_stats_counter=self.train_stats_counter)
        save(to_save, self.path + self.name)

    def show_W(self):
        W = self.W
        try:
            self.plotter.__call__(W)
        except:
            try:
                show(self.W[0].show_W())
            except:
                show(self.W.show_W(),-1,1)


    def plot_train_stats(self, key='loss',plot_params='-', log_scale=False, down_sampling=1, averaging_down_sampling=True):
        Xs = sort(array(self.train_stats.keys()))
        Ys = 0 * Xs.astype('d')
        for i in range(len(Xs)):
            Ys[i] = self.train_stats[int(Xs[i])][key]
        if log_scale:
            Xs = log10(Xs)

        XX = Xs[::down_sampling]
        if averaging_down_sampling==False:
            YY = Ys[::down_sampling]
        else:
            YY = Ys[::down_sampling]
            for t in range(1, down_sampling):
                Yz = Ys[t::down_sampling]
                YY[:len(Yz)] += Yz
            YY /= float(down_sampling)

        plot(XX, YY, plot_params)

    def plot_valid_stats(self, key='loss',plot_params='-',log_scale=False, down_sampling=1):
        Xs = sort(array(self.valid_stats.keys()))
        
        Ys = 0 * Xs.astype('d')
        for i in range(len(Xs)):
            Ys[i] = self.valid_stats[int(Xs[i])][key]
        if log_scale:
            Xs = log10(Xs)
        plot(Xs[::down_sampling], Ys[::down_sampling], plot_params)


def show_normalized_stats(name, stats_dict, total_batch_size):
    print name
    for key in stats_dict.iterkeys():
        if key!='batch_size':
            print key,'=', stats_dict[key]/float(total_batch_size)

def show_stats(name, stats_dict):
    print name, ':::',
    i = 0
    for key in stats_dict.iterkeys():
        if key!='batch_size':
            print key,'=', stats_dict[key],';',
            if i==3:
                i=0
                print ''
            i+=1
