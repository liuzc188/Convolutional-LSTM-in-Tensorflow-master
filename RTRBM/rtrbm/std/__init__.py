from std.basic0 import ind2sub, min_N, max_N

from std.basic import (load, save, show, show_seq, gsave, print_aligned, sigmoid,
                       Rsigmoid, Dsigmoid, stochastic, monomial, multinomial, id,
                       print_seq, gload, fload, gsave, aux, aux2, true_multinomial,
                       softmax, generic_softmax_deriv, expand, e_true_multinomial, ieye, LOG)
                  
from pylab import figure, exp, log, concatenate,floor, ceil
from weight_decay import l_p, dl_p


from std.basic2 import rev, random_subset, im_show_resize
from std.basic3 import (grabI, fast_sigmoid, fast_Rsigmoid, log_sum_exp, log_sum_exp_fast_d, 
                        log_sum_exp_fast, log_sum_exp_fast_2d, pgrabI, update_features
                        )

from scipy import nan,inf



def isscalar(x):
    return x.__class__==float or x.__class__==int or x.__class__==complex


#from img import imread -- that's much worse than---
#existing imread in scipy.misc. Great! 
#I ought to be able to port the camera calibration code...
#will I do it? Find out in the next episodes.
#from scipy.misc import imread
from matplotlib.image import imread
#def imread(*args):
#    raise 
  
#from trainers.loss_fns import expand
#from mio import loadmat, savemat 



from std.par import pipe_par, pipe_par_learn, waitall, par, pipe_par_labs2, pipe_par_labs, pipe_par_labs_reducer, pipe_par_reducer
from std.basic import to_list_lens, from_list_lens



def ifloor(x):
    return int(floor(x))
def iceil(x):
    return int(ceil(x))

from std.named_vec import named_vec
from std.basic5 import printf, itera, iterm

from std.print_aligned2 import print_aligned2


