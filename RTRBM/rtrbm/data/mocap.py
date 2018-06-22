from pylab import loadmat, array, concatenate, cumsum

A = loadmat('data/MOCAP')

data = A['batchdata']

seqlengths = A['seqlengths']

seqstarts = concatenate(([0], cumsum(seqlengths)))



seqprobs = seqlengths / float(seqlengths.sum())

from pylab import rand,ifloor
def int_rand_range(a,b):
    return ifloor(rand()*(b-a)+a)


from pylab import true_multinomial, newaxis
def data_sample(batch_size):
    seq_id = int(true_multinomial(seqprobs[newaxis,:]))

    seq_len = seqlengths[seq_id]
    seq_absolute_A = seqstarts[seq_id]
    seq_absolute_B = seqstarts[seq_id+1]

    seq_pos_A = int_rand_range(seq_absolute_A, seq_absolute_B - batch_size)
    
    return data[seq_pos_A:seq_pos_A + batch_size]




test_frac = .2

from pylab import true_multinomial, newaxis
## sample from the training set
def data_sample_tr(batch_size):
    seq_id = int(true_multinomial(seqprobs[newaxis,:]))

    seq_len = seqlengths[seq_id]
    seq_absolute_A = seqstarts[seq_id]
    seq_absolute_B = seqstarts[seq_id+1]

    assert(seq_len == seq_absolute_B - seq_absolute_A)

    seq_absolute_B -= seq_len * test_frac

    seq_pos_A = int_rand_range(seq_absolute_A, seq_absolute_B - batch_size)
    
    return data[seq_pos_A:seq_pos_A + batch_size]


from pylab import true_multinomial, newaxis
## sample from the test set
def data_sample_tst(batch_size):
    seq_id = int(true_multinomial(seqprobs[newaxis,:]))

    seq_len = seqlengths[seq_id]
    seq_absolute_A = seqstarts[seq_id]
    seq_absolute_B = seqstarts[seq_id+1]


    assert(seq_len == seq_absolute_B - seq_absolute_A)

    seq_absolute_A = seq_absolute_B - seq_len * test_frac

    seq_pos_A = int_rand_range(seq_absolute_A, seq_absolute_B - batch_size)
    
    return data[seq_pos_A:seq_pos_A + batch_size]





class counter:
    count = 0
def show_mocap_seq(X):
    import mlabwrap

    if counter.count == 0:
        mlabwrap.mlab.addpath('data/mhmublv/')
        mlabwrap.mlab.addpath('data/mhmublv/Data')
        mlabwrap.mlab.addpath('data/mhmublv/Motion')
        mlabwrap.mlab.addpath('data/mhmublv/Results')

    Z = mlabwrap.mlab.show_mocap(X)
    counter.count += 1



