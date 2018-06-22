#from scipy.interpolate import interp2d is interp2
from pylab import *
from scipy.weave import converters, inline

def pgrabI(B,I,ANS):
    I=array(I,'i')
    sh_I   = shape(I)
    sh_B   = shape(B)
    sh_ANS = shape(ANS)
    len_B  = sh_B[0]
    assert(sh_ANS==sh_I==(len_B,))


    code="""
    for(int i=0; i<len_B; i++)
     ANS(i)=B(i,I(i));     
    """
    inline(code, ['I', 'B', 'ANS', 'len_B'],
           type_converters=converters.blitz, compiler='gcc')
    

def grabI(B,I,ANS):
    I = array(I,'i')
    sh_B  =shape(B)
    sh_ANS=shape(ANS)

    sh_b  = [sh_B  [0], prod(sh_B  [1:])]
    sh_ans= [sh_ANS[0], prod(sh_ANS[1:])]

    B.resize(*sh_b)
    ANS.resize(*sh_ans)

    dim_B = int(sh_b[1])
    len_I = len(I)

    code="""
    for(int i=0; i<len_I; i++)
     for(int j=0; j<dim_B; j++)
      ANS(i,j)=B(I(i), j);
    """
    inline(code, ['I', 'B', 'ANS', 'dim_B', 'len_I'],
           type_converters=converters.blitz, compiler='gcc')

    ANS.resize(*sh_ANS)
    B.resize(*sh_B)
#### when the time comes, write a routine for addiI(B,I,VALS):
#### that does the analouge of B[I]+=VALS, with possible repetitions in I.
      







def fast_sigmoid(B,ANS):
    b=B.ravel()
    ans=ANS.ravel()

    assert(all(shape(ans)==shape(b)))
    U = len(ans)

    #### we operateno on b and ans, whose shapes are known and good.
    inline("""
    for(int i=0; i<U; i++)
      ans(i)=1/(1+exp(-b(i)));
    """, ['U','ans','b'],
    type_converters=converters.blitz, compiler='gcc')

def fast_Rsigmoid(B,ANS):
    b=B.ravel()
    ans=ANS.ravel()
    z=rand(len(ans))

    assert(all(shape(ans)==shape(b)))
    U = len(ans)

    #### we operateno on b and ans, whose shapes are known and good.
    inline("""
    for(int i=0; i<U; i++)
      ans(i)=z(i) < 1/(1+exp(-b(i)));
    """, ['U','ans','b','z'],
    type_converters=converters.blitz, compiler='gcc')



def update_features(Features, positions, Updaters):
    Num_features, Dim = Features.shape

    #Features = Features.astype(float32)
    Updaters = Updaters.astype(float32)

    positions = array(positions).astype(int32)

    num_updaters, Dim_1 = Updaters.shape

    assert(Dim_1 == Dim and num_updaters==len(positions))
    assert(positions.max() < Num_features and 0 <= positions.min())

    inline("""
    using namespace std;
    for (int i=0; i<num_updaters; i++)
      for (int d=0; d<Dim; d++){
         Features(positions(i),d) += Updaters(i,d);
      }
    """, ['num_updaters','Dim','Features','Updaters','positions'],
               type_converters=converters.blitz, compiler='gcc')


def log_sum_exp(A,d=None):
    """
    log_sum_exp(A,d=None): input: A is an n-dimensional array;
    return the summation according to the dimension specified in d.
    """

    m = A.max(d)
    sh_m = shape(m)
    sh = list(shape(A))
    if d!=None:
        sh[d] = 1
    m.resize(sh)

    B = A - m

    ANS = exp(B).sum(d)

    # ANS *= exp(m).reshape(sh_m)
    # boy, was I stupid or waht?!! This is outrageously stupid.

    if isnan(ANS).any():
        pdb.set_trace()

    if isinf(ANS).any():
        pdb.set_trace()

    FINAL = log(ANS)+m.reshape(sh_m)

    if isnan(FINAL).any():
        pdb.set_trace()


    return FINAL


def log_sum_exp_fast_d(A,d):
    """
    log_sum_exp(A,d): input: A is an n-dimensional array;
    return the summation according to the dimension specified in d.
    """

    m = A.max(d)
    sh_m = shape(m)
    sh = list(shape(A))
    sh[d]=1
    m.resize(sh)

    B = A - m

    ANS = exp(B).sum(d)

    FINAL = log(ANS)+m.reshape(sh_m)

    return FINAL

def log_sum_exp_fast_2d(A,d):
    """
    log_sum_exp(A,d): input: A is an 2-dimensional array;
    return the summation according to the dimension specified in d.
    """
    u,v = shape(A)
    if d==0:
        m = A.max(0)
        m.resize(1,v)
        B = A - m
        ANS = exp(B).sum(0)
        FINAL = log(ANS)+m #.reshape(u,1)
    elif d==1:
        m = A.max(1)
        m.resize(u,1)
        B = A - m
        ANS = exp(B).sum(1)
        FINAL = log(ANS)+m.reshape(u)
    else:
        raise Exception ("log_sum_exp_fast_2d is not designed with non-2d arrays in mind")
    return FINAL


def log_sum_exp_fast(A):
    """
    log_sum_exp_fast(A): input: A is an n-dimensional array;
    no checks, no dimension.
    """
    m = A.max()

    B = A - m

    ANS = exp(B).sum()

    FINAL = log(ANS)+m

    return FINAL



