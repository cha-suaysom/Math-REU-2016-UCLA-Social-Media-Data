import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy.sparse import issparse, csr_matrix
import warnings
import sklearn.preprocessing

"""Non-negative least square solver
   Solves a non-negative least squares subproblem using the projected
   gradient descent algorithm.
   Parameters
   ----------
   V : array-like, shape (n_samples, n_features)
       Constant matrix.
   W : array-like, shape (n_samples, n_components)
       Constant matrix.
   H : array-like, shape (n_components, n_features)
       Initial guess for the solution.
   tol : float
       Tolerance of the stopping condition.
   max_iter : int
       Maximum number of iterations before timing out.
   alpha : double, default: 0.
       Constant that multiplies the regularization terms. Set it to zero to
       have no regularization.
   l1_ratio : double, default: 0.
       The regularization mixing parameter, with 0 <= l1_ratio <= 1.
       For l1_ratio = 0 the penalty is an L2 penalty.
       For l1_ratio = 1 it is an L1 penalty.
       For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
   sigma : float
       Constant used in the sufficient decrease condition checked by the line
       search.  Smaller values lead to a looser sufficient decrease condition,
       thus reducing the time taken by the line search, but potentially
       increasing the number of iterations of the projected gradient
       procedure. 0.01 is a commonly used value in the optimization
       literature.
   beta : float
       Factor by which the step size is decreased (resp. increased) until
       (resp. as long as) the sufficient decrease condition is satisfied.
       Larger values allow to find a better step size but lead to longer line
       search. 0.1 is a commonly used value in the optimization literature.
   Returns
   -------
   H : array-like, shape (n_components, n_features)
       Solution to the non-negative least squares problem.
   grad : array-like, shape (n_components, n_features)
       The gradient.
   n_iter : int
       The number of iterations done by the algorithm.
   References
   ----------
   C.-J. Lin. Projected gradient methods for non-negative matrix
   factorization. Neural Computation, 19(2007), 2756-2779.
   http://www.csie.ntu.edu.tw/~cjlin/nmf/
   """




def norm(x):
    """Compute the Euclidean or Frobenius norm of x.
    Returns the Euclidean norm when x is a vector, the Frobenius norm when x
    is a matrix (2-d array). More precise than sqrt(squared_norm(x)).
    """
    x = np.asarray(x)
    nrm2 = linalg.norm(x, 'fro')
    return nrm2



def safe_sparse_dot(a, b, dense_output=False):
    """Dot product that handle the sparse matrix case correctly
    Uses BLAS GEMM as replacement for numpy.dot where possible
    to avoid unnecessary copies.
    """
    if issparse(a) or issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)

def _nls_subproblem(V, W, H, tol, max_iter, alpha=0., l1_ratio=0.,
                    sigma=0.01, beta=0.1):
    WtV = safe_sparse_dot(W.T, V)
    WtW = np.dot(W.T, W)
    # values justified in the paper (alpha is renamed gamma)
    gamma = 1
    for n_iter in range(1, max_iter + 1):
        grad = np.dot(WtW, H) - WtV
        if alpha > 0 and l1_ratio == 1.:
            grad += alpha
        elif alpha > 0:
            grad += alpha * (l1_ratio + (1 - l1_ratio) * H)

        # The following multiplication with a boolean array is more than twice
        # as fast as indexing into grad.
        if norm(grad * np.logical_or(grad < 0, H > 0)) < tol:
            break

        Hp = H

        for inner_iter in range(20):
            # Gradient step.
            Hn = H - gamma * grad
            # Projection step.
            Hn *= Hn > 0
            d = Hn - H
            gradd = np.dot(grad.ravel(), d.ravel())
            dQd = np.dot(np.dot(WtW, d).ravel(), d.ravel())
            suff_decr = (1 - sigma) * gradd + 0.5 * dQd < 0
            if inner_iter == 0:
                decr_gamma = not suff_decr

            if decr_gamma:
                if suff_decr:
                    H = Hn
                    break
                else:
                    gamma *= beta
            elif not suff_decr or (Hp == Hn).all():
                H = Hp
                break
            else:
                gamma /= beta
                Hp = Hn
    if n_iter == max_iter:
        warnings.warn("Iteration limit reached in nls subproblem.")
    return H, grad, n_iter

import pickle
rows = 100
cols = 100
(W, H) = pickle.load(open('Location_NMF_100_topics_barc_WH.pkl','rb'))
rest_of_tweets_TFIDF  = pickle.load(open('rest_of_tweets_TFIDF_barc.pkl','rb'))
print(W.shape, H.shape)
Spatial_sample = pickle.load(open('Location_pandas_data_barc.pkl', 'rb'))
Topics = W.argmax(axis=1)
Spatial_sample["topics"] = Topics.tolist()
pickle.dump(Spatial_sample, open('Location_pandas_data_barc.pkl', 'wb'))

rest_of_tweets_Data = pickle.load(open('rest_of_tweets_pandas_data_barc.pkl','rb'))
normalized_H = sklearn.preprocessing.normalize(H[:,:-rows*cols])
print(np.linalg.norm((normalized_H[0:2, :]), 'fro'))
print(normalized_H.shape,rest_of_tweets_TFIDF.shape)

normalized_tweets = sklearn.preprocessing.normalize(rest_of_tweets_TFIDF)

topics = normalized_H*(normalized_tweets.T)
pickle.dump(topics, open('test_topic_distribution_barc.pkl', 'wb'))
H_ = topics
W_ = (H[:,:-rows*cols]).T
V_ = (rest_of_tweets_TFIDF).T
print(H_.shape, W_.shape, V_.shape)
(newH, somegrad, numberOfIterations) = _nls_subproblem(V_,W_,H_, 0.001,1000)
W_testing= newH.T
W_testing = sklearn.preprocessing.normalize(W_testing)
pickle.dump(W_testing, open('test_distribution_barc_nls.pkl', 'wb'))
print(linalg.norm((W_testing-topics.T), 'fro'))