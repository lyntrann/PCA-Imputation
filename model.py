import numpy as np
from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
import pickle
import time
from sklearn.model_selection import cross_val_score
from keras.datasets import fashion_mnist
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from sklearn.linear_model import LogisticRegression
from utils import *
from tqdm import tqdm
import sys
import sklearn.neighbors._base
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.impute import IterativeImputer, KNNImputer
from missingpy import MissForest
from fancyimpute import SoftImpute
from numpy.linalg import norm, inv
from PCA import PCA

def gain (data_x, gain_parameters):
    '''Impute missing values in data_x
    
    Args:
      - data_x: original data with missing values
      - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations
        
    Returns:
      - imputed_data: imputed data
    '''
    # Define mask matrix
    data_m = 1-np.isnan(data_x)
    
    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']
    
    # Other parameters
    no, dim = data_x.shape
    
    # Hidden state dimensions
    h_dim = int(dim)
    
    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)
    
    ## GAIN architecture   
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape = [None, dim])
    # Mask vector 
    M = tf.placeholder(tf.float32, shape = [None, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape = [None, dim])
    
    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
    
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
    
    #Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
    G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
    
    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape = [dim]))
    
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
    
    ## GAIN functions
    # Generator
    def generator(x,m):
      # Concatenate Mask and Data
      inputs = tf.concat(values = [x, m], axis = 1) 
      G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
      G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
      # MinMax normalized output
      G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
      return G_prob
        
    # Discriminator
    def discriminator(x, h):
      # Concatenate Data and Hint
      inputs = tf.concat(values = [x, h], axis = 1) 
      D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
      D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
      D_logit = tf.matmul(D_h2, D_W3) + D_b3
      D_prob = tf.nn.sigmoid(D_logit)
      return D_prob
    
    ## GAIN structure
    # Generator
    G_sample = generator(X, M)
  
    # Combine with observed data
    Hat_X = X * M + G_sample * (1-M)
    
    # Discriminator
    D_prob = discriminator(Hat_X, H)
    
    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                  + (1-M) * tf.log(1. - D_prob + 1e-8)) 
    
    G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
    
    MSE_loss = \
    tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
    
    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss #f
    
    ## GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    
    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Start Iterations
    for it in tqdm(range(iterations)):    
        
      # Sample batch
      batch_idx = sample_batch_index(no, batch_size)
      X_mb = norm_data_x[batch_idx, :]  
      M_mb = data_m[batch_idx, :]  
      # Sample random vectors  
      Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
      # Sample hint vectors
      H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
      H_mb = M_mb * H_mb_temp
        
      # Combine random vectors with observed vectors
      X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
        
      _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                                feed_dict = {M: M_mb, X: X_mb, H: H_mb})
      _, G_loss_curr, MSE_loss_curr = \
      sess.run([G_solver, G_loss_temp, MSE_loss],
              feed_dict = {X: X_mb, M: M_mb, H: H_mb})
              
    ## Return imputed data      
    Z_mb = uniform_sampler(0, 0.01, no, dim) 
    M_mb = data_m
    X_mb = norm_data_x          
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
        
    imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
    
    imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
    
    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)  
    
    # Rounding
    imputed_data = rounding(imputed_data, data_x)  
            
    return imputed_data


def main_GAIN (X_nan, batch_size = 128, hint_rate = 0.9, alpha = 100, iterations = 10000):
    '''Main function imputing by GAIN method for dataset with missing data.
  
    Args:
    - X_nan: NumPy ndarray: dataset with missing data.
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
    Returns:
    - imputed_gain: data after GAIN imputation
    '''
  

    gain_parameters = {'batch_size': 128,
                     'hint_rate': 0.9,
                     'alpha': 100,
                     'iterations': 10000}
    X = X_nan
    
  # Impute missing data & timer function
    imputed_gain = gain(X, gain_parameters)

    return imputed_gain

def generate_nan(X, non_missing_cols = None, missing_rate = 0.2):
    X_non_missing = X[:, non_missing_cols]
    X_missing = X[:, [i for i in range(X.shape[1]) if i not in non_missing_cols]]
    XmShape = X_missing.shape
    na_id = np.random.randint(0, X_missing.size, round(missing_rate * X_missing.size))
    X_nan = X_missing.flatten()
    X_nan[na_id] = np.nan
    X_nan = X_nan.reshape(XmShape)
    X_nan = np.hstack((X_non_missing, X_nan))
    return X_nan

def softImputer(X0, q, missing_rate):
    X = X0
    scaler = StandardScaler()

    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    scaler.fit(Xm)
    Xm = scaler.transform(Xm)            
    Xfull = scaler.transform(X)  
            
    start = time.time()
    Xsoft = SoftImpute(max_iters = 2, verbose = 2).fit_transform(Xm)       
    tsoft = time.time() - start            
            
    err = norm((Xfull - Xsoft)[:,q:])
    err = err**2/Xfull[:,q:].size
    return Xsoft, np.array([err, tsoft])

def pcaSoft(X0, q, missing_rate, pca_pjt = "pca1"):
    X = X0
    scaler = StandardScaler()

    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    scaler.fit(Xm)
    Xm = scaler.transform(Xm)            
    Xfull = scaler.transform(X)  
    start = time.time()
    pca_model = PCA(q, pca_type=pca_pjt)
    Xpr = pca_model.fit_transform(Xm)
    XpcaSoft = SoftImpute(max_iters = 2, verbose = False).fit_transform(np.hstack((Xpr, Xm[:,q:])))
    tpcaSoft = time.time() - start 
    n_take = pca_model.n_take
    err = norm((Xfull[:, q:] - XpcaSoft[:, n_take:]))
    err = err**2/Xfull[:,q:].size
    return XpcaSoft, pca_model, np.array([err, tpcaSoft])

def miceImputer(X0, q, missing_rate):
    np.random.seed(7)
    X = X0
    scaler = StandardScaler()

    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    scaler.fit(Xm)
    Xm = scaler.transform(Xm)            
    Xfull = scaler.transform(X)  
            
    start = time.time()
    XMice = IterativeImputer(max_iter = 2, verbose=0).fit_transform(Xm)
    tmice = time.time() - start
            
    err = norm((Xfull - XMice)[:,q:])
    err = err**2/Xfull[:,q:].size
    return XMice, np.array([err, tmice])

def pcaMice(X0, q, missing_rate, pca_pjt = "pca1"):
    np.random.seed(7)
    X = X0
    scaler = StandardScaler()

    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    scaler.fit(Xm)
    Xm = scaler.transform(Xm)            
    Xfull = scaler.transform(X)  
                        
    start = time.time()
    pca_model = PCA(q, pca_type=pca_pjt)
    Xpr = pca_model.fit_transform(Xm)
    tXpr = time.time() - start
            
    start = time.time()
    XpcaMice = IterativeImputer(max_iter = 2, verbose=0).fit_transform(np.hstack((Xpr, Xm[:,q:])))
    tpcaMice = time.time() - start + tXpr
    n_take = pca_model.n_take
    err = norm(Xfull[:,q:] - XpcaMice[:, n_take:])
    err = err**2/Xfull[:,q:].size
    return XpcaMice, pca_model, np.array([err, tpcaMice])
        
def pca_missf(X0, q, missing_rate, pca_pjt = "pca1"):
    np.random.seed(7)
    X = X0
    scaler = StandardScaler()

    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    scaler.fit(Xm)
    Xm = scaler.transform(Xm)            
    Xfull = scaler.transform(X)              
                        
    start = time.time()
    pca_model = PCA(q, pca_type=pca_pjt)
    Xpr = pca_model.fit_transform(Xm)
    XpcaMissf = MissForest(random_state=None, n_estimators=2, max_depth=3, verbose = 0).fit_transform(np.hstack((Xpr, Xm[:,q:])))
    rtime = time.time() - start 
    n_take = pca_model.n_take
    err = norm(Xfull[:,q:] - XpcaMissf[:, n_take:])
    err = err**2/Xfull[:,q:].size
        
    return XpcaMissf, pca_model, np.array([err, rtime])
        
def missfImputer(X0, q, missing_rate):
    np.random.seed(7)
    X = X0
    scaler = StandardScaler()

    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    scaler.fit(Xm)
    Xm = scaler.transform(Xm)            
    Xfull = scaler.transform(X)              
            
    start = time.time()
    XMissf = MissForest(random_state=None, n_estimators=2, max_depth=3, verbose = 0).fit_transform(Xm)  
    rtime = time.time() - start

    err = norm((Xfull - XMissf)[:,q:])
    err = err**2/Xfull[:,q:].size
        
    return XMissf, np.array([err, rtime])  
        
def pca_knn(X0, q, missing_rate, pca_pjt = "pca1"):
    np.random.seed(7)
    X = X0
    scaler = StandardScaler()

    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    scaler.fit(Xm)
    Xm = scaler.transform(Xm)            
    Xfull = scaler.transform(X)              
                        
    start = time.time()
    pca_model = PCA(q, pca_type=pca_pjt)
    Xpr = pca_model.fit_transform(Xm)
    XpcaMissf = KNNImputer(n_neighbors=5).fit_transform(np.hstack((Xpr, Xm[:,q:])))
    rtime = time.time() - start 
    n_take = pca_model.n_take
    err = norm(Xfull[:,q:] - XpcaMissf[:, n_take:])
    err = err**2/Xfull[:,q:].size
        
    return XpcaMissf, pca_model, np.array([err, rtime])
        
def knnImputer(X0, q, missing_rate):
    np.random.seed(7)
    X = X0
    scaler = StandardScaler()

    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    scaler.fit(Xm)
    Xm = scaler.transform(Xm)            
    Xfull = scaler.transform(X)              
            
    start = time.time()
    XMissf = KNNImputer(n_neighbors=5).fit_transform(Xm)  
    rtime = time.time() - start

    err = norm((Xfull - XMissf)[:,q:])
    err = err**2/Xfull[:,q:].size
        
    return XMissf, np.array([err, rtime])   
        
def gainImputer(X0, q, missing_rate):
    np.random.seed(7)
    X = X0
    scaler = StandardScaler()

    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    scaler.fit(Xm)
    Xm = scaler.transform(Xm)            
    Xfull = scaler.transform(X)  
            
    start = time.time()
    Ximputed = main_GAIN(Xm, batch_size = 128, hint_rate = 0.9, alpha = 100, iterations = 10000)
    rtime = time.time() - start
            
    err = norm((Xfull - Ximputed)[:,q:])
    err = err**2/Xfull[:,q:].size # Newly added: 16.05.2022 #
    
    return Ximputed, np.array([err, rtime])
        
def pcaGain(X0, q, missing_rate, pca_pjt="pca1"):
    np.random.seed(7)
    X = X0
    scaler = StandardScaler()

    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    scaler.fit(Xm)
    Xm = scaler.transform(Xm)            
    Xfull = scaler.transform(X)  
                        
    start = time.time()
    pca_model = PCA(q, pca_type=pca_pjt)
    Xpr = pca_model.fit_transform(Xm)    
    XpcaGain = main_GAIN(np.hstack((Xpr, Xm[:,q:])), batch_size = 128, hint_rate = 0.9, alpha = 100, iterations = 10000)
    tpcaGain = time.time() - start 
    n_take = pca_model.n_take
    err = norm(Xfull[:,q:] - XpcaGain[:, n_take:])
    err = err**2/Xfull[:,q:].size
    return XpcaGain, pca_model, np.array([err, tpcaGain])

def svd_soft(X0, q, missing_rate):
    X = X0
    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    start = time.time()
    n_take = 150
    svd_model = TruncatedSVD(150)
    Xpr = svd_model.fit_transform(Xm)
    XsvdSoft = SoftImpute(max_iters = 2, verbose = False).fit_transform(np.hstack((Xpr, Xm[:,q:])))
    err = norm((X[:, q:] - XsvdSoft[:, n_take:]))
    tsvd_soft = time.time() - start
    err = err**2/X[:,q:].size
    return XsvdSoft, svd_model, np.array([err, tsvd_soft])

def svd_mice(X0, q, missing_rate):
    X = X0
    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    start = time.time()
    n_take = 150
    svd_model = TruncatedSVD(150)
    Xpr = svd_model.fit_transform(Xm)
    XsvdMice = IterativeImputer(max_iter = 2, verbose=0).fit_transform(Xm)
    err = norm((X[:, q:] - XsvdMice[:, n_take:]))
    tsvd_soft = time.time() - start
    err = err**2/X[:,q:].size
    return XsvdMice, svd_model, np.array([err, tsvd_soft])

def svd_missf(X0, q, missing_rate):
    X = X0
    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    start = time.time()
    n_take = 150
    svd_model = TruncatedSVD(150)
    Xpr = svd_model.fit_transform(Xm)
    XsvdMissf = MissForest(random_state=None, n_estimators=2, max_depth=3, verbose = 0).fit_transform(np.hstack((Xpr, Xm[:,q:])))
    err = norm((X[:, q:] - XsvdMissf[:, n_take:]))
    tsvd_soft = time.time() - start
    err = err**2/X[:,q:].size
    return XsvdMissf, svd_model, np.array([err, tsvd_soft])

def svd_knn(X0, q, missing_rate):
    X = X0
    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    start = time.time()
    n_take = 150
    svd_model = TruncatedSVD(150)
    Xpr = svd_model.fit_transform(Xm)
    Xsvdknn = KNNImputer(n_neighbors=5).fit_transform(np.hstack((Xpr, Xm[:,q:])))
    err = norm((X[:, q:] - Xsvdknn[:, n_take:]))
    tsvd_soft = time.time() - start
    err = err**2/X[:,q:].size
    return Xsvdknn, svd_model, np.array([err, tsvd_soft])

def svd_gain(X0, q, missing_rate):
    X = X0
    Xm = generate_nan(X, non_missing_cols = np.arange(q), missing_rate = missing_rate)
    start = time.time()
    n_take = 150
    svd_model = TruncatedSVD(150)
    Xpr = svd_model.fit_transform(Xm)
    XsvdGain = main_GAIN(np.hstack((Xpr, Xm[:, q:])), batch_size=128, hint_rate=0.9, alpha=100, iterations=10000)
    err = norm((X[:, q:] - XsvdGain[:, n_take:]))
    tsvd_soft = time.time() - start
    err = err**2/X[:,q:].size
    return XsvdGain, svd_model, np.array([err, tsvd_soft])