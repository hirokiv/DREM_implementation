import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math 
import scipy
from sklearn.decomposition import TruncatedSVD

import sys
sys.path.append('Cpp_lib')
import adjugate_module # this is cpp library

# get adj matrix
def compute_adjugate_old(matrix):
    # Ensure the matrix is square
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    
    # Initialize the cofactor matrix
    cofactor_matrix = np.zeros_like(matrix, dtype=float)
    
    # Iterate over the matrix to compute the cofactor of each element
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Create a submatrix excluding the current row and column
            submatrix = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            
            # Compute the cofactor
            cofactor_matrix[i, j] = ((-1) ** (i + j)) * np.linalg.det(submatrix)
    
    # Transpose the cofactor matrix to get the adjugate matrix
    adjugate_matrix = cofactor_matrix.T
    
    return adjugate_matrix

def compute_cofactor(matrix, i, j):
    # Create a submatrix by removing the i-th row and j-th column
    submatrix = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
    try:
        # Compute the determinant of the submatrix
        det = scipy.linalg.det(submatrix)
    except np.linalg.LinAlgError:
        # Handle the case where the submatrix is singular
        det = 0
    # Compute the cofactor
    cofactor = ((-1) ** (i + j)) * det
    return cofactor

def compute_adjugate_scipy(matrix):
    size = matrix.shape[0]
    cofactor_matrix = np.zeros_like(matrix, dtype=float)

    for i in range(size):
        for j in range(size):
            cofactor_matrix[i, j] = compute_cofactor(matrix, i, j)

    # Transpose the cofactor matrix to get the adjugate
    adjugate_matrix = cofactor_matrix.T
    return adjugate_matrix

def compute_adjugate_pybind11(matrix):
    # this use cpp module under Cpp_lib directory
    adjugate_matrix = adjugate_module.compute_adjugate(matrix)
    return adjugate_matrix

# 
# def compute_adjugate_scipy(matrix):
#     det = np.linalg.det(matrix)
#     try:
#         inv_matrix = np.linalg.inv(matrix)
#     except np.linalg.LinAlgError:
#         inv_matrix = np.zeros_like(matrix)
#         print(' replaced by ones due to singular matrix')
#     
#     # The adjugate is the transpose of the determinant multiplied by the inverse
#     adjugate = det * inv_matrix.T
#     return adjugate
# 


# solve Y for tau=Ytheta
def regressor(tau, theta):

    if np.linalg.norm(theta) != 0:
        # Initialize Y as a zero matrix with dimensions n x m
        Y = np.zeros((len(tau), len(theta)))
    
        # Calculate the scaling factor based on the norm of theta
        scaling_factor = np.linalg.norm(theta)**2
    
        # Construct Y such that Y theta = tau, taking into account the correct scaling
        for i in range(len(theta)):
            Y[:, i] = tau.squeeze() * (theta[i] / scaling_factor)

    
        # Verify the result by calculating the predicted tau
    return Y
 
def regularizing_operator(s, alpha):

    if type(alpha) == float:
        alpha = np.array([alpha] * len(s))
    else:
        if len(alpha) != len(s):
            print('input error')
            return np.nan

    reg_s = np.array([])
    for idx, s_elem in enumerate(s):
        reg_s = np.append(reg_s, np.abs(s_elem)**alpha[idx] * np.sign(s_elem))

    return reg_s


def trucatedSVDbyPercentage(A, variance_thre=0.5):
    # Desired percentage of variance to retain
    #variance_pct = 90  # for example, 90%

    # Step 1: Perform full SVD
    U, Sigma, VT = scipy.linalg.svd(A)
    
    # Step 2: Calculate total variance
    total_variance = np.sum(Sigma**2)
    # Step 3: Determine the number of components to keep
    variance_sum = 0
    num_components = 0
    for s in Sigma:
        variance_sum += s**2
        num_components += 1
        if (variance_sum / total_variance) >= variance_thre:
            break

    # Step 4: Perform Truncated SVD with the determined number of components
    #svd = TruncatedSVD(n_components=num_components)
    #A_reduced = svd.fit_transform(A)

    # Components (V^T)
    #Vt_r = svd.components_
    Vt_r = VT[:num_components, :]
    # Singular values (Σ)
    #Sigma_r = np.diag(svd.singular_values_)
    Sigma_r = Sigma[:num_components]
    # Construct U from A_reduced and Sigma
    #U_r = A_reduced @ np.linalg.inv(Sigma_r)
    U_r = U[:, :num_components]

    return U_r, Sigma_r, Vt_r

# check if Sigma has significant information
def checkSigmaSignificance(Sigma, threshold = 1e-5):
    # if Sigma is a zero matrix
    #is_zero_matrix = np.all(Sigma == 0)

    # Check for significant information (set your own threshold)
    # threshold = 0.01  # Example threshold
    no_significant_information = np.all(np.abs(Sigma) < threshold)

    return no_significant_information

def drem_adapt(theta_hat, tau_f, Y_f, Gamma, alpha, phi0mode=False, phieps=1e-2, hardthre=False):
    # Eq 33 - Eq. 37
    phi = np.linalg.det(Y_f)
    # adjYf = compute_adjugate_scipy(Y_f)
    if (phi0mode == True) & (phi < phieps):
        U, S, Vt = trucatedSVDbyPercentage(Y_f, variance_thre=0.8)

        if checkSigmaSignificance(S): 
            # S has no significant information
            Sigma_inv = np.zeros((len(S), len(S)))
            Overall = np.zeros((len(Gamma), len(Gamma)))
        else:
            # S has sufficient information
            Sigma_inv = np.diag(1/S)
            Overall = np.eye(len(Gamma)) * 1e-15
        f_theta = Overall @ Gamma @ Vt.T @ (Vt @ theta_hat - Sigma_inv @ U.T @ tau_f)

        # hard thresholding to avoid overflow
        if hardthre == True:
            f_theta[abs(f_theta)>1e2] = np.sign( f_theta[abs(f_theta)>1e2] )*1e2
        
    else:
        adjYf = compute_adjugate_pybind11(Y_f)
        phi_adjYf = phi*adjYf # do this first
        phi_tau_e = phi_adjYf @ tau_f
        f_theta = Gamma @ regularizing_operator( ((phi * phi) * theta_hat - phi_tau_e) , alpha)
    return f_theta, phi*phi



###############################################################
# ## below is old ones
# def approximate_adjugate(A, epsilon=1e-5):
#     A_reg = A + epsilon * np.eye(A.shape[0])  # Regularize A
#     det_A_reg = np.linalg.det(A_reg)  # Compute determinant of regularized A
#     inv_A_reg = np.linalg.inv(A_reg)  # Compute inverse of regularized A
#     adjugate_approx = det_A_reg * inv_A_reg.T  # Transpose of the cofactor matrix
#     return adjugate_approx
# 
# # get adj matrix
# def compute_adjugate(matrix):
#     # Ensure the matrix is square
#     assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
#     
#     # Initialize the cofactor matrix
#     cofactor_matrix = np.zeros_like(matrix, dtype=float)
#     
#     # Iterate over the matrix to compute the cofactor of each element
#     for i in range(matrix.shape[0]):
#         for j in range(matrix.shape[1]):
#             # Create a submatrix excluding the current row and column
#             submatrix = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
#             
#             # Compute the cofactor
#             cofactor_matrix[i, j] = ((-1) ** (i + j)) * np.linalg.det(submatrix)
#     
#     # Transpose the cofactor matrix to get the adjugate matrix
#     adjugate_matrix = cofactor_matrix.T
# 
#     #adjugate_matrix = np.array(sympy.Matrix(matrix).adjugate())
#     return adjugate_matrix
# 