import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math 
import scipy

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


def drem_adapt(theta_hat, tau_f, Y_f, Gamma, alpha):
    # Eq 33 - Eq. 37
    phi = np.linalg.det(Y_f)
    # adjYf = compute_adjugate_scipy(Y_f)
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