import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math 




# Function to compute the matrices in Equation (66)
def compute_matrices(q, q_dot, theta):
    # Unpack joint angles and velocities
    q1, q2, q3             = q
    q1_dot, q2_dot, q3_dot = q_dot
    q12_dot                = q1_dot + q2_dot
    q23_dot                = q2_dot + q3_dot
    q123_dot               = q1_dot + q2_dot + q3_dot

    
    # Compute trigonometric functions of joint angles
    s2, s3          = np.sin(q2), np.sin(q3)
    s12, s23, s123  = np.sin(q1 + q2), np.sin(q2 + q3), np.sin(q1 + q2 + q3)
    c1, c2, c3, c23 = np.cos(q1), np.cos(q2), np.cos(q3), np.cos(q2 + q3)
    
    # Compute matrices and vectors (placeholders, actual computation depends on robot's specifics)
    H = np.array([[theta[0] + 2*theta[1]*s2 + 2*theta[2]*c3 + 2*theta[3]*s23, theta[1]*s2 + 2*theta[2]*c3 + theta[3]*s23 + theta[4], theta[2]*c3 + theta[3]*s23 + theta[5]],
                  [theta[1]*s2 + 2*theta[2]*c3 + theta[3]*s23 + theta[4] ,    2*theta[2]*c3 + theta[4],                              theta[2]*c3 + theta[5]               ],
                  [theta[2]*c3 + theta[3]*s23 + theta[5],                     theta[2]*c3 + theta[5],                                theta[5]                             ]])
    
    C = np.array([[theta[1]*c2*q2_dot - theta[2]*s3*q3_dot + theta[3]*c23*q23_dot,  theta[1]*c2*q12_dot - theta[2]*s3*q3_dot + theta[3]*c23*q123_dot, -theta[2]*s3*q123_dot + theta[3]*c23*q123_dot],
                  [- theta[1]*c2*q1_dot - theta[2]*s3*q3_dot - theta[3]*c23*q1_dot, -theta[2]*s3*q3_dot,                                              -theta[2]*s3*q123_dot                        ],
                  [theta[2]*s3*q12_dot - theta[3]*c23*q1_dot,                       theta[2]*s3*q12_dot,                                              0                                            ]])
    
    
    D = np.array([[theta[9], 0,                 0         ], 
                  [0,        theta[10] ,        0         ],
                  [0,        0,                 theta[11] ]])

    g = np.array([theta[6]*c1 + theta[7]*s12 + theta[8]*s123, theta[7]*s12 + theta[8]*s123, theta[8]*s123])

    return H, C, D, g

def compute_Y(q, q_dot, q_r_dot, q_r_ddot):
    # Unpack joint angles and velocities
    q1, q2, q3             = q
    q1_dot, q2_dot, q3_dot = q_dot
    q12_dot                = q1_dot + q2_dot
    q23_dot                = q2_dot + q3_dot
    q123_dot               = q1_dot + q2_dot + q3_dot

    q1_r_dot, q2_r_dot, q3_r_dot    = q_r_dot
    q1_r_ddot, q2_r_ddot, q3_r_ddot = q_r_ddot

    
    # Compute trigonometric functions of joint angles
    s2, s3          = np.sin(q2), np.sin(q3)
    s12, s23, s123  = np.sin(q1 + q2), np.sin(q2 + q3), np.sin(q1 + q2 + q3)
    c1, c2, c3, c23 = np.cos(q1), np.cos(q2), np.cos(q3), np.cos(q2 + q3)
    
    # # Compute matrices and vectors (placeholders, actual computation depends on robot's specifics)

    #    theta 0    theta 1                                                                     theta 2                                                                                                           theta 3               
    #      theta 4              theta 5               theta 6     theta 7     theta 8     theta 9     theta 10    theta 11             
    Y = np.array([
        [q1_r_ddot, 2*s2*q1_r_ddot + s2*q2_r_ddot + c2*q2_dot*q1_r_dot + c2*q12_dot*q2_r_dot,   2*c3*q1_r_ddot + 2*c3*q2_r_ddot + c3*q3_r_ddot - s3*q3_dot*q1_r_dot - s3*q3_dot*q2_r_dot - s3*q123_dot*q3_r_dot,  2*s23*q1_r_ddot + s23*q2_r_ddot + s23*q3_r_ddot + c23*q23_dot*q1_r_dot + c23*q123_dot*q2_r_dot + c23*q123_dot*q3_r_dot,
         q2_r_ddot,             q3_r_ddot,             c1,         s12,       s123,          q1_r_dot,   0,          0 ],
        [0,         s2*q1_r_ddot - c2*q1_dot*q2_r_dot,                                          2*c3*q1_r_ddot + 2*c3*q2_r_ddot + c3*q3_r_ddot - s3*q3_dot*q1_r_dot - s3*q3_dot*q2_r_dot - s3*q123_dot*q3_r_dot,  s23*q1_r_ddot - c23*q1_dot*q1_r_ddot,
         q1_r_ddot + q2_r_ddot, q3_r_ddot,             0,         s12,        s123,          0,          q2_r_dot,   0 ],
        [0,         0,                                                                          c3*q1_r_ddot + c3*q2_r_ddot + s3*q12_dot*q1_r_dot + s3*q12_dot*q2_r_dot,                                         s23*q1_r_ddot - c23*q1_dot*q1_r_dot,
         0,         q1_r_ddot + q2_r_ddot + q3_r_ddot, 0,         0,          s123,          0,          0,          q3_r_dot ]
    ])

    return Y




# Dynamics function for the ODE solver
def robot_dynamics(t, y, theta, tau):
    q = y[:3]
    q_dot = y[3:]
    
    H, C, D, g = compute_matrices(q, q_dot, theta)
    
    # Assuming no external torque for simplicity
    
    # Compute joint accelerations
    q_ddot = np.linalg.inv(H) @ (tau - C @ q_dot - D @ q_dot - g)

    
    return np.concatenate((q_dot, q_ddot)), H,C,D,g

# error dynamics of controller
def error_dynamics(y, q_d):
    # for fixed target, simply q_dot
    q = y[:3]
    q_dot = y[3:]
    e = q - q_d
    e_dot = q_dot

    # H, C, D, g = compute_matrices(q, q_dot, theta)
    # # Assuming no external torque for simplicity
    # # Compute joint accelerations
    # q_ddot = np.linalg.inv(H) @ (tau - C @ q_dot - D - g)
    # e_ddot = q_ddot

    return np.array(e_dot), np.array(e)


def filter_dynamics(Y_f, Y_t, tau_f, tau_t, f_const):

    ndim = f_const['ndim']
    dYf =   np.zeros((f_const['nrows'], f_const['ncols']))
    dtauf = np.zeros(f_const['nrows'])

    dYf[      :  ndim, :] = - f_const['lambda_phi'] * Y_f[      :  ndim, :] + f_const['lambda_phi'] * Y_t
    dYf[1*ndim:2*ndim, :] = - f_const['b2']         * Y_f[1*ndim:2*ndim, :] + f_const['a2']         * Y_f[:ndim, :]
    dYf[2*ndim:3*ndim, :] = - f_const['b3']         * Y_f[2*ndim:3*ndim, :] + f_const['a3']         * Y_f[:ndim, :]
    dYf[3*ndim:4*ndim, :] = - f_const['b4']         * Y_f[3*ndim:4*ndim, :] + f_const['a4']         * Y_f[:ndim, :]

    dtauf[      :  ndim]  = - f_const['lambda_phi'] * tau_f[      :  ndim] + f_const['lambda_phi'] * tau_t 
    dtauf[1*ndim:2*ndim]  = - f_const['b2']         * tau_f[1*ndim:2*ndim] + f_const['a2']         * tau_f[:ndim]
    dtauf[2*ndim:3*ndim]  = - f_const['b3']         * tau_f[2*ndim:3*ndim] + f_const['a3']         * tau_f[:ndim]
    dtauf[3*ndim:4*ndim]  = - f_const['b4']         * tau_f[3*ndim:4*ndim] + f_const['a4']         * tau_f[:ndim]

    return dYf, dtauf
