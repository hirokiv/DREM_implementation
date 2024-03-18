import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from noise_generator import  stack_noise, interp_stack_noise
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math 
from noise_generator import wrap_interp_func, parse_noise_interp_dict

from robot_dynamics import compute_Y, robot_dynamics, error_dynamics, filter_dynamics
from matrix_operation import  drem_adapt, regularizing_operator, regressor
from pseudo_signal import MaxLengthSequence3D
import json


import time

# List to store elapsed times
elapsed_times = []
elapsed_dict = {'et1': elapsed_times, 
                'et2': elapsed_times, 
                'et3': elapsed_times, 
                'et4': elapsed_times, 
                'et5': elapsed_times, 
}
DEBUGTIME = 0
DSTARTTIME = 0 #define as global variable

def measure_time_begin():
    global DSTARTTIME, DEBUGTIME
    if DEBUGTIME:
        DSTARTTIME = time.time()  # Start timing

    else:
        pass

def measure_time_end(key):
    global DSTARTTIME, DEBUGTIME, elapsed_dict
    if DEBUGTIME:
        end_time = time.time()  # End timing
        elapsed_dict[key].append(end_time - DSTARTTIME)
    else:
        pass

######### usage ################
#   measure_time_begin()
#   # some func to be measured
#   measure_time_end('et1')


def robot_dynamics_w_control_adaptation(t, ytheta_hat, control_gains, f_const, noise_interp_dict, sequence3D):

    # if (t%0.01) == 0:
    #     print('t = {}'.format(t))


    theta = f_const['theta']
    SWITCHFLAG = f_const['SWITCHFLAG']
    CONTROLFLAG = f_const['CONTROLFLAG']
    DREMON = f_const['DREMON']
    q_d = f_const['q_d']
    Kv = control_gains['Kv']
    Psi = control_gains['Psi']
    Gamma = control_gains['Gamma']
    alpha = control_gains['alpha']
    Lambda = control_gains['Lambda']
    Lambda_i = control_gains['Lambda_i'] # lambda for integral 
    Sigma_mod = control_gains['Sigma_mod'] # sigma modification

    # ytheta_hat will be decomposed to 
    # q, qdot, theta_hat, Y_f, tau_f, e_i, phi2sum, P

    # add small observation noise to position and angle
    y = ytheta_hat[:6]  # + np.append(wn, fd) if system noise
    q = y[:3]
    q_dot = y[3:6]
    theta_hat = ytheta_hat[6:6+len(theta)]
    ####################################################
    # filter dynamics

    Y_f =   ytheta_hat[6+len(theta)                                   : 6+len(theta)+f_const['nrows']*f_const['ncols']].reshape([f_const['nrows'], f_const['ncols']])
    tau_f = ytheta_hat[6+len(theta)+f_const['nrows']*f_const['ncols'] : 6+len(theta)+f_const['nrows']*f_const['ncols'] + f_const['nrows']]
    ### note , exact theta should not be known here ###
    tau_f_true = Y_f @ theta #ytheta_hat[6+len(theta)+f_const['nrows']*f_const['ncols'] : 6+len(theta)+f_const['nrows']*f_const['ncols'] + f_const['nrows']]
    ####################################################
    # error integral
    e_i = ytheta_hat[ 6+len(theta)+f_const['nrows']*f_const['ncols'] + f_const['nrows']:6 + len(theta) + f_const['nrows']*f_const['ncols'] + f_const['nrows'] + f_const['ndim'] ]
    phi2sum = ytheta_hat[ 6 + len(theta) + f_const['nrows']*f_const['ncols'] + f_const['nrows'] + f_const['ndim'] ] # scalar

    pindex =  6 + len(theta) + f_const['nrows']*f_const['ncols'] + f_const['nrows'] + f_const['ndim'] + 1
    P = ytheta_hat[pindex : pindex + f_const['nrows'] * f_const['nrows']  ].reshape([f_const['nrows'], f_const['nrows']]) # Ya.T@Ya


    ####################################################
    measure_time_begin()
    # interpolate from precomputed noise information
    # not sd calculation is too shallow and not used here
    # wn, fd, sd = interp_dict_noise(t, noise_info) # whitenoise, its first derivative and second derivative
    wn, fd, sd = parse_noise_interp_dict(t, noise_interp_dict)

    q_obs     = q + wn
    q_dot_obs = q_dot + fd
    measure_time_end('et1')
    ####################################################


    # regime switching if true
    if SWITCHFLAG == False: 
        pass
    else:
        if t > 10.0: # change parameter
            theta = 0.5 * theta


 

    ####################################################
    measure_time_begin()
    # adding observation noise
    e_dot, e = error_dynamics(np.append(q_obs, q_dot_obs), q_d)
    # integral of error
    ei = e
    qr_dot  = - Lambda@e
    qr_ddot = - Lambda@e_dot
 
    #H_hat, C_hat, D_hat, g_hat = compute_matrices(q, q_dot, theta_hat)
    #Ytheta_hat = H_hat@qr_ddot + C_hat@qr_dot + D_hat@qr_dot + g_hat
    measure_time_end('et2')

    ####################################################
    measure_time_begin()
    # Eq. 21
    # Ya = regressor(Ytheta_hat, theta_hat)  #np.dot(Ytheta_hat.reshape([-1,1]), np.linalg.pinv(theta_hat.reshape([-1,1])).T )
    if DREMON == 'YaZERO':
        Ya = np.zeros(shape=(f_const['ndim'], f_const['ncols']))
    else:
        Ya = compute_Y(q_obs, q_dot_obs , qr_dot, qr_ddot)
    # Ya = compute_Y(q, q_dot, qr_dot, qr_ddot)


    if f_const['INTEGRALERROR']:
        s       = e_dot + Lambda@e + Lambda_i @ e_i
    else:
        s       = e_dot + Lambda@e 


    # s squared sum
    if DREMON in ['DREM_ONLY', True, 'YaZERO']:
        # update theta thorugh Psi
        # force s = 0
        f_theta, phi2 = drem_adapt(theta_hat, tau_f, Y_f, Gamma, alpha, phi0mode=f_const['PHI0MODE'], phieps=1e-2) # Eq. 37
        phi2 = [phi2]
    else:
        # no DREM case
        f_theta = np.zeros(f_const['ncols'])
        phi2 = [0]



    # RLS asymptotic noise reduction
    if f_const['ASYMDREM'] == 'RLS':
        # update f_theta
        f_theta = f_theta / (1 + phi2sum)

    # RLS asymptotic noise reduction
    elif f_const['ASYMDREM'] == 'RLS_P':
        # update f_theta
        #f_theta = f_theta / (1 + phi2sum)
        #theta_hat_dot = theta_hat_dot / (1+phi2sum)

        if np.linalg.matrix_rank(P) < np.min( np.shape(P) ): # case for singular
            #matP = np.eye(f_const['nrows'], f_const['nrows'])
            invP = np.eye(P.shape[0])
        else:
            # Regularization parameter
            lambda_ = np.square( f_const['theta'] )
            ## Identity matrix
            #I = np.eye(P.shape[0])
            # Regularized inverse calculation
            #invP =  np.diag(lambda_ / (np.diag(P) + lambda_)) 
            #invP = np.linalg.inv(np.diag(np.diag(P)) + np.diag(lambda_)) 
            invP = np.linalg.inv(P.T @ P + np.diag(lambda_) ) @ P.T
            # Regularized inverse calculation
            #matP = np.eye(f_const['nrows'], f_const['nrows'])

        dP = Ya.T @ Ya # P = int_0^t Ya(tau).T @ Ya(tau) ) dtau

    else:
        # initialize matP
        invP = np.eye(P.shape[0])
        dP = np.zeros( (f_const['nrows'], f_const['nrows']) )






    if (DREMON in ['DREM_ONLY', 'YaZERO']) or f_const['PHI0MODE']:
        theta_hat_dot =  - Psi @ (                  f_theta ) # Eq. 38
    else:
        theta_hat_dot =  - Psi @ ( invP @ Ya.T @ s + f_theta) # Eq. 38



    measure_time_end('et3')

    ####################################################
    measure_time_begin()
    if (f_const['THREON'] == 'const') | (f_const['THREON'] == True):
        # apply thresholding for assimilation
        theta_hat_dot = theta_dot_thre(theta_hat, theta, f_const['THRERT'], theta_hat_dot)
    elif f_const['THREON'] == 'Sigma_mod':
        # sigma modification
        theta_hat_dot = theta_hat_dot - Sigma_mod @ theta_hat
    else:
        #print('THREON parameter undefined')
        pass

 
    if CONTROLFLAG == 'YaZERO':
        tau =  - Kv @ s  # Eq. 22     
    elif CONTROLFLAG == 'NOISEFEED':
        tau = -Lambda @ Kv @ e_dot
    else:
        tau =  - Kv @ regularizing_operator(s, alpha) + Ya @ theta_hat # Eq. 22     
 
    measure_time_end('et4')

    # external force for sys ID
    if f_const['EXT_F'] == True:
        #new_sequence3D = MaxLengthSequence3D.from_json(control_gains['Extern'])
        #input = MaxLengthSequence3D.from_json(control_gains['Extern'])
        input = sequence3D
        #input = control_gains['Extern']
        tau += input.get_value_at_time(t)
    else:
        pass
 
    ####################################################
    # true dynamics of robot
    y_dot, H,C,D,g = robot_dynamics(t, y, theta, tau )

    # Y_t = regressor(tau, theta_hat)  # current Y for filter 1
    #######  Test implementation for Y(q,qdot,qddot) #
    # Y_t = regressor(H@y_dot[3:] + C@y_dot[:3] + D@y_dot[:3] + g, theta)
    # Y_t = compute_Y(q, q_dot, q_dot, y_dot[3:6]) # to feed for Eq. 25
    q_ddot_obs = y_dot[3:6] + sd
    Y_t = compute_Y(q_obs, q_dot_obs, q_dot_obs, q_ddot_obs) # to feed for Eq. 25

    ##################################################
    measure_time_begin()
    dYf, dtauf = filter_dynamics(Y_f, Y_t, tau_f, tau, f_const)
    measure_time_end('et5')
    ##################################################
 
    return np.concatenate([y_dot, theta_hat_dot, dYf.flatten(), dtauf.flatten(), ei, phi2, dP.flatten()])



# def thresholding(theta_hat, minvalue, maxvalue):
#     if (theta_hat < minvalue) or (theta_hat > maxvalue):
#         dtheta_hat = 0
#     return dtheta_hat
 
def theta_dot_thre(theta_hat, theta, ratio, theta_hat_dot):
    # ratio indicates parameter estimation errror allowance
    # currently only define for theta>0 case

    def are_all_positive(numbers):
        return all(n > 0 for n in numbers)
    
    # Example usage

    if are_all_positive(theta):
        theta_hat_dot [( theta * ratio[0] > theta_hat) & (theta_hat_dot<0) ] = 0
        theta_hat_dot [( theta * ratio[1] < theta_hat) & (theta_hat_dot>0) ] = 0
        return theta_hat_dot

    else:
        raise ValueError("Not all elements are positive.")



if __name__ == '__main__':

    numbers = [1,2,3,4,5]
    estimate = np.random.rand(5) - 1/2
    theta_hat_dot = np.random.rand(5) - 1/2
    theta_dot_thre(estimate, numbers, [0.2, 2], theta_hat_dot)


