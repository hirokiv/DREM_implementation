import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pickle
from DREM_routine import robot_dynamics_w_control_adaptation, tau_reconstructor

def check_dir(directory_path):
    # Use os.makedirs() to create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory '{directory_path}' was created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


#def analysis_result(t,y,q_d,theta_hat0,theta,ndim,nfilter,SWITCHFLAG,save=False,savepath='',figshow=1):
def analysis_result(t,y,theta_hat0,f_const, control_gains, noise_interp_dict, save=False,savepath='',figshow=1, ):

    q_d = f_const['q_d']
    theta = f_const['theta']
    ndim = f_const['ndim']
    nfilter = f_const['nfilter']
    SWITCHFLAG = f_const['SWITCHFLAG']


    img_savepath = savepath
    if save==True:
        check_dir(img_savepath)

    # Plot the results
    plt.figure(figsize=(12, 8))
    for i in range(ndim):
        plt.subplot(3, 1, i+1)
        plt.plot(t, y[i], label=f'Joint {i+1} Position (rad)')
        plt.plot(t, y[i+3], label=f'Joint {i+1} Velocity (rad/s)')
        plt.plot(t, [q_d[i]]*len(t), label=f'Joint {i+1} Target (rad)')
        plt.legend()
    plt.xlabel('Time (s)')
    plt.grid()
    
    if save == True:
        savepath = img_savepath + '/img_control_result.png'
        plt.savefig(savepath)
    if figshow == 1:
        plt.show()
    plt.close()

    #####################
    # Plot the tau_f results
    #####################
    plt.figure(figsize=(12, 8))
    tau_idx = 2 * ndim + f_const['nrows'] + f_const['nrows']*f_const['ncols']  # y, theta_hat, Yf
    for i in range(f_const['nrows']) :
        plt.subplot(12, 1, i+1)
        plt.plot(t, y[i+tau_idx], label=f'Tau_f {i+1} history ')
        plt.legend()
        plt.grid()
    plt.xlabel('Time (s)')
    
    if save == True:
        savepath = img_savepath + '/img_tauf_result.png'
        plt.savefig(savepath)
    if figshow == 1:
        plt.show()
    plt.close()




    #####################
    # Reconstruct tauf from Yf and true theta
    tauf_recon = np.zeros((f_const['nrows'], len(t)))
    for idx in np.arange(len(t)):
        #theta_hat = y[6:6+len(theta)][idx] 
        Yf_recon =  y[6+len(theta):6+len(theta)+f_const['nrows']*f_const['ncols']][:,idx].reshape([f_const['nrows'], f_const['ncols']]) 
        tauf_recon[:, idx] = Yf_recon @ theta
    
    plt.figure(figsize=(12, 8))
    tau_idx = 2 * ndim + f_const['nrows'] + f_const['nrows']*f_const['ncols']  # y, theta_hat, Yf
    for i in range(f_const['nrows']) :
        plt.subplot(12, 1, i+1)
        plt.plot(t, tauf_recon[i, :], label=f'Tau_f reconstructed {i+1} history ' )
        plt.legend(loc='right')
        plt.grid()
    plt.xlabel('Time (s)')

    if save == True:
        savepath = img_savepath + '/img_tauf_reconstruct_result.png'
        plt.savefig(savepath)
    if figshow == 1:
        plt.show()
    plt.close()


    #####################
    # Plot real tau-s
    #####################
    true_tau = np.zeros((f_const['ndim'], len(t)))
    for idx, temp in enumerate(t):
        true_tau[:, idx] = tau_reconstructor(y[:,idx], temp, f_const, control_gains, noise_interp_dict)
    plt.figure(figsize=(12, 5))
    for i in range(f_const['ndim']) :
        plt.subplot(3, 1, i+1)
        plt.plot(t, true_tau[i,:], label=f'True Tau {i+1} history ')
        plt.legend()
        plt.grid()
    plt.xlabel('Time (s)')
    
    if save == True:
        savepath = img_savepath + '/img_true_tau.png'
        plt.savefig(savepath)
    if figshow == 1:
        plt.show()
    plt.close()


    #####################
    # Assimilation error dynamics
    pdim = len(theta_hat0)
    # Plot the results
    for iter in [0, 1]:
        plt.figure(figsize=(12, 8))
        for i in range(pdim):
    
            # Regime switch setting
            theta_t = [theta[i]]*len(t)
            #theta_t[t>10] = theta_t[t>10] * 0.5
            if SWITCHFLAG:
                theta_t = [theta_t[i] * 0.5 if t[i] > 10 else theta_t[i] for i in range(len(t))]
    
    
            plt.subplot(pdim, 1, i+1)
            plt.plot(t, y[2*ndim+i] - theta_t, label=f'Theta {i+1} error')
            plt.plot(t, [0]*len(t), 'k--') #, label=f'Theta {i+1} error' )
            # plt.plot(t, [theta[i]]*len(t), label=f'Theta {i+1} true')
            plt.legend(loc='right')
            plt.grid()
    
            # magnified case
            if iter ==  1:
                plt.ylim(-0.5*theta[i], 0.5*theta[i])
        plt.xlabel('Time (s)')
     
        if save == True:
            if iter == 0 : 
                savepath = img_savepath + '/img_parameter_estimation.png'
            elif iter == 1:
                savepath = img_savepath + '/img_parameter_estimation_magnified.png'
            plt.savefig(savepath)   
        if figshow == 1:
            plt.show()
        plt.close()

    # Phi computation  
    Yf_list = []
    phi = []
    for idx, tt in enumerate(t):
        Yf_list.append( y[2*ndim + pdim:2*ndim+pdim + (nfilter * ndim)*pdim][:,idx].reshape((nfilter * ndim, pdim)) )
        phi.append(np.linalg.det(Yf_list[-1]))
    
    
    phi2t = np.power(phi,2)
    
    #print(phi2t)
    
    plt.figure(figsize=(12, 8))
    plt.plot(t,phi2t)
    plt.ylabel('\phi^2')
    plt.xlabel('t')
    plt.grid()
     
    if save == True:
        savepath = img_savepath + '/img_phi2.png'
        plt.savefig(savepath)   
    if figshow == 1:
        plt.show()
    plt.close()

class NumpyArrayEncoder(json.JSONEncoder):
    """ Custom encoder for numpy arrays to make them JSON serializable """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return json.JSONEncoder.default(self, obj)

def save_params(parameters,filename='parameters.json',savepath=''):
    # Define your parameters as a dictionary
    # parameters = {
    # 'learning_rate': 0.01,
    # 'batch_size': 32,
    # 'epochs': 10,
    # 'activation': 'relu'
    # }

    # Specify the filename you want to save the parameters to
    filename = savepath + '/' + filename
    
    # Use a with statement to open the file and ensure it gets closed properly
    with open(filename, 'w') as file:
        # Write the parameters dictionary to the file as JSON
        json_data = json.dumps(parameters, indent=4, cls=NumpyArrayEncoder, separators=(',', ':'), )
        # json.dump(parameters, file, indent=4, cls=NumpyArrayEncoder)
        file.write(json_data)
    
    print(f'Parameters have been saved to {filename}.')


def save_as_pickle(variable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)
    print(f"Variable successfully saved to {filename}")
