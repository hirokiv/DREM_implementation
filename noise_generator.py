import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math 
from scipy.interpolate import interp1d


# predetermined noise signal


# compute second derivative of noise from previous two signals
def calculate_realtime_acceleration(positions, dt):
    # Initialize an array to hold the acceleration values
    accelerations = np.zeros(len(positions))
    
    # Calculate acceleration using backward difference method
    # for all points except the first two, as they don't have enough previous data
    for i in range(2, len(positions)):
        accelerations[i] = (positions[i] - 2*positions[i-1] + positions[i-2]) / dt**2
    
    # The first two acceleration values can't be calculated using this method
    # because there aren't enough previous points, so they are set to None (or NaN)
    accelerations[0] = 0  # Not enough data to compute acceleration
    accelerations[1] = 0  # Not enough data to compute acceleration
    
    return accelerations

# Parameters
def white_noise_derivative(n_samples=100, tbegin=0, tend=10, dbgfig=0, sigma_o=1.0):
    #n_samples = 1000
    time = np.linspace(tbegin, tend, n_samples)
    dt = time[1] - time[0]  # Time step
    
    # Generate white noise
    white_noise = np.random.normal(0, sigma_o, n_samples)
    
    # Compute the first derivative (approximation)
    first_derivative = np.diff(np.insert(white_noise, 0, 0)) / dt
    
    # Compute the second derivative (approximation)
    # Note: The second derivative will have one less point than the first derivative,
    # because np.diff reduces the length by 1 each time it's applied.
    second_derivative = calculate_realtime_acceleration(white_noise, dt)
    #second_derivative = np.diff(np.insert(first_derivative, 0, 0)) / dt
    
    if dbgfig==1:
        # Plotting
        plt.figure(figsize=(12, 4))
        
        plt.subplot(3, 1, 1)
        plt.plot(time, white_noise, label='White Noise')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(time, first_derivative, label='First Derivative', color='orange')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(time, second_derivative, label='Second Derivative', color='green')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    return white_noise, first_derivative, second_derivative, time


def wrap_interp_func(time, noise_info):

    wn = noise_info['wn']
    fd = noise_info['fd']
    sd = noise_info['sd']

    # make list of interpolation function from precomuputed noise
    def interp_func(time, white_noise, first_derivative, second_derivative): 
        # Create interpolation functions
        # For white noise, we use the original time array

        noise_time = np.linspace(time[0], time[-1], len(white_noise)) 

        interp_wn = interp1d(noise_time, white_noise, kind='linear')
        
        # For the first derivative, we need to exclude the last time point due to np.diff
        interp_fd = interp1d(noise_time, first_derivative, kind='linear')
        
        # For the second derivative, we need to exclude the last two time points
        interp_sd = interp1d(noise_time, second_derivative, kind='linear')
    
        return interp_wn, interp_fd, interp_sd

    #wnt, fdt, sdt = np.zeros(3), np.zeros(3), np.zeros(3)
    wnt, fdt, sdt = [], [], [] 
    for idx in np.arange(len(wn)):
        wn_temp, fd_temp, sd_temp = interp_func(time, wn[idx], fd[idx], sd[idx])
        wnt.append(wn_temp)
        fdt.append(fd_temp)
        sdt.append(sd_temp)


    noise_interp_dict = {'wn' : wnt, 
                        'fd' : fdt, 
                        'sd' : sdt
                        }
    return noise_interp_dict

def parse_noise_interp_dict(at_t, noise_interp_dict):

    
    wn_at_t, fd_at_t, sd_at_t = np.zeros(3), np.zeros(3), np.zeros(3)

    for idx in np.arange(len(noise_interp_dict['wn'])):
        wn_at_t[idx] = noise_interp_dict['wn'][idx](at_t)
        fd_at_t[idx] = noise_interp_dict['fd'][idx](at_t)
        sd_at_t[idx] = noise_interp_dict['sd'][idx](at_t)

    return wn_at_t, fd_at_t, sd_at_t


def interp_noise(t_example, time, interp_wn, interp_fd, interp_sd):
   
    # Example time t for interpolation
    # t_example = 5.5  # You can change this value to any time within the range [0, 10]
    
    # Interpolate values at time t_example
    wn_at_t = interp_wn(t_example)
    fd_at_t = interp_fd(t_example)
    sd_at_t = interp_sd(t_example)
    
    return wn_at_t, fd_at_t, sd_at_t

# wn, fd, sd, time = white_noise_derivative(n_samples=100, tbegin=0, tend=10)
# at_t = 0.5
# wnt, fdt, sdt = interp_noise(at_t, time, wn, fd, sd)

def stack_noise(stack_ndim, n_samples=100, tbegin=0, tend=10, sigma_o=0.01, dbgfig=0):

    wn_ndim, fd_ndim, sd_ndim = np.array([]), np.array([]), np.array([])

    for idx in np.arange(stack_ndim):
        wn, fd, sd, time = white_noise_derivative(n_samples=n_samples, tbegin=tbegin, tend=tend, dbgfig=dbgfig, sigma_o=sigma_o)
        wn_ndim = np.append(wn_ndim, wn)
        fd_ndim = np.append(fd_ndim, fd)
        sd_ndim = np.append(sd_ndim, sd)

    wn_ndim = wn_ndim.reshape(stack_ndim, n_samples)
    fd_ndim = fd_ndim.reshape(stack_ndim, n_samples )
    sd_ndim = sd_ndim.reshape(stack_ndim, n_samples )

    noise_info = {'wn' : wn_ndim, # white noise
                  'fd' : fd_ndim, # first derivative
                  'sd' : sd_ndim, # first derivative
                  'time' : time,
               }
    return noise_info 

def interp_stack_noise(at_t, noise_info):
    wn, fd, sd, time  = noise_info

    wnt, fdt, sdt = np.zeros(3), np.zeros(3), np.zeros(3)

    for idx in np.arange(len(wn)):
        wnt[idx], fdt[idx], sdt[idx] = interp_noise(at_t, time, wn[idx], fd[idx], sd[idx])

    return wnt, fdt, sdt

# def interp_dict_noise_lib(at_t, noise_info):
#     wn  = noise_info['wn']
#     fd = noise_info['fd']
#     sd = noise_info['sd']
#     time  = noise_info['time']
# 
#     wnt, fdt, sdt = np.zeros(3), np.zeros(3), np.zeros(3)
# 
#     for idx in np.arange(len(wn)):
#         wnt[idx], fdt[idx], sdt[idx] = interp_noise(at_t, time, wn[idx], fd[idx], sd[idx])
# 
#     return wnt, fdt, sdt


if __name__ == '__main__':
    stack_ndim = 3 #ndim
    noise_info =  stack_noise(stack_ndim, n_samples=100, tbegin=0, tend=10, dbgfig=1)
    
    interp_stack_noise(5, noise_info)
    #quantize_stack_noise(5, noise_info)