import json
import numpy as np

class Experiment:
    def __init__(self, sequence3D, exp_name, DREMON=True, sig_o=0.000, THREON=False, TMAX=2.0, gain_multiple=10,
                  INTEGRALERROR=False, Lambda_i=10, ASYMDREM=False, gain_Sigma_mod=0.1,  axx=700, 
                  SWITCHFLAG=False, ext_F=False, EXT_F_GAIN=1, CONTROLFLAG=False, b2=0.2, b3=0.3, b4=0.4, phi0mode=False,
                  TAUTHRE=np.inf, eval_freq=20, q_d=np.array([np.pi/2, -np.pi/2, 0]), theta_hat0=np.zeros(12), pseudo_tauf=False ):

        self.set_defaults()
        self.set_external_signal(sequence3D)

        self.gain_multiple = gain_multiple

        self.f_const['exp_name'] = exp_name
        self.f_const['DREMON'] = DREMON
        self.f_const['sig_o'] = sig_o
        self.f_const['THREON'] = THREON
        self.f_const['TMAX'] = TMAX
        self.f_const['INTEGRALERROR'] = INTEGRALERROR
        self.f_const['ASYMDREM'] = ASYMDREM
        self.f_const['ext_F'] = ext_F
        self.f_const['axx'] = axx # gain
        self.f_const['SWITCHFLAG'] = SWITCHFLAG
        self.f_const['EXT_F_GAIN'] = EXT_F_GAIN
        self.f_const['CONTROLFLAG'] = CONTROLFLAG
        self.f_const['b2'] = b2 # gain
        self.f_const['b3'] = b3 # gain
        self.f_const['b4'] = b4 # gain
        self.f_const['phi0mode'] = phi0mode
        self.f_const['tau_thre'] = TAUTHRE 
        self.f_const['eval_freq'] =  eval_freq
        self.f_const['q_d'] = q_d 
        self.f_const['theta_hat0'] =  theta_hat0

        self.control_gains['Sigma_mod'] = gain_Sigma_mod
        self.control_gains['Lambda_i'] = Lambda_i


        self.f_const['pseudo_tauf'] = pseudo_tauf

        # change external input gain if set
        if self.f_const['EXT_F'] == True:
            self.sequence3D.change_gain(self.f_const['Ext_F_GAIN'])


    def return_params(self):
        return self.f_const, self.control_gains

    def set_defaults(self):
        ##  default parameters
        # initial estimate of theta0

        theta = np.array([6.3922, 1.4338, 0.0706, 0.0653, 2.4552, 0.2868, 113.6538, 46.1168, 2.0993, 2.6, 2.5, 1.5])
        theta_hat0 = 0.5 * theta
        pdim = len(theta_hat0)
        ndim = 3 # state vector dim
        nfilter = 4 # including Y1
        #q_d = np.array([np.pi/2, -np.pi/2, 0]) # desired position
        #init_y = np.array([0, 0, np.pi/2, 0, 0, 0])

        self.f_const = {'lambda_phi' : 1.0, 
                        'b2'    : 0.2, 
                        'b3'    : 0.3, 
                        'b4'    : 0.4, 
                        'a2'    : 700.0, 
                        'a3'    : 700.0, 
                        'a4'    : 700.0, 
                        'nfilter' : nfilter, 
                        'ndim' : ndim, 
                        'nrows' : nfilter * ndim, 
                        'ncols' : pdim, 
                        'pdim' : pdim, 
                        'theta' : theta,
                        'TMAX' : 2, #  seconds
                        'SWITCHFLAG' : False,   # if do regime switching
                        'THREON' : False,   # if do thresholding for estimated value. Either 1. 'const' constant threshold, 2. 'Sigma_mod' Sigma modification
                        'THRERT' : np.array([0.0, 2.0]),  # active when making thresholding on
                        'DREMON' : True,        # if use drem adaptation f_theta
                        'INTEGRALERROR' : False,        # if use drem adaptation f_theta
                        'q_d' : np.array([np.pi/2, -np.pi/2, 0]), # q_d,    # desired position
                        'noise_freq' : 5,  # hz
                        'eval_freq' : 100,  # hz
                        'sigma_o' : 0.01,
                        'init_y' : np.array([0, 0, np.pi/2, 0, 0, 0]), # init_y,
                        'theta_hat0' : theta_hat0,
                        'EXT_F' : False, # if you input external force as MLS (M-系列)
                        'EXT_F_GAIN' : [1, 1, 1], # if you input external force as MLS (M-系列)
                        'ASYMDREM' : False, # update on DREM noise reduction asymptotically in RLS
                        'CONTROLFLAG' : False, # define baseline tau
                        'PHI0MODE' : False, # behavior of assimilation when phi close to 0
                        'PHI0EPS' : 0.01, # behavior of assimilation when phi close to 0
                        'TAUTHRE' : np.inf, # behavior of assimilation when phi close to 0
                        }



        ###########################################
        # Initial conditions: joint angles and velocities
        tau = np.zeros(ndim)
        Lambda_i = 5 * np.eye(ndim)  # lambda for integral error compensation
        Lambda = 10 * np.eye(ndim) 
        Kv =     10 * np.eye(ndim)
        Psi =    10 * np.eye(pdim)
        Gamma =  10 * np.eye(pdim)
        Sigma_mod =  0.1 * np.eye(pdim)
        alpha = 0.50001
        
        self.control_gains = {
            'Lambda' : Lambda, 
            'Lambda_i' : Lambda_i, 
            'Kv' : Kv,
            'Psi' : Psi,
            'Gamma' : Gamma, 
            'alpha' : alpha,
            'Sigma_mod' : Sigma_mod,
        }


    def set_external_signal(self, sequence3D):
        self.sequence3D = sequence3D
        self.control_gains['Extern'] = sequence3D.to_json()



    # Legacy
    # def to_dict(self):
    #     return {
    #         "exp_name": self.exp_name,
    #         "DREMON": self.DREMON,
    #         "sig_o": self.sig_o,
    #         "THREON": self.THREON,
    #         "TMAX": self.TMAX,
    #         "gain_multiple": self.gain_multiple,
    #         "INTEGRALERROR": self.INTEGRALERROR,
    #         "Lambda_i": self.Lambda_i,
    #         "ASYMDREM": self.ASYMDREM,
    #         "gain_Sigma_mod": self.gain_Sigma_mod,
    #         "ext_F": self.ext_F,
    #         "axx": self.axx,
    #         "SWITCHFLAG": self.SWITCHFLAG,
    #         "EXT_F_GAIN": self.EXT_F_GAIN,
    #         "CONTROLFLAG": self.CONTROLFLAG,
    #         "b2": self.b2,
    #         "b3": self.b3,
    #         "b4": self.b4,
    #         "phi0mode": self.phi0mode,
    #         "TAUTHRE": self.tau_thre,
    #         "eval_freq": self.eval_freq,
    #         "q_d": self.q_d,
    #         "theta_hat0": self.theta_hat0,
    #     }

    @staticmethod
    def export_to_json(exp_instances, filename='config/experiment_configs.json'):
        with open(filename, 'w') as f:
            json.dump([exp.to_dict() for exp in exp_instances], f, indent=4)


    # def return_dict(self):
    #     # modify default params for experiments
    #     self.f_const['TMAX'] = self.TMAX #  seconds
    #     self.f_const['DREMON'] = self.DREMON
    #     self.f_const['sigma_o'] = self.sig_o
    #     self.f_const['THREON'] = self.THREON 
    #     self.f_const['INTEGRALERROR'] = self.INTEGRALERROR
    #     self.f_const['EXT_F'] = self.ext_F
    #     self.control_gains['Lambda'] = self.gain_multiple * np.eye(self.ndim) 
    #     self.control_gains['Kv'] =     self.gain_multiple * np.eye(self.ndim)
    #     self.control_gains['Psi'] =    self.gain_multiple * np.eye(self.pdim)
    #     self.control_gains['Gamma'] =  self.gain_multiple * np.eye(self.pdim)
    #     self.control_gains['Lambda_i'] = self.Lambda_i * np.eye(self.ndim) 
    #     self.control_gains['Sigma_mod'] =  gain_Sigma_mod * np.eye(self.pdim)
    #     self.f_const['ASYMDREM'] = self.ASYMDREM
    #     self.f_const['a2'] = self.axx
    #     self.f_const['a3'] = self.axx
    #     self.f_const['a4'] = self.axx
    #     self.f_const['b2'] = self.b2
    #     self.f_const['b3'] = self.b3
    #     self.f_const['b4'] = self.b4
    #     self.f_const['q_d'] = self.q_d
    #     self.f_const['SWITCHFLAG'] =  self.SWITCHFLAG
    #     self.f_const['Ext_F_GAIN'] =  self.EXT_F_GAIN 
    #     self.f_const['CONTROLFLAG'] = self.CONTROLFLAG 
    #     self.f_const['PHI0MODE'] =  self.phi0mode
    #     self.f_const['TAUTHRE'] =   self.tau_thre
    #     self.f_const['eval_freq'] = self.eval_freq 
    #     self.f_const['theta_hat'] = self.theta_hat0 


if __name__ == '__main__':
    exp_instances = [Experiment(*exp) for exp in exp_list]
    
    # Exporting to JSON
    Experiment.export_to_json(exp_instances)