import json

class Experiment:
    def __init__(self, exp_name, DREMON=None, sig_o=None, THREON=None, TMAX=None, gain_multiple=None, INTEGRALERROR=None, Lambda_i=None, ASYMDREM=None, gain_Sigma_mod=None, ext_F=None, axx=None, SWITCHFLAG=None, EXT_F_GAIN=None, CONTROLFLAG=None, b2=None, b3=None, b4=None, phi0mode=False):
        self.exp_name = exp_name
        self.DREMON = DREMON
        self.sig_o = sig_o
        self.THREON = THREON
        self.TMAX = TMAX
        self.gain_multiple = gain_multiple
        self.INTEGRALERROR = INTEGRALERROR
        self.Lambda_i = Lambda_i
        self.ASYMDREM = ASYMDREM
        self.gain_Sigma_mod = gain_Sigma_mod
        self.ext_F = ext_F
        self.axx = axx # gain
        self.SWITCHFLAG = SWITCHFLAG
        self.EXT_F_GAIN = EXT_F_GAIN
        self.CONTROLFLAG = CONTROLFLAG
        self.b2 = b2 # gain
        self.b3 = b3 # gain
        self.b4 = b4 # gain
        self.phi0mode = phi0mode

    def to_dict(self):
        return {
            "exp_name": self.exp_name,
            "DREMON": self.DREMON,
            "sig_o": self.sig_o,
            "THREON": self.THREON,
            "TMAX": self.TMAX,
            "gain_multiple": self.gain_multiple,
            "INTEGRALERROR": self.INTEGRALERROR,
            "Lambda_i": self.Lambda_i,
            "ASYMDREM": self.ASYMDREM,
            "gain_Sigma_mod": self.gain_Sigma_mod,
            "ext_F": self.ext_F,
            "axx": self.axx,
            "SWITCHFLAG": self.SWITCHFLAG,
            "EXT_F_GAIN": self.EXT_F_GAIN,
            "CONTROLFLAG": self.CONTROLFLAG,
            "b2": self.b2,
            "b3": self.b3,
            "b4": self.b4,
            "phi0mode": self.phi0mode,
        }

    @staticmethod
    def export_to_json(exp_instances, filename='config/experiment_configs.json'):
        with open(filename, 'w') as f:
            json.dump([exp.to_dict() for exp in exp_instances], f, indent=4)

if __name__ == '__main__':
    exp_instances = [Experiment(*exp) for exp in exp_list]
    
    # Exporting to JSON
    Experiment.export_to_json(exp_instances)