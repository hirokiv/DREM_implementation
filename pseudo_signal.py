import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import max_len_seq
from scipy.interpolate import interp1d
import json

def mls(register_length, taps):
    # Initialize the LFSR with all zeros and a single one
    lfsr = [0] * (register_length - 1) + [1]
    sequence = []

    for _ in range(2**register_length - 1):
        feedback = 0
        for tap in taps:
            feedback ^= lfsr[tap - 1]  # XOR the tapped bits
        sequence.append(lfsr.pop())  # Output the bit at the end of the LFSR
        lfsr.insert(0, feedback)  # Insert the feedback bit at the start

    return sequence

class MaxLengthSequence3D:
    def __init__(self, order, seed=42, gain=1):
        if seed is not None:
            np.random.seed(seed)  # Fix the seed for reproducibility

        # Define taps for the LFSR based on the size. These taps are examples.
        #taps = [3, 2]
        # self.sequence_x = mls(size, taps)
        # self.sequence_y = mls(size, taps)
        # self.sequence_z = mls(size, taps)
        self.ndim = 3
        self.gain = [gain] * self.ndim
        self.order = order
        sequence, _ = max_len_seq(self.order)
        self.sequence_x = sequence 
        self.sequence_y = np.roll(sequence, int(np.floor(len(sequence)/3)))
        self.sequence_z = np.roll(sequence, int(np.floor(len(sequence)*2/3)))
        # Create interpolation functions for each dimension
        self.interp_x = interp1d(np.arange(len(self.sequence_x)), self.sequence_x, kind='linear', fill_value="extrapolate")
        self.interp_y = interp1d(np.arange(len(self.sequence_y)), self.sequence_y, kind='linear', fill_value="extrapolate")
        self.interp_z = interp1d(np.arange(len(self.sequence_z)), self.sequence_z, kind='linear', fill_value="extrapolate")

    def change_gain(self, gain):
        if len(gain) == 1:
            self.gain = [gain] * self.ndim
        elif len(gain) == self.ndim:
            self.gain = gain
        else:
            print('pseudo_signal.py : Pseudo signal gain setting Error')
  
    def get_value_at_time(self, t):
        # Ensure the requested time is within the bounds of the sequences
        if not (0 <= t < len(self.sequence_x)):
            raise IndexError("Time out of bounds")

        # Fetch the values from each sequence at time t
        # value_x = self.sequence_x[t]
        # value_y = self.sequence_y[t]
        # value_z = self.sequence_z[t]
        value_x = self.gain[0] * self.interp_x(t) 
        value_y = self.gain[1] * self.interp_y(t) 
        value_z = self.gain[2] * self.interp_z(t) 
        return value_x, value_y, value_z



    def plot_time_series(self, length):
        times = np.linspace(0, length, num=1000)  # Use a high-resolution time array for plotting
        values_x = self.gain[0] * self.interp_x(times)
        values_y = self.gain[1] * self.interp_y(times)
        values_z = self.gain[2] * self.interp_z(times)

        plt.figure(figsize=(14, 6))

        # Plot for X dimension
        plt.subplot(3, 1, 1)
        plt.plot(times, values_x, label='Dimension X', color='r')
        plt.title('Time Series for Dimension X')
        plt.ylabel('Value')
        plt.legend()

        # Plot for Y dimension
        plt.subplot(3, 1, 2)
        plt.plot(times, values_y, label='Dimension Y', color='g')
        plt.title('Time Series for Dimension Y')
        plt.ylabel('Value')
        plt.legend()

        # Plot for Z dimension
        plt.subplot(3, 1, 3)
        plt.plot(times, values_z, label='Dimension Z', color='b')
        plt.title('Time Series for Dimension Z')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        plt.tight_layout()
        plt.show()


    def to_json(self):
        # Convert the instance to a dictionary of basic data types
        data = {
            'order': self.order,
            'sequence_x': self.sequence_x.tolist(),  # Convert numpy arrays to lists
            'sequence_y': self.sequence_y.tolist(),
            'sequence_z': self.sequence_z.tolist()
        }
        return json.dumps(data)  # Serialize the dictionary to a JSON string

    @staticmethod
    def from_json(json_str):
        # Deserialize the JSON string back to a dictionary
        data = json.loads(json_str)
        # Create a new instance of the class using the dictionary
        instance = MaxLengthSequence3D(data['order'])
        instance.sequence_x = np.array(data['sequence_x'])
        instance.sequence_y = np.array(data['sequence_y'])
        instance.sequence_z = np.array(data['sequence_z'])
        # Reinitialize the interpolation functions
        instance.interp_x = interp1d(np.arange(len(instance.sequence_x)), instance.sequence_x, kind='linear', fill_value="extrapolate")
        instance.interp_y = interp1d(np.arange(len(instance.sequence_y)), instance.sequence_y, kind='linear', fill_value="extrapolate")
        instance.interp_z = interp1d(np.arange(len(instance.sequence_z)), instance.sequence_z, kind='linear', fill_value="extrapolate")
        return instance


# Example usage
if __name__ == '__main__':
    size = 1023  # Define the size of the sequence
    sequence3D = MaxLengthSequence3D(size)
    
    # Fetch values at a specific time t
    t = 500
    values_at_t = sequence3D.get_value_at_time(t)
    print(f"Values at time {t}: {values_at_t}")
    
    
    # Plot the time series for each dimension up to a specified length
    sequence3D.plot_time_series(100)  # Plot the first 100 time steps for each dimension