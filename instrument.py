"""
This file contains the code for the class that will serve to simulate the inner workings of an X-ray interferometer. 
It will also contain methods used for manipulating these virtual instruments as real instruments would be during observations.
"""

import numpy as np
import scipy.constants as spc


class baseline():
    """
    This class defines a single baseline in an interferometer object, and is used as a helper for the interferometer class objects.
    #TODO add more relevant parameters to make this more realistic. In order to fully accurately model an observation,
    this class can be expanded to for example include details on the mirrors, with their effective areas and angles at which they are
    set determining how many and what photons are detected by which baseline. This would necesarily also include another conceptual shift with
    consequences through the rest of the code, as at the moment the image class represents a collection of all photons that will be detected, 
    which would need to shift to being a collection of photons that could be detected, with the number of input photons likely being much greater
    than the detected photons. 
    """

    def __init__(self, D, L, W, F):
        """ 
        Function that generates a single x-ray interferometer baseline according to given specifications.
        
        
        Parameters:\n
        D (float) = Baseline of the interferometer (in meters)\n
        L (float) = Length from last mirror to CCD surface (in meters)\n
        W (float) = incident photon beam width (in micrometers)\n
    	F (float) = effective focal length of interferometer (in meters)\n
        """
        # Converting all input parameters into self.parameters in SI units.
        self.D = D
        self.L = L
        self.W = W * 1e-6
        self.F = F
        
class interferometer():
    """ 
    Class defining a hypothetical x-ray interferometer.
    It contains the code needed to generate the interferometer and adapt some of its characteristics afterwards.
    """

    def __init__(self, res_E, res_t, res_pos, E_range, pos_range, 
                    wobbler = None, wobble_I = 0., wobble_c = None, wobble_file = '', 
                    roller = None, roll_speed = 0., roll_stop_t = 0., roll_stop_a = 0., roll_init = 0.):
        """ 
        Function that generates a virtual x-ray interferometer according to given specifications.
        
        Parameters:
        #TODO update this description

        res_E (float) = Energy resolution of CCD's in instrument (in KeV)\n
        res_t (float) = Time resolution of CCD's in instrument (seconds)\n
        res_pos (float) = Position resolution of CCD's in instrument (in micrometers)\n
        E_range (array-like of floats) = Range of energies that can be recorded (in KeV)\n
    	pos_range (array-like of floats) = Range of positions that can be recorded (in micrometers)\n
        wobble_I (float) = Intensity of wobble effect, used as sigma in normally distributed random walk steps. Default is 0, which means no wobble. (in arcsec)\n
        wobble_c (function) = Function to use to correct for spacecraft wobble in observation (possibly not relevant here)\n
        roller (function) = Function to use to simulate the spacecraft rolling. Options are 'smooth_roll' and 'discrete_roll'.\n
        roll_speed (float) = Indicator for how quickly spacecraft rolls around. Default is 0, meaning no roll. (in rad/sec)\n
        roll_stop_t (float) = Indicator for how long spacecraft rests at specific roll if using 'discrete_roll'. Default is 0, meaning it doesn't stop. (in seconds)\n
        roll_stop_a (float) = Indicator for at what angle increments spacecraft rests at if using 'discrete_roll'. Default is 0, meaning it doesn't stop. (in rads)\n
        """

        self.baselines = []

        # Different resolutions, with energy, time and pixel size.
        self.res_E = res_E * 1e3 * spc.eV
        self.res_t = res_t
        self.res_pos = res_pos * 1e-6

        self.E_range = E_range * 1e3 * spc.eV
        self.pos_range = pos_range * 1e-6

        self.wobbler = wobbler
        self.wobble_I = wobble_I
        self.wobble_c = wobble_c
        self.wobble_file = wobble_file

        self.roller = roller
        self.roll_speed = roll_speed
        self.roll_stop_t = roll_stop_t
        self.roll_stop_a = roll_stop_a
        self.roll_init = roll_init

    def random_wobble(self, pointing):
        """ 
        Function that adds 'wobble' to the spacecraft, slightly offsetting its pointing every timestep.
        It models wobble as a random walk with a given intensity that is used as the sigma for a normally distributed
        step size in both the pitch and yaw directions.

        Parameters:

        Instrument (interferometer class object): instrument to offset.\n

        Returns:

        pointing (array): That same, but now with wobble data.\n
        """
        pointing[1:, :2] = pointing[:-1, :2] + np.random.normal(0, self.wobble_I, size=(len(pointing[:, 0]) - 1, 2)) * 2 * np.pi / (3600 * 360)
        return pointing
    
    def file_wobble(self, pointing):
        """ 
        Function that adds 'wobble' to the spacecraft, slightly offsetting its pointing every timestep.
        This function uses an input file in a csv format (with ',' as delimiter) to read out pointing data, 
        probably generated with a different simulator.
        #TODO This function is mostly a placeholder, to be replaced later to adapt to the actual format this data
        will take. This is only one way it could look, but it should how to structure an eventual replacement for 
        whoever wants to adapt the code.

        Parameters:

        Instrument (interferometer class object): instrument to offset.\n

        Returns:

        pointing (array): That same, but now with wobble data.\n
        """
        pointing[:, :2] = np.genfromtxt(self.wobble_file, np.float64, delimiter=',')
        return pointing

    def smooth_roller(self, pointing):
        """
        Function that generates the roll portion of the pointing data for the instrument. 
        This function is used for a continuous model of rolling the instrument, with a predefined roll
        velocity.

        Parameters:

        pointing (array): 3d array of pointing angles as deviations from observation start for every observational timestep.

        Returns:

        pointing (array): That same, but now with roll data.
        """
        pointing[:, 2] = (np.arange(pointing[:, 2].size) * self.roll_speed * self.res_t) + self.roll_init
        return pointing

    def discrete_roller(self, pointing):
        """
        Function that generates the roll portion of the pointing data for the instrument. 
        This function is used for a discrete model of rolling the instrument, with starts and stops
        at specified roll angle intervals.

        Parameters:

        pointing (array): 3d array of pointing angles as deviations from observation start for every observational timestep.

        Returns:

        pointing (array): That same, but now with roll data.
        """
        # Calculates the stopping interval in timestep units 
        time_to_move = self.roll_stop_t // self.res_t 
        # The angle over which to move after the stopping interval
        angle_to_move = self.roll_stop_a

        for i in pointing[:, 2]:
            t_to_move = i - time_to_move

            if t_to_move > 0.:
                pointing[i, 2] = pointing[i - 1, 2] + self.roll_speed * self.res_t
            else:
                pointing[i, 2] = pointing[i - 1, 2]

            # Defining the next timestep to move at and angle to move to.
            if pointing[i, 2] > angle_to_move:
                angle_to_move += self.roll_stop_a
                time_to_move += self.roll_stop_t // self.res_t
            
        return pointing

    def gen_pointing(self, t_exp):
        """ 
        This function generates a 3d pointing vector for each time step in an observation. It consists of 
        three angles, the pitch, yaw and roll. The first two are linked and generated together by the wobbler 
        function, while the roll is fundamentally different and thus generated differently. If no wobbler or 
        roller are given, the corresponding pointing values will be zero, indicating stillness.
        """
        pointing = np.zeros((t_exp + 2, 3))

        # These try except statements are there for the case that no roller or wobbler are given.
        try:
            pointing = self.roller(self, pointing)
        except Exception:
            pass

        try:
            pointing = self.wobbler(pointing)
        except Exception:
            pass

        if self.wobble_c:
            pass

        return pointing

    def add_baseline(self, D, L, W, F):
        """
        Function that adds a baseline of given parameters to the interferometer object. Call this function multiple times to
        construct a full interferometer capable of actually observing images. Without these, no photons can be measured.
        
        Parameters:
        D (float) = Baseline of the interferometer (in meters)\n
        L (float) = Length from last mirror to CCD surface (in meters)\n
        W (float) = incident photon beam width (in micrometers)\n
    	F (float) = effective focal length of interferometer (in meters)\n
        """
        self.baselines.append(baseline(D, L, W, F))

    def add_willingale_baseline(self, D):
        self.baselines.append(baseline(D, 10, 300, D/(2*np.tan(6 * np.pi / (3600 * 360)))))

    def clear_baselines(self):
        self.baselines.clear()
