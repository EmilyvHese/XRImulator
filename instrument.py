"""
This file contains the code for the class that will serve to simulate the inner workings of an X-ray interferometer. 
It will also contain methods used for manipulating these virtual instruments as real instruments would be during observations.
"""

import numpy as np
import scipy.interpolate as spinter

class baseline():
    """
    This class defines a single baseline in an interferometer object, and is used as a helper for the interferometer class objects as a result.
    """

    def __init__(self, D, L, W, F, theta_g, length):
        """ 
        Function that generates a single x-ray interferometer baseline according to given specifications.
        
        Parameters:\n
        D (float) = Baseline of the interferometer (in meters)\n
        D_varspeed (float) = Speed at which the baseline can be varied in the interferomter (meters per time step)\n
        L (float) = Length from last mirror to CCD surface (in meters)\n
        W (float) = incident photon beam width (in micrometers)\n
    	F (float) = effective focal length of interferometer (in meters)\n
        theta_g (float) = grazing incidence angle of mirrors (in degrees)\n
        length (float) = length of mirrors, needed to define collecting area (in meters)\n
        #TODO add more relevant parameters
        """
        # Converting all input parameters into self.parameters in SI units.
        self.D = D
        self.L = L
        self.W = W * 10**-6
        self.F = F
        self.theta_g = theta_g * 2 * np.pi / 360
        self.length = length

        # Calculating some more relevant parameters for ease of access.
        self.theta_b = D / (2 * F)
        self.colarea = W * length * 2
        
class interferometer():
    """ 
    Class defining a hypothetical x-ray interferometer.
    It contains the code needed to generate the interferometer and adapt some of its characteristics afterwards.
    """

    def __init__(self, res_E, res_t, res_pos, E_range, pos_range, 
                    wobble_I = 0., wobble_c = None, 
                    roller = None, roll_speed = 0., roll_stop_t = 0., roll_stop_a = 0.):
        """ 
        Function that generates a virtual x-ray interferometer according to given specifications.
        
        Parameters:
        #TODO update this description

        res_E (float) = Energy resolution of CCD's in instrument (in KeV)\n
        res_t (float) = Time resolution of CCD's in instrument (seconds)\n
        res_pos (float) = Length from last mirror to CCD surface (in meters)\n
        E_range (array-like of floats) = incident photon beam width (in micrometers)\n
    	pos_range (array-like of floats) = effective focal length of interferometer (in meters)\n
        wobble_I (float) = Intensity of wobble effect, used as sigma in for normally distributed random walk steps. Default is 0, which means no wobble. (in arcsec)\n
        wobble_c (function) = function to use to correct for spacecraft wobble in observation (possibly not relevant here)\n
        roller (function) = function to use to simulate the spacecraft rolling. Options are 'smooth_roll' and 'discrete_roll'.\n
        roll_speed (float) = Indicator for how quickly spacecraft rolls around. Default is 0, meaning no roll. (in rad/sec)\n
        roll_stop_t (float) = Indicator for how long spacecraft rests at specific roll if using 'discrete_roll'. Default is 0, meaning it doesn't stop. (in seconds)\n
        roll_stop_a (float) = Indicator for at what angle increments spacecraft rests at if using 'discrete_roll'. Default is 0, meaning it doesn't stop. (in rads)\n
        """

        # # Setting target values for changeable parameters
        # # These are necessary to track their changes over time
        # self.target_D = D

        self.baselines = []

        # Pointing directions, starting at 0 (straight at whatever is to be observed)
        self.pointing = np.zeros(2)

        # Roll direction, to support filling out entire u,v plane
        # Standard is pi/2 since the default axis of measurement in literature is the y-axis
        self.roll = 0

        # Different resolutions, with energy, time and pixel size.
        self.res_E = res_E * 1.602177733e-16
        self.res_t = res_t
        self.res_pos = res_pos * 10**-6

        self.E_range = E_range * 1.602177733e-16
        self.pos_range = pos_range * 10**-6

        self.wobble_I = wobble_I
        self.wobble_c = wobble_c

        self.roller = roller
        self.roll_speed = roll_speed
        self.roll_stop_t = roll_stop_t
        self.roll_stop_a = roll_stop_a

    def update_D(self):
        """ Function that updates the baseline towards the target baseline on a single timestep. """
        if self.D < self.target_D - self.D_varspeed:
            self.D += self.D_varspeed

        elif self.D > self.target_D + self.D_varspeed:
            self.D -= self.D_varspeed

        else:
            self.D = self.target_D

    def set_target_D(self, target_D):
        """ Function that sets a new target value for the baseline, to adjust the device towards. """
        self.target_D = target_D

    def wobbler(self, pointing):
        """ 
        Function that adds 'wobble' to the spacecraft, slightly offsetting its pointing every timestep.

        Parameters:

        Instrument (interferometer class object): instrument to offset.\n
        wobble_I (float): wobble intensity
        """
        pointing[1:, :2] = pointing[:-1, :2] + np.random.normal(0, self.wobble_I, size=(len(pointing[:, 0]) - 1, 2)) * 2 * np.pi / (3600 * 360)
        return pointing

    def smooth_roller(self, pointing):

        pointing[1:, 2] = pointing[:-1, 2] + self.roll_speed * self.res_t
        return pointing

    def discrete_roller(self, pointing):
        time_to_move = self.roll_stop_t // self.res_t
        angle_to_move = self.roll_stop_a
        for i, a in enumerate(pointing[:, 2]):
            t_to_move = i - time_to_move

            if t_to_move > 0.:
                pointing[i, 2] = pointing[i - 1, 2] + self.roll_speed * self.res_t
            else:
                pointing[i, 2] = pointing[i - 1, 2]

            if pointing[i, 2] > angle_to_move:
                angle_to_move += self.roll_stop_a
                time_to_move += self.roll_stop_t // self.res_t
            
        return pointing

    def gen_pointing(self, t_exp):
        """ 
        This function generates a pointing vector for each time step in an observation
        """
        pointing = np.zeros(((t_exp // self.res_t) + 1, 3))
        if self.wobble_I:
            pointing = self.wobbler(pointing)

        if self.roll_speed:
            pointing = self.roller(self, pointing)

        if self.wobble_c:
            pass

        return pointing

    def add_baseline(self, D, L, W, F, theta_g, length):
        """
        Function that adds a baseline of given parameters to the interferometer object. Call this function multiple times to
        construct a full interferometer capable of actually observing images. Without these, no photons can be measured.
        
        Parameters:
        D (float) = Baseline of the interferometer (in meters)\n
        D_varspeed (float) = Speed at which the baseline can be varied in the interferomter (meters per time step)\n
        L (float) = Length from last mirror to CCD surface (in meters)\n
        W (float) = incident photon beam width (in micrometers)\n
    	F (float) = effective focal length of interferometer (in meters)\n
        theta_g (float) = grazing incidence angle of mirrors (in degrees)\n
        length (float) = length of mirrors, needed to define collecting area (in meters)\n
        #TODO add more relevant parameters
        """
        self.baselines.append(baseline(D, L, W, F, theta_g, length))
