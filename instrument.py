"""
This file contains the code for the class that will serve to simulate the inner workings of an X-ray interferometer. 
It will also contain methods used for manipulating these virtual instruments as real instruments would be during observations.
"""

import numpy as np


class interferometer():
    """ 
    Class defining a hypothetical x-ray interferometer.
    It contains the code needed to generate the interferometer and adapt some of its characteristics afterwards.
    """

    def __init__(self, D, D_varspeed, L, W, F, theta_g):
        """ 
        Function that generates a virtual x-ray interferometer according to given specifications.
        
        Parameters:
        D (float) = Baseline of the interferometer (in meters)
        D_varspeed (float) = Speed at which the baseline can be varied in the interferomter (meters per time step)
        L (float) = Length from last mirror to CCD surface (in meters)
        W (float) = incident photon beam width (in micrometers)
    	F (float) = effective focal length of interferometer (in meters)
        theta_g (float) = grazing incidence angle of mirrors (in degrees)
        #TODO add more relevant parameters
        """

        # Converting all input parameters into self.parameters in SI units.
        self.D = D
        self.D_varspeed = D_varspeed
        self.L = L
        self.W = W * 10**-6
        self.F = F
        self.theta_g = theta_g * 2 * np.pi / 360

        # Calculating some more relevant parameters for ease of access.
        self.theta_b = D / (2 * F)

        # Setting target values for changeable parameters
        # These are necessary to track their changes over time
        self.target_D = D
        # Pointing directions, starting at 0 (straight at whatever is to be observed)
        self.theta = 0.
        self.phi = 0.

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

    def wobbler(self, wobble_I):
        """ 
        Function that adds 'wobble' to the spacecraft, slightly offsetting its pointing every timestep.

        Parameters:

        Instrument (interferometer class object): instrument to offset.
        wobble_I (float): wobble intensity
        """

        #TODO think of good way to model wobble

        self.theta += np.random.randn() * wobble_I
        self.phi += np.random.randn() * wobble_I