"""
This file contains the code that links images.py, instrument.py and analysis.py together.
The functions and classes here pertain to the processing of image data into instrument data via
a simulation of the instrument observing said image. The main function here is process_image(), 
with other functions in the file being subsidiary helper functions. The file also contains the 
definition for the interferometer_data class, which acts as the standardised data structure used
in the functions in analysis.py.
"""

from turtle import pos
import numpy as np
import instrument as ins
import scipy.constants as spc
import scipy.special as sps
import scipy.interpolate as spinter
import matplotlib
import matplotlib.pyplot as plt
import time

class interferometer_data():
    """ 
    Class that serves as a container for interferometer output data.
    Constructed as such for ease of use by way of standardization.
    Does not contain manipulation methods, data inside will have to be edited via external methods.
    """

    def __init__(self, instrument, image, N_f, samples):
        """ 
        This function is the main function that takes an image and converts it to instrument data as
        if the instrument had just observed the object te image is a representation of. 
        It models the individual photons coming in each timestep, at what detector they end up, 
        whether they are absorbed along the way, how much noise there is, and also whether the 
        spacecraft the instrument is on wobbles, and possible correction for this.

        Parameters:

        instrument (interferometer class object): Instrument object to be used to simulate observing the image.
        image (image class object): Image object to be observed.
        N_f (int): number of fringes we want to consistently see.
        samples (int): number of samples to use for approximating fresnell difraction pattern.
        """
        # Useful shorthand
        self.size = image.size

        self.process_photon_energies(instrument, image)
        self.discretize_E(instrument)
        self.process_photon_toa(instrument, image)
        self.discretize_t(instrument)

        #TODO look into using pdf
        self.process_photon_dpos(instrument, image, N_f, samples)
        self.discretize_pos(instrument)

        self.test_sin()

    def process_photon_energies(self, instrument, image):
        """
        This function is a helper function for process_image that specifically processes the energies that photons have and
        how the instrument records them.
        """

        self.image_energies = image.energies
        # self.energies = np.random.normal(self.image_energies, instrument.res_E)
        self.energies = self.image_energies

    def process_photon_toa(self, instrument, image):
        """
        This function is a helper function for process_image that specifically processes the times at which photons arrive and
        how the instrument records them.
        """

        self.image_toa = image.toa
        # self.toa = np.random.normal(self.image_toa, instrument.res_t)
        # Forcing it to be impossible for photons to arrive late
        # self.toa[self.toa > np.amax(image.toa)] = np.amax(image.toa)
        self.toa = self.image_toa

    def process_photon_dpos(self, instrument, image, N_f, samples):
        """
        This function is a helper function for process_image that specifically processes the locations where photons impact
        on the detector (hence the d(etector)pos(ition) name). Not to be used outside the process_image context.

        Paramater definitions can be found in process_image.
        """

        def fre_dif(N_f, samples):
            """
            Helper function that calculates the fresnell difraction pattern for two overlapping
            beams such as is the case in the interferometer. Does so according to a specified number
            of fringes to model out to, and a number of samples to use to interpolate between.
            """
            u_0 = np.sqrt(2 * N_f)
            u_1 = lambda u, u_0: u + u_0/2
            u_2 = lambda u, u_0: u - u_0/2

            # Times 3 to probe a large area for the later interpolation
            u = np.linspace(-u_0, u_0, samples) * 3

            S_1, C_1 = sps.fresnel(u_1(u, u_0))
            S_2, C_2 = sps.fresnel(u_2(u, u_0))

            A = (C_2 - C_1 + 1j*(S_2 - S_1)) * (1 + np.exp(np.pi * 1j * u_0 * u))
            A_star = np.conjugate(A)

            I = A * A_star
            I_pdf = I / sum(I)

            return spinter.interp1d(u, I_pdf, bounds_error=False, fill_value=0)

        self.actual_pos = np.zeros((self.size, 2))
        # Calculating photon wavelengths and phases from their energies
        lambdas = spc.h * spc.c / image.energies

        # Randomly selecting a baseline for the photon to go in. 
        self.baseline_indices = np.random.randint(0, len(instrument.baselines), self.size)

        # Getting data from those baselines that is necessary for further calculations.
        # Must be done this way since baselines[unacc_ind].W doesn't go element-wise, and .W is not an array operation
        baseline_data = np.array([[instrument.baselines[index].W, 
                                    instrument.baselines[index].F,
                                    instrument.baselines[index].L] for index in self.baseline_indices])

        # Calculating the fresnell diffraction pattern for set number of fringes and samples
        self.inter_pdf = fre_dif(N_f, samples)

        # Defining the pointing, relative position and off-axis angle for each photon over time.
        # Relative position is useful for the calculation of theta, since the off-axis angle is very dependent on where the axis is.
        # There is a 1e-20 factor in a denominator, to prevent divide by zero errors. Typical values for pos_rel are all much larger, 
        # so this does not simply move the problem.
        self.pointing = instrument.gen_pointing(np.max(image.toa))
        pos_rel = self.pointing[image.toa, :2] - image.loc
        theta = np.cos(self.pointing[image.toa, 2] - np.arctan2(pos_rel[:, 0], pos_rel[:, 1])) * np.sqrt(pos_rel[:, 0]**2 + pos_rel[:, 1]**2)

        # Doing an accept/reject method to find the precise location photons impact at.
        # This array records which photons are accepted so far, starting as all False and becoming True when one is accepted
        accepted_array = np.full(image.size, False, bool)
        while np.any(accepted_array == False):
            # Indices of all unaccepted photons, which are the only ones that need to be generated again each loop.
            unacc_ind = np.nonzero(accepted_array == False)[0]

            # Generating new photons for all the unaccepted indices with accurate y-locations and random intensities.
            photon_y = (np.random.rand(unacc_ind.size) * baseline_data[unacc_ind, 0] - baseline_data[unacc_ind, 0]/2)

            # Converting y positions to u positions for scaling the fresnell diffraction to            
            photon_u = (photon_y + baseline_data[unacc_ind, 1] * theta[unacc_ind]) * np.sqrt(2 / (lambdas[unacc_ind] * baseline_data[unacc_ind, 2]))
            photon_fresnell = self.inter_pdf(photon_u)
            
            photon_I = np.random.rand(unacc_ind.size) * np.amax(photon_fresnell) 

            # Checking which photons will be accepted, and updating the accepted_array accordingly
            accepted_array[unacc_ind] = photon_I <= photon_fresnell
            self.actual_pos[unacc_ind, 1] = photon_y

    def discretize_E(self, ins):
        """
        Method that discretizes energies of incoming photons into energy channels.
        Adds an array of these locations stored to the class under the name self.discrete_E.

        Parameters:
        ins (interferometer-class object): object containing the specifications for discretisation.\n
        """
        self.discrete_E = (self.energies - ins.E_range[0]) // ins.res_E

    def channel_to_E(self, ins):
        """ Method that turns discretized energies into the energies at the center of their respective channels. """
        return (self.discrete_E + 1) * ins.res_E + ins.E_range[0] + ins.res_E / 2

    def discretize_pos(self, ins):
        """
        Method that discretizes positions of incoming photons into pixel positions.
        Adds an array of these locations stored to the class under the name self.discrete_pos.

        Parameters:
        ins (interferometer-class object): object containing the specifications for discretisation.\n
        """
        self.discrete_pos = (self.actual_pos - ins.pos_range[0]) // ins.res_pos 
        
    def pixel_to_pos(self, ins):
        """ Method that turns discretized positions into the positions at the center of their respective pixels. """
        return (self.discrete_pos + .5) * ins.res_pos + ins.pos_range[0]

    def discretize_t(self, ins):
        """
        Method that discretizes times of arrival of incoming photons into time steps since start of observation.
        Adds an array of these times stored to the class under the name self.discrete_t.

        Parameters:
        ins (interferometer-class object): object containing the specifications for discretisation.\n
        """
        self.discrete_t = ((self.toa - self.toa[0]) // ins.res_t).astype(int)

    def tstep_to_t(self, ins):
        """ Method that turns discretized time steps into the times at the center of their respective steps. """
        return (self.discrete_t + 1) * ins.res_t + self.toa[0] + ins.res_t / 2

    def test_sin(self):
        self.test_data = np.zeros(self.size)
        for i in range(self.size):
            self.test_data[i] = np.sin(i * 1500)