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

def fre_dif(N_f, samples):
    """
    Helper function that calculates the fresnell difraction pattern for two overlapping
    beams such as is the case in the interferometer. Does so according to a specified number
    of fringes to model out to, and a number of samples to use to interpolate between.
    """
    u_0 = np.sqrt(2 * N_f)
    u_1 = lambda u, u_0: u + u_0/2
    u_2 = lambda u, u_0: u - u_0/2

    # Times 5 to probe a large area for the later interpolation
    u = np.linspace(-u_0, u_0, samples) * 5 

    S_1, C_1 = sps.fresnel(u_1(u, u_0))
    S_2, C_2 = sps.fresnel(u_2(u, u_0))

    A = (C_2 - C_1 + 1j*(S_2 - S_1)) * (1 + np.exp(np.pi * 1j * u_0 * u))
    A_star = np.conjugate(A)

    I = A * A_star
    I_pdf = I / sum(I)

    return spinter.interp1d(u, I_pdf, bounds_error=False, fill_value=0)

def detected_intensity(theta_b, theta, k, D, y):
    """
    Helper function for process_photon_dpos, that calculates the projected intensity at the detector.
    Written in this way for legibility, by keeping both functions free of clutter.
    """
    # Functions necessary to calculate I later (from willingale's paper)
    delta_d = 2 * y * np.sin(theta_b/2) + np.sin(theta) * D

    # Projected intensity as a function of y position
    return 2 + 2 * np.cos(k * delta_d)

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

    def process_photon_energies(self, instrument, image):
        """
        This function is a helper function for process_image that specifically processes the energies that photons have and
        how the instrument records them.
        """

        self.image_energies = image.energies
        self.energies = np.random.normal(self.image_energies, instrument.res_E)

    def process_photon_toa(self, instrument, image):
        """
        This function is a helper function for process_image that specifically processes the times at which photons arrive and
        how the instrument records them.
        """

        self.image_toa = image.toa
        self.toa = np.random.normal(self.image_toa, instrument.res_t)
        # Forcing it to be impossible for photons to arrive late
        self.toa[self.toa > np.amax(image.toa)] = np.amax(image.toa)

    def process_photon_dpos(self, instrument, image, N_f, samples):
        """
        This function is a helper function for process_image that specifically processes the locations where photons impact
        on the detector (hence the d(etector)pos(ition) name). Not to be used outside the process_image context.

        Paramater definitions can be found in process_image.
        """

        self.actual_pos = np.zeros((self.size, 2))
        # Calculating photon wavelengths and phases from their energies
        lambdas = spc.h * spc.c / image.energies
        k = 2 * spc.pi / lambdas

        # Randomly selecting a baseline for the photon to go in. Possible #TODO fix this to be more accurate than just random?
        self.baseline_indices = np.random.randint(0, len(instrument.baselines), self.size)

        # Getting data from those baselines that is necessary for further calculations.
        # Must be done this way since baselines[unacc_ind].W doesn't go element-wise, and .W is not an array operation
        baseline_data = np.array([[instrument.baselines[index].W, 
                                    instrument.baselines[index].F, 
                                    instrument.baselines[index].theta_b,
                                    instrument.baselines[index].D,
                                    instrument.baselines[index].L] for index in self.baseline_indices])

        # Calculating the fresnell diffraction pattern for set number of fringes and samples
        self.inter_pdf = fre_dif(N_f, samples)

        # Defining the pointing, relative position and off-axis angle for each photon over time.
        # Relative position is useful for the calculation of theta, since the off-axis angle is very depedent on where the axis is.
        # There is a 1e-20 factor in a denominator, to prevent divide by zero errors. Typical values for pos_rel are all much larger, 
        # so this does not simply move the problem.
        self.pointing = instrument.gen_pointing(np.max(image.toa))
        pos_rel = image.loc - self.pointing[image.toa, :2]
        theta = np.cos(self.pointing[image.toa, 2] - np.arctan(pos_rel[:, 0] / (pos_rel[:, 1] + 1e-20))) * np.sqrt(pos_rel[:, 0]**2 + pos_rel[:,1]**2)

        # Quick visualization code
        # plt.plot(image.toa[:], theta)
        # plt.plot(image.toa[:], self.pointing[image.toa[:], 2])
        # plt.plot(image.toa[-1], np.cos(self.pointing[-1, 2]), 'o')
        # plt.show()

        # Doing an accept/reject method to find the precise location photons impact at.
        # It uses a formula from #TODO add reference to that one presentation Haniff primer something
        # This array records which photons are accepted so far, starting as all False and becoming True when one is accepted
        accepted_array = np.full(image.size, False, bool)
        while np.any(accepted_array == False):
            # Indices of all unaccepted photons, which are the only ones that need to be generated again each loop.
            unacc_ind = np.nonzero(accepted_array == False)[0]

            # Generating new photons for all the unaccepted indices with accurate y-locations and random intensities.
            photon_y = (np.random.rand(len(unacc_ind)) * baseline_data[unacc_ind, 0] 
                        - baseline_data[unacc_ind, 0]/2)

            # Converting y positions to u positions for scaling the fresnell diffraction to            
            photon_u = photon_y * np.sqrt(2 / (lambdas[unacc_ind] * baseline_data[unacc_ind, 4]))
            photon_fresnell = self.inter_pdf(photon_u)

            photon_I = np.random.rand(len(unacc_ind)) * 4 * np.amax(photon_fresnell)

            # Calculating the projected intensity with the detected_intensity function
            I = detected_intensity(baseline_data[unacc_ind, 2], 
                                    theta[unacc_ind], k[unacc_ind], 
                                    baseline_data[unacc_ind, 3], photon_y)

            # Checking which photons will be accepted, and updating the accepted_array accordingly
            accepted_array[unacc_ind] = photon_I < (I * photon_fresnell)
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
        return (self.discrete_pos + 1) * ins.res_pos + ins.pos_range[0] + ins.res_pos / 2

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