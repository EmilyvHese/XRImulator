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

    def __init__(self, instrument, image, samples, noise=False):
        """ 
        This function is the main function that takes an image and converts it to instrument data as
        if the instrument had just observed the object te image is a representation of. 
        It models the individual photons coming in each timestep, at what detector they end up, 
        whether they are absorbed along the way, how much noise there is, and also whether the 
        spacecraft the instrument is on wobbles, and possible correction for this.

        Parameters:\n

        instrument (interferometer class object): Instrument object to be used to simulate observing the image.
        image (image class object): Image object to be observed.\n
        N_f (int): number of fringes we want to consistently see.\n
        samples (int): number of samples to use for approximating fresnell difraction pattern.\n
        noise (boolean) = Whether or not to include Gaussian noise, with default False for not including it.\n
        """
        # Useful shorthand
        self.size = image.size
        self.noise = noise

        self.process_photon_energies(instrument, image)
        self.discretize_E(instrument)
        self.process_photon_toa(instrument, image)
        self.discretize_t(instrument)

        self.process_photon_dpos(instrument, image, samples)
        self.discretize_pos(instrument)

    def process_photon_energies(self, instrument, image):
        """
        This function is a helper function for process_image that specifically processes the energies that photons have and
        how the instrument records them.
        """

        self.image_energies = image.energies
        if self.noise:
            self.energies = np.random.normal(self.image_energies, self.image_energies/10)
        else:
            self.energies = self.image_energies

    def process_photon_toa(self, instrument, image):
        """
        This function is a helper function for process_image that specifically processes the times at which photons arrive and
        how the instrument records them.
        """

        self.image_toa = image.toa
        if self.noise:
            self.toa = np.random.normal(self.image_toa, instrument.res_t)
            # Forcing it to be impossible for photons to arrive late
            self.toa[self.toa > np.amax(image.toa)] = np.amax(image.toa)
        else:
            self.toa = self.image_toa

    def process_photon_dpos(self, instrument, image, samples):
        """
        This function is a helper function for process_image that specifically processes the locations where photons impact
        on the detector (hence the d(etector)pos(ition) name). Not to be used outside the process_image context.

        Paramater definitions can be found in process_image.
        """

        def fre_dif(wavelength, baseline):
            """
            Helper function that calculates the fresnell difraction pattern for two overlapping
            beams such as is the case in the interferometer. Does so according to a specified number
            of fringes to model out to, and a number of samples to use to interpolate between.
            """
            u_0 = baseline.W * np.sqrt(2 / (wavelength * baseline.L))
            u_1 = lambda u, u_0: u + u_0/2
            u_2 = lambda u, u_0: u - u_0/2

            # Times 3 to probe a large area for the later interpolation
            u = np.linspace(-u_0, u_0, samples)

            S_1, C_1 = sps.fresnel(u_1(u, u_0))
            S_2, C_2 = sps.fresnel(u_2(u, u_0))

            A = (C_2 - C_1 + 1j*(S_2 - S_1)) * (1 + np.exp(np.pi * 1j * u_0 * u))
            A_star = np.conjugate(A)

            I = np.abs(A * A_star)
            I_pmf = I / sum(I)

            return I_pmf, u

        self.actual_pos = np.zeros(self.size)

        # Randomly selecting a baseline for the photon to go in. 
        self.baseline_indices = np.random.randint(0, len(instrument.baselines), self.size)

        # Defining the pointing, relative position and off-axis angle for each photon over time.
        # Relative position is useful for the calculation of theta, since the off-axis angle is very dependent on where the axis is.
        self.pointing = instrument.gen_pointing(np.max(image.toa))
        pos_rel = self.pointing[image.toa, :2] - image.loc
        theta = np.cos(self.pointing[image.toa, 2] - np.arctan2(pos_rel[:, 0], pos_rel[:, 1])) * np.sqrt(pos_rel[:, 0]**2 + pos_rel[:, 1]**2)

        # Here the actual drawing of photons happens, in two for loops since both looped variables impact the diffraction pattern.
        # Only populated energy channels are selected.
        for channel in np.unique(self.discrete_E):
            # Here each photon in the current energy channel is selected, and a wavelength corresponding to the energy is calculated.
            photons_in_channel = self.discrete_E == channel
            wavelength = spc.h * spc.c / ((channel + .5) * instrument.res_E + instrument.E_range[0])

            # Again, only populated baselines are selected
            for baseline_i in np.unique(self.baseline_indices[photons_in_channel]):
                # Here each photon in both the baseline and energy channel is selected, and the baseline is called to shorten later calls to it. 
                photons_to_generate = self.baseline_indices[photons_in_channel] == baseline_i
                baseline = instrument.baselines[baseline_i]

                # The diffraction pattern point mass function and the sampled locations are calculated, and then sampled.
                diffraction_pattern, u_samples = fre_dif(wavelength, baseline)
                u_pos = np.random.choice(u_samples,
                                            photons_to_generate.nonzero()[0].size, 
                                            replace=True, 
                                            p=diffraction_pattern)
                
                # Here the sampled Fresnell u position is converted to a physical detector position.
                self.actual_pos[photons_to_generate] = (u_pos / np.sqrt(2 / (wavelength * baseline.L)) 
                                - baseline.F * theta[photons_to_generate])
        
        # Noises up the data
        self.noisy_pos = self.actual_pos + np.random.normal(0, instrument.res_pos, self.actual_pos.size)

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
        return (self.discrete_E + .5) * ins.res_E + ins.E_range[0]

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
        return (self.discrete_t + .5) * ins.res_t