"""
This file contains the code that links images.py, instrument.py and analysis.py together.
The functions and classes here pertain to the processing of image data into instrument data via
a simulation of the instrument observing said image. The main function here is process_image(), 
with other functions in the file being subsidiary helper functions. The file also contains the 
definition for the interferometer_data class, which acts as the standardised data structure used
in the functions in analysis.py.
"""

import numpy as np
import scipy.constants as spc
import scipy.special as sps

class interferometer_data():
    """ 
    Class that serves as a container for interferometer output data.
    Constructed as such for ease of use by way of standardization.
    Does not contain manipulation methods, data inside will have to be edited via external methods.
    """

    def __init__(self, instrument, image, samples, pos_noise=0., energy_noise=0., t_noise=0.):
        """ 
        This function is the main function that takes an image and converts it to instrument data as
        if the instrument had just observed the object te image is a representation of. 
        It models the individual photons coming in each timestep, at what detector they end up, 
        whether they are absorbed along the way, how much noise there is, and also whether the 
        spacecraft the instrument is on wobbles, and possible correction for this.

        Parameters:\n

        instrument (interferometer class object) = Instrument object to be used to simulate observing the image.\n
        image (image class object) = Image object to be observed.\n
        N_f (int) = number of fringes we want to consistently see.\n
        samples (int) = number of samples to use for approximating fresnell difraction pattern.\n
        pos_noise (float) = Noise value in micrometers used as sigma in normal distribution around 'true' position. Default 0. means no noise.\n
        energy_noise (float) = Noise value used as percentage of 'true' energy to determine sigma in normal distribution. Default 0. means no noise.\n
        t_noise (float) = Noise value in seconds used as sigma in normal distribution around 'true' time of arrival. Default 0. means no noise.\n
        """
        # Useful shorthand
        self.size = image.size
        self.pos_noise = pos_noise * 1e-6
        self.energy_noise = energy_noise * spc.eV * 1e3
        self.t_noise = t_noise

        self.process_photon_energies(instrument, image)
        self.discretize_E(instrument)
        self.process_photon_toa(image)
        self.discretize_t(instrument)

        self.process_photon_dpos(instrument, image, samples)
        self.discretize_pos(instrument)

    def process_photon_energies(self, instrument, image):
        """
        This function is a helper function for process_image that specifically processes the energies that photons have and
        how the instrument records them.
        """

        self.image_energies = image.energies
        if self.energy_noise > 0.:
            # % is for forcing it to be impossible for photons to be measured above or below energy range, while keeping random distribution
            # If you want to avoid high energies bleeding over in the case of for example an emission line you want to image,
            # simply set the energy range too big to have this contamination. 
            self.energies = ((self.image_energies + np.random.normal(0, self.energy_noise, self.size) - instrument.E_range[0])
                                             % (instrument.E_range[1] - instrument.E_range[0]) 
                                             + instrument.E_range[0])
        else:
            self.energies = self.image_energies

    def process_photon_toa(self, image):
        """
        This function is a helper function for process_image that specifically processes the times at which photons arrive and
        how the instrument records them.
        """

        self.image_toa = image.toa
        if self.t_noise > 0.:
            # % is for forcing it to be impossible for photons to arrive late or early, while keeping random distribution
            self.toa = np.random.normal(self.image_toa, self.t_noise, self.size) % np.max(self.image_toa)
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
        for channel in np.unique(self.actual_discrete_E):
            # Here each photon in the current energy channel is selected, and a wavelength corresponding to the energy is calculated.
            photons_in_channel = self.actual_discrete_E == channel
            wavelength = spc.h * spc.c / ((channel + .5) * (instrument.res_E / 10) + instrument.E_range[0])

            # Again, only populated baselines are selected
            for baseline_i in np.unique(self.baseline_indices[photons_in_channel]):
                # Here each photon in both the baseline and energy channel is selected, and the baseline is called to shorten later calls to it. 
                photons_to_generate = (self.baseline_indices == baseline_i) * photons_in_channel
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
        if self.pos_noise > 0.:
            self.pos = ((self.actual_pos + np.random.normal(0, self.pos_noise, self.size) - instrument.pos_range[0])
                                             % (instrument.pos_range[1] - instrument.pos_range[0]) 
                                             + instrument.pos_range[0])
        else:
            self.pos = self.actual_pos

    def discretize_E(self, ins):
        """
        Method that discretizes energies of incoming photons into energy channels.
        Adds an array of these locations stored to the class under the name self.discrete_E.

        Parameters:
        ins (interferometer-class object): object containing the specifications for discretisation.\n
        """
        self.discrete_E = (self.energies - ins.E_range[0]) // ins.res_E

        # Also discretizing the actual energy for use in determining the photon position, but with higher resolution than 
        # we can measure the actual energy (factor 10 is arbitrary). The discretization is a necesary evil to avoid having
        # to generate pmfs for each individual photon, and we need to discretize the actual photon energies since the 
        # potential readout noise on energy measurements wouldn't impact the locations photons actually arrive at.
        self.actual_discrete_E = (self.image_energies - ins.E_range[0]) // (ins.res_E / 10)

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
        self.discrete_pos = (self.pos - ins.pos_range[0]) // ins.res_pos 
        
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