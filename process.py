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
import scipy.interpolate as spint
import matplotlib
import matplotlib.pyplot as plt

class interferometer_data():
    """ 
    Class that serves as a container for interferometer output data.
    Constructed as such for ease of use by way of standardization.
    Does not contain manipulation methods, data inside will have to be edited via external methods.
    """

    def __init__(self, size):
        # x and y coordinates of every photon
        self.pos = np.zeros((size, 2))
        # Pixel coordinates of every photon
        self.pix = np.zeros((size, 2))

        self.toa = np.zeros(size)
        self.energies = np.zeros(size)

        # Useful shorthand
        self.size = size

def process_photon_energies(insturment, image, data):
    """
    This function is a helper function for process_image that specifically processes the energies that photons have and
    how the instrument records them.
    """

    #TODO add the energy channels and noise to energy recording
    data.energies = image.energies

def process_photon_toa(insturment, image, data):
    """
    This function is a helper function for process_image that specifically processes the times at which photons arrive
    how the instrument records them.
    """

    #TODO add noise to toa recording
    data.toa = image.toa

def process_photon_dpos(instrument, image, data):
    """
    This function is a helper function for process_image that specifically processes the locations where photons impact
    on the detector (hence the d(etector)pos(ition) name). Not to be used outside the process_image context.

    Paramater definitions can be found in process_image.
    """
    # Calculating photon wavelengths and phases from their energies
    lambdas = spc.h * spc.c / image.energies
    k = 2 * spc.pi / lambdas

    # Randomly selecting a baseline for the photon to go in. Possible #TODO fix this to be more accurate than just random?
    baseline_indices = np.random.randint(0, len(instrument.baselines), data.size)
    baseline_data = np.array([[instrument.baselines[index].W, 
                                instrument.baselines[index].F, 
                                instrument.baselines[index].theta_b,
                                instrument.baselines[index].D] for index in baseline_indices])

    # Defining the roll and off-axis angle for each photon over time #TODO add time dependent roll and find better name
    roll = np.zeros(np.max(image.toa) + 1)
    theta = (np.cos(roll[image.toa[:]]) * image.loc[:, 0] + 
                np.sin(roll[image.toa[:]]) * image.loc[:, 1])

    # Doing an accept/reject method to find the precise location photons impact at.
    # It uses a formula from #TODO add reference to that one presentation
    # This array records which photons are accepted so far, starting as all False and becoming True when one is accepted
    accepted_array = np.full(image.size, False, bool)
    while np.any(accepted_array == False):
        # Indices of all unaccepted photons, which are the only ones that need to be generated again each loop.
        unacc_ind = np.nonzero(accepted_array == False)[0]

        # Generating new photons for all the unaccepted indices with accurate y-locations and random intensities.
        photon_y = (np.random.rand(len(unacc_ind)) * baseline_data[unacc_ind, 0] 
                    - baseline_data[unacc_ind, 0]/2 
                    + baseline_data[unacc_ind, 1] * theta[unacc_ind])
        photon_I = np.random.rand(len(unacc_ind)) * 4

        # Functions necessary to calculate I later
        delta_d = (lambda y: 2 * y * np.sin(baseline_data[unacc_ind, 2]/2) 
                    + baseline_data[unacc_ind, 3] * np.sin(theta[unacc_ind]))
        D = lambda y: np.sin(theta[unacc_ind]) * baseline_data[unacc_ind, 3] + delta_d(y)

        # Projected intensity as a function of y position
        I = lambda y: 2 + 2 * np.cos(k[unacc_ind] * D(y))

        # Checking which photons will be accepted, and updating the accepted_array accordingly
        accepted_array[unacc_ind] = photon_I < I(photon_y)

        data.pos[unacc_ind, 1] = photon_y

    #TODO convert precise location to pixel position depending on interferometer specs

    # The on-axis angle (as opposed to theta, the off-axis angle)
    # psi = np.cos(instrument.roll) * image.loc[:, 0] + np.sin(instrument.roll) * image.loc[:, 1]
    # data.pos[:, 0] = baseline_data[:, 1] * psi

def fre_dif(N_f, samples):
    u_0 = np.sqrt(2 * N_f)
    u_1 = lambda u, u_0: u + u_0/2
    u_2 = lambda u, u_0: u - u_0/2
    u = np.linspace(-5, 5, samples)

    S_1, C_1 = sps.fresnel(u_1(u, u_0))
    S_2, C_2 = sps.fresnel(u_2(u, u_0))

    A = (C_2 - C_1 + 1j*(S_2 - S_1)) * (1 + np.exp(np.pi * 1j * u_0 * u))
    A_star = np.conjugate(A)

    I = A * A_star
    I_pdf = I / sum(I)

    return u, I_pdf

def detected_intensity(theta_b, theta, k, D, y):
    # Functions necessary to calculate I later
    delta_d = 2 * y * np.sin(theta_b/2) + np.sin(theta) * D

    # Projected intensity as a function of y position
    return 2 + 2 * np.cos(k * delta_d)

def process_photon_dpos2(instrument, image, data):
    """
    This function is a helper function for process_image that specifically processes the locations where photons impact
    on the detector (hence the d(etector)pos(ition) name). Not to be used outside the process_image context.

    Paramater definitions can be found in process_image.
    """
    # Calculating photon wavelengths and phases from their energies
    lambdas = spc.h * spc.c / image.energies
    k = 2 * spc.pi / lambdas

    # Randomly selecting a baseline for the photon to go in. Possible #TODO fix this to be more accurate than just random?
    baseline_indices = np.random.randint(0, len(instrument.baselines), data.size)
    baseline_data = np.array([[instrument.baselines[index].W, 
                                instrument.baselines[index].F, 
                                instrument.baselines[index].theta_b,
                                instrument.baselines[index].D,
                                instrument.baselines[index].L] for index in baseline_indices])

    u, fre_pdf = fre_dif(10, int(1e4))
    inter_pdf = spint.interp1d(u, fre_pdf)

    # Defining the roll and off-axis angle for each photon over time #TODO add time dependent roll and find better name
    roll = np.zeros(np.max(image.toa) + 1)
    theta = (np.cos(roll[image.toa[:]]) * image.loc[:, 0] + 
                np.sin(roll[image.toa[:]]) * image.loc[:, 1])

    # Doing an accept/reject method to find the precise location photons impact at.
    # It uses a formula from #TODO add reference to that one presentation
    # This array records which photons are accepted so far, starting as all False and becoming True when one is accepted
    accepted_array = np.full(image.size, False, bool)
    while np.any(accepted_array == False):
        # Indices of all unaccepted photons, which are the only ones that need to be generated again each loop.
        unacc_ind = np.nonzero(accepted_array == False)[0]

        # Generating new photons for all the unaccepted indices with accurate y-locations and random intensities.
        photon_y = (np.random.rand(len(unacc_ind)) * baseline_data[unacc_ind, 0] 
                    - baseline_data[unacc_ind, 0]/2 
                    + baseline_data[unacc_ind, 1] * theta[unacc_ind])

        # Converting y positions to u positions for scaling the fresnell difraction to            
        photon_u = photon_y * np.sqrt(2 / (lambdas[unacc_ind] * baseline_data[unacc_ind, 4]))
        photon_fresnell = inter_pdf(photon_u)

        photon_I = np.random.rand(len(unacc_ind)) * 4 * np.amax(photon_fresnell)

        # Projected intensity as a function of y position
        I = detected_intensity(baseline_data[unacc_ind, 2], 
                                theta[unacc_ind], k[unacc_ind], 
                                baseline_data[unacc_ind, 3], photon_y)

        # Checking which photons will be accepted, and updating the accepted_array accordingly
        accepted_array[unacc_ind] = photon_I < (I * photon_fresnell)
        data.pos[unacc_ind, 1] = photon_y

    #TODO convert precise location to pixel position depending on interferometer specs

    # The on-axis angle (as opposed to theta, the off-axis angle)
    # psi = np.cos(instrument.roll) * image.loc[:, 0] + np.sin(instrument.roll) * image.loc[:, 1]
    # data.pos[:, 0] = baseline_data[:, 1] * psi

def process_image(instrument, image, noise, wobble = False, wobble_I = 0, wobble_c = None):
    """ 
    This function is the main function that takes an image and converts it to instrument data as
    if the instrument had just observed the object te image is a representation of. 
    It models the individual photons coming in each timestep, at what detector they end up, 
    whether they are absorbed along the way, how much noise there is, and also whether the 
    spacecraft the instrument is on wobbles, and possible correction for this.

    Parameters:

    instrument (interferometer class object): Instrument object to be used to simulate observing the image.
    image (image class object): Image object to be observed.
    noise (float): root mean square noise level to be introduced in simulated observation.
    wobble (boolean): Whether or not the spacecraft wobbles in this simulation. (default False)
    wobble_I (float): intensity of wobble effect, used as step size in random walk.
    wobble_c (string): Name of function (options: #TODO) to use to correct for wobble. (default None, so no correction)
    """

    data = interferometer_data(image.size)

    process_photon_energies(instrument, image, data)
    process_photon_toa(instrument, image, data)

    #TODO look into using pdf
    process_photon_dpos2(instrument, image, data)
             
    return data