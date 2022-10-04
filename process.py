"""
This file contains the code that links images.py, instrument.py and analysis.py together.
The functions and classes here pertain to the processing of image data into instrument data via
a simulation of the instrument observing said image. The main function here is process_image(), 
with other functions in the file being subsidiary helper functions. The file also contains the 
definition for the interferometer_data class, which acts as the standardised data structure used
in the functions in analysis.py.
"""

import numpy as np
import instrument as ins
import scipy.constants as spc
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

        self.toa = np.zeros(size)
        self.energies = np.zeros(size)

        # Useful shorthand
        self.size = size

# def process_photons(instrument, image, data, timestep_photons):
#     """
#     This function takes an input image and photon by photon on a discrete timestep passes them through 
#     the specified instrument, giving back the data that this instrument has recorded.

#     Parameters:
#     instrument (interferometer class object): The instrument to pass the image through.
#     image (image class object): Image to pass through instrument.
#     data (interferometer_data class object): Object to save the instrument data in.
#     timestep_photons: array containing indices of all photons that arrive at the to be processed timestep.
#     """

#     data.toa[timestep_photons] = image.toa[timestep_photons]
#     data.energies[timestep_photons] = image.energies[timestep_photons]
    
#     # Calculating photon wavelengths and phases from their energies
#     lambdas = spc.h * spc.c / image.energies[timestep_photons]
#     k = 2 * spc.pi / lambdas

#     for photon in timestep_photons:
#         # Randomly selecting a baseline for the photon to go in. Possible #TODO fix this to be more accurate than just random?
#         baseline = instrument.baselines[np.random.randint(0, len(instrument.baselines))]

#         # Calculating off axis angle of photon
#         theta = np.cos(instrument.roll) * image.loc[photon, 0][0] + np.sin(instrument.roll) * image.loc[photon, 1][0]
#         delta_d = lambda y: 2 * y * np.sin(baseline.theta_b/2) + baseline.D * np.sin(theta)
#         D = lambda y: np.sin(theta) * baseline.D + delta_d(y)

#         # projected intensity as a function of y position
#         I = lambda y: 2 + 2*np.cos(k*D(y))

#         accepted = False
#         # Doing an accept/reject method to find the precise location photons impact at
#         while accepted != True:
#             photon_y = np.random.rand() * baseline.W - baseline.W/2 + baseline.F * theta
#             photon_I = np.random.rand() * 4

#             # Left-over test code
#             # y_test = np.linspace(-baseline.W/2,baseline.W/2, 10000) + baseline.F * theta
#             # plt.plot(y_test, I(y_test))
#             # plt.plot(y_test, 2 + 2*np.cos(D(y_test)))
#             # plt.plot(photon_y, photon_I, 'r.')
#             # plt.show()

#             if photon_I < I(photon_y):
#                 accepted = True

#         #TODO convert precise location to pixel position depending on interferometer specs

#         # The on-axis angle (as opposed to theta, the off-axis angle)
#         psi = np.cos(instrument.roll) * image.loc[photon, 0] + np.sin(instrument.roll) * image.loc[photon, 1]
#         data.pos[timestep_photons, 0] = baseline.F * psi
#         data.pos[photon, 1] = photon_y
             
# def process_image(instrument, image, noise, wobble = False, wobble_I = 0, wobble_c = None):
#     """ 
#     This function is the main function that takes an image and converts it to instrument data as
#     if the instrument had just observed the object te image is a representation of. 
#     It models the individual photons coming in each timestep, at what detector they end up, 
#     whether they are absorbed along the way, how much noise there is, and also whether the 
#     spacecraft the instrument is on wobbles, and possible correction for this.

#     Parameters:

#     instrument (interferometer class object): Instrument object to be used to simulate observing the image.
#     image (image class object): Image object to be observed.
#     noise (float): root mean square noise level to be introduced in simulated observation.
#     wobble (boolean): Whether or not the spacecraft wobbles in this simulation. (default False)
#     wobble_I (float): intensity of wobble effect, used as step size in random walk.
#     wobble_c (string): Name of function (options: #TODO) to use to correct for wobble. (default None, so no correction)
#     """

#     data = interferometer_data(image.size)

#     for t in np.arange(0., image.toa[-1]):
#         #TODO add stuff that processes the actual image
#         timestep_photons = np.where(image.toa == t)

#         if wobble:
#             instrument.wobbler(wobble_I)

#             if wobble_c != None:
#                 #TODO
#                 wobble_c(instrument)

#         # if instrument.D != instrument.target_D:
#         #     instrument.update_D()

#         process_photons(instrument, image, data, timestep_photons)

#     #TODO add noise

#     return data

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

    data.toa = image.toa
    data.energies = image.energies
    
    # Calculating photon wavelengths and phases from their energies
    lambdas = spc.h * spc.c / image.energies
    k = 2 * spc.pi / lambdas

    # Randomly selecting a baseline for the photon to go in. Possible #TODO fix this to be more accurate than just random?
    baseline_indices = np.random.randint(0, len(instrument.baselines), data.size)
    baseline_data = np.array([[instrument.baselines[index].W, 
                                instrument.baselines[index].F, 
                                instrument.baselines[index].theta_b,
                                instrument.baselines[index].D] for index in baseline_indices])

    roll = np.zeros(np.max(image.toa) + 1)
    theta = (np.cos(roll[image.toa[:]]) * image.loc[:, 0] + 
                np.sin(roll[image.toa[:]]) * image.loc[:, 1])
    delta_d = lambda y: 2 * y * np.sin(baseline_data[:, 2]/2) + baseline_data[:, 3] * np.sin(theta)
    D = lambda y: np.sin(theta) * baseline_data[:, 3] + delta_d(y)

    # projected intensity as a function of y position
    I = lambda y: 2 + 2 * np.cos(k * D(y))

    # Doing an accept/reject method to find the precise location photons impact at
    accepted_array = np.full(image.size, False, bool)
    while np.any(accepted_array == False):
        unacc_ind = np.nonzero(accepted_array == False)[0]

        photon_y = np.random.rand(image.size) * baseline_data[:, 0] - baseline_data[:, 0]/2 + baseline_data[:, 1] * theta
        photon_I = np.random.rand(image.size) * 4

        # Left-over test code
        # y_test = np.linspace(-baseline_data[:,0]/2,baseline_data[:,0]/2, 10000) + baseline_data[:,1] * theta
        # plt.plot(y_test, I(y_test))
        # # plt.plot(y_test, 2 + 2*np.cos(D(y_test)))
        # plt.plot(photon_y, photon_I, 'r.')
        # plt.show()

        accepted_ys = photon_I < I(photon_y)
        accepted_array[unacc_ind] = accepted_ys[unacc_ind]

        data.pos[unacc_ind, 1] = photon_y[unacc_ind]

    #TODO convert precise location to pixel position depending on interferometer specs

    # The on-axis angle (as opposed to theta, the off-axis angle)
    psi = np.cos(instrument.roll) * image.loc[:, 0] + np.sin(instrument.roll) * image.loc[:, 1]
    data.pos[:, 0] = baseline_data[:, 1] * psi
             

    return data