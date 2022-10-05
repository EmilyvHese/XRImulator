""" 
This file is a repository of leftover code that has since been replaced. 
It is still kept here as a repository for possible repurposing later.
"""

import numpy as np
import scipy.constants as spc

import images
import instrument
from process import interferometer_data
import analysis

def process_photons(instrument, image, data, timestep_photons):
    """
    This function takes an input image and photon by photon on a discrete timestep passes them through 
    the specified instrument, giving back the data that this instrument has recorded.

    Parameters:
    instrument (interferometer class object): The instrument to pass the image through.
    image (image class object): Image to pass through instrument.
    data (interferometer_data class object): Object to save the instrument data in.
    timestep_photons: array containing indices of all photons that arrive at the to be processed timestep.
    """

    data.toa[timestep_photons] = image.toa[timestep_photons]
    data.energies[timestep_photons] = image.energies[timestep_photons]
    
    # Calculating photon wavelengths and phases from their energies
    lambdas = spc.h * spc.c / image.energies[timestep_photons]
    k = 2 * spc.pi / lambdas

    for photon in timestep_photons:
        # Randomly selecting a baseline for the photon to go in. Possible #TODO fix this to be more accurate than just random?
        baseline = instrument.baselines[np.random.randint(0, len(instrument.baselines))]

        # Calculating off axis angle of photon
        theta = np.cos(instrument.roll) * image.loc[photon, 0][0] + np.sin(instrument.roll) * image.loc[photon, 1][0]
        delta_d = lambda y: 2 * y * np.sin(baseline.theta_b/2) + baseline.D * np.sin(theta)
        D = lambda y: np.sin(theta) * baseline.D + delta_d(y)

        # projected intensity as a function of y position
        I = lambda y: 2 + 2*np.cos(k*D(y))

        accepted = False
        # Doing an accept/reject method to find the precise location photons impact at
        while accepted != True:
            photon_y = np.random.rand() * baseline.W - baseline.W/2 + baseline.F * theta
            photon_I = np.random.rand() * 4

            # Left-over test code
            # y_test = np.linspace(-baseline.W/2,baseline.W/2, 10000) + baseline.F * theta
            # plt.plot(y_test, I(y_test))
            # plt.plot(y_test, 2 + 2*np.cos(D(y_test)))
            # plt.plot(photon_y, photon_I, 'r.')
            # plt.show()

            if photon_I < I(photon_y):
                accepted = True

        #TODO convert precise location to pixel position depending on interferometer specs

        # The on-axis angle (as opposed to theta, the off-axis angle)
        psi = np.cos(instrument.roll) * image.loc[photon, 0] + np.sin(instrument.roll) * image.loc[photon, 1]
        data.pos[timestep_photons, 0] = baseline.F * psi
        data.pos[photon, 1] = photon_y
             
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

    for t in np.arange(0., image.toa[-1]):
        #TODO add stuff that processes the actual image
        timestep_photons = np.where(image.toa == t)

        if wobble:
            instrument.wobbler(wobble_I)

            if wobble_c != None:
                #TODO
                wobble_c(instrument)

        # if instrument.D != instrument.target_D:
        #     instrument.update_D()

        process_photons(instrument, image, data, timestep_photons)

    #TODO add noise

    return data