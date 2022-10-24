""" 
This file is a repository of leftover code that has since been replaced. 
It is still kept here as a repository for possible repurposing later.
"""

import numpy as np
import scipy.constants as spc
import scipy.interpolate as spinter

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

def discretize_pos(self, ins, data, pixel_pos = True):
    """
    Function that discretizes positions of incoming photons into pixel positions.
    Currently only takes into account x_position as that is the only one that should matter,
    as in the y-direction everyhting should be uniform. #TODO check whether this should change

    Parameters:
    data (interferometer-class object): data object containing the position data to discretize.
    pixel_pos (boolean): Indicates whether output is wanted in terms of pixel position or not. 
    If False, will return discretized position values that are in the middle of their respective pixels\n

    Returns:
    An array containing the pixel positions of each photon if pixel_pos is True.
    Pixel positions start at 0 and continue until (pos_range[0,1] - pos_range[0,0]) // res_pos.\n
    An array containing the positions of the middle of associated pixels of each photon if pixel_pos is False.
    """
    pix_edges = np.arange(ins.pos_range[0, 0], ins.pos_range[0, 1], ins.res_pos)
    pix_binner = spinter.interp1d(pix_edges, pix_edges, 'nearest', bounds_error=False)
    if pixel_pos:
        return ((pix_binner(data.energies) - ins.pos_range[0]) // ins.res_pos) - 1
    else:
        return pix_binner(data.energies) + ins.res_pos / 2

""

def update_D_test(p = False):
    """ Test to verify whether the 'update_D' from instrument.py works correctly. """


    #TODO 
    """Fix for current setup"""


    test_i = instrument.interferometer(2, .6, 1, 1, 1, .5)

    initial_D = test_i.D
    test_array = np.array([True for i in range(10)])

    if p:
        print(initial_D)

    test_i.set_target_D(test_i.D * 2)
    for i in range(len(test_array)):
        test_i.update_D()
        test_array[i] = test_i.D == initial_D

        if p: 
            print(test_i.D)

    if test_array[-1]:
        print("update_D failed to update D over time.")
    else:
        print("update_D succesfully updated D.")
        print("Please check with print enabled whether values are correct.")
    
    return

def wobbler_test(wobble_I, p = False):
    """ Test for the wobbler() function in instrument.py. """

    #TODO 
    """Fix for current setup"""

    test_i = instrument.interferometer(2, .6, 1, 1, 1, .5)
    test_i.wobbler(wobble_I)

    if test_i.theta and test_i.phi:
        print('The wobbler changed theta succesfully.')
        print('Please check with p = True whether values are correct.')
    else:
        print('The wobbler failed.')

    if p:
        for i in range(10):
            print(test_i.theta, test_i.phi)
            test_i.wobbler(wobble_I)
        print(test_i.theta, test_i.phi)

""

# E_edges = np.linspace(ins.E_range[0], ins.E_range[1], (ins.E_range[1] - ins.E_range[0])/ins.res_E)
# E_binner = spinter.interp1d(E_edges, E_edges, 'nearest', bounds_error=False)

# self.discrete_E = ((E_binner(self.energies) - ins.E_range[0]) // ins.res_E) - 1

# pix_edges = np.linspace(ins.pos_range[0, 0], ins.pos_range[0, 1], (ins.pos_range[0, 1] - ins.pos_range[0, 0])/ins.res_pos)
# pix_binner = spinter.interp1d(pix_edges, pix_edges, 'nearest', bounds_error=False)

# self.discrete_pos = ((pix_binner(self.pos[:, 0]) - ins.pos_range[0, 0]) // ins.res_pos) - 1