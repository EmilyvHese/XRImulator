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

def image_recon_smooth(data, instrument, pointing, point_binsize):
    pos_data = data.pixel_to_pos(instrument)[:, 1]
    time_data = data.discrete_t
    base_ind = data.baseline_indices
    
    fourier_data = np.zeros((len(pos_data), 3))
    index = 0
    for roll in np.arange(0, pointing[-1, 2], point_binsize):
        ind_in_range = (pointing[time_data, 2] > roll) * (pointing[time_data, 2] <= roll + point_binsize)
        data_bin = pos_data[ind_in_range]

        for i in range(len(instrument.baselines)):
            data_bin_i = data_bin[base_ind[ind_in_range] == i]
            samples = data_bin_i.size
            y_data, edges = np.histogram(data_bin_i, samples)

            ft_x_data = ft.fftfreq(samples, edges[-1] - edges[-2])
            sampled_freq_range = ((ft_x_data > instrument.baselines[i].D / 1e-10) * (ft_x_data >= instrument.baselines[i].D / 1e-8))
            sliced_ft_x = ft_x_data[sampled_freq_range]
            actual_samples = sliced_ft_x.size

            # Calculating u for middle of current bin
            fourier_data[index:index+actual_samples, 1] = sliced_ft_x * np.cos(roll + point_binsize / 2)
            # Calculating v for middle of current bin
            fourier_data[index:index+actual_samples, 2] = sliced_ft_x * np.sin(roll + point_binsize / 2)
            # Calculating magnitudes of fourier components
            fourier_data[index:index+actual_samples, 0] = ft.fft(y_data)[sampled_freq_range]

            index += samples

    recon_image = ft.ifft2(fourier_data)

    return fourier_data, recon_image

def image_recon_smooth(data, instrument, point_binsize):
    """
    This function is to be used to reconstruct images from interferometer data.
    Bins input data based on roll angle, which is important to fill out the uv-plane that will be fed into 
    the inverse fourier transform.

    Args:
        data (interferometer_data class object): The interferometer data to recover an image from.
        instrument (interferometer class object): The interferometer used to record the aforementioned data.
        point_binsize (float): Size of roll angle bins.

    Returns:
        array, array: Two arrays, first of which is the recovered image, second of which is the array used in the ifft.
    """    
    # Setting up some necessary parameters
    pos_data = data.pixel_to_pos(instrument)[:, 1]
    time_data = data.discrete_t
    base_ind = data.baseline_indices
    pointing = data.pointing
    samples = 512
    
    # Calculating a grid image of the fourier transformed data that can be 2d-inverse fourier transformed.
    f_grid = np.zeros((samples, samples), dtype=np.complex_)
    fft_freqs = ft.fftfreq(samples, instrument.res_pos)
    fft_freq_ind = np.arange(0, fft_freqs.size, 1, dtype=np.int_)
    freqs_conv = spinter.interp1d(fft_freqs, fft_freq_ind, kind='nearest')

    for roll in np.arange(0, pointing[-1, 2], point_binsize):
        # Binning data based on roll angle.
        ind_in_range = (pointing[time_data - 1, 2] >= roll) * (pointing[time_data - 1, 2] < roll + point_binsize)
        data_bin = pos_data[ind_in_range]

        for i in range(len(instrument.baselines)):
            # Setting up data for the fourier transform, taking only relevant photons from the current baseline
            delta_u = 1 / np.sqrt(instrument.baselines[i].L * spc.h * spc.c / (np.array([1.17, 1.23]) * 1.602177733e-16 * 10))
            data_bin_i = data_bin[base_ind[ind_in_range] == i]
            y_data, edges = np.histogram(data_bin_i, samples)
            ft_x_data = ft.fftfreq(samples, edges[-1] - edges[-2])

            # Making a mask to ensure only relevant data is taken
            sampled_freq_range = ((abs(ft_x_data) > delta_u[0]) * (abs(ft_x_data) <= delta_u[1]))
            sliced_ft_x = ft_x_data[sampled_freq_range]

            # # Calculating u for middle of current bin
            # print(instrument.baselines[i].D * np.cos(roll + point_binsize / 2) / (spc.h * spc.c / (1.2 * 1.602177733e-16)))
            # u = freqs_conv(instrument.baselines[i].D * np.cos(roll + point_binsize / 2) / (spc.h * spc.c / (1.2 * 1.602177733e-16)))
            # # Calculating v for middle of current bin
            # v = freqs_conv(-instrument.baselines[i].D * np.cos(roll + point_binsize / 2) / (spc.h * spc.c / (1.2 * 1.602177733e-16)))

            # Calculating u for middle of current bin
            u = freqs_conv(sliced_ft_x * np.cos(roll + point_binsize / 2))
            # Calculating v for middle of current bin
            v = freqs_conv(sliced_ft_x * np.sin(roll + point_binsize / 2))
            # Calculating magnitudes of fourier components
            # TODO Look into directly calculating fourier transform
            f_grid[u.astype(int), v.astype(int)] += ft.fft(y_data)[sampled_freq_range]

    # Doing the final inverse fourier transform, and also returning the pre-ifft data.
    return ft.ifft2(f_grid), f_grid

# New and improved version using the pdf directly which should result in slightly faster results
# Nevermind does not work faster in cases where light is coming from edges of FOV
# u_pos = np.random.choice(u, self.size, True, I_pdf)
# self.actual_pos[:, 1] = (u_pos / np.sqrt(2 / (lambdas * baseline_data[:, 2]))) - baseline_data[:, 1] * theta
# i = 0
# while self.actual_pos[abs(self.actual_pos[:, 1]) > baseline_data[:, 0]/2].any():
#     unacc_ind = abs(self.actual_pos[:, 1]) > baseline_data[:, 0]/2

#     u_pos = np.random.choice(u, self.actual_pos[unacc_ind, 1].size, True, I_pdf)
#     self.actual_pos[unacc_ind, 1] = (u_pos / np.sqrt(2 / (lambdas[unacc_ind] * baseline_data[unacc_ind, 2])) 
#                                      - baseline_data[unacc_ind, 1] * theta[unacc_ind])
    
#     print(i, u_pos, self.actual_pos[unacc_ind, 1].size)
#     i += 1


def image_recon_smooth(data, instrument, point_binsize, test_data=np.zeros((1,1)), samples = 512, exvfast = 0):
    """
    This function is to be used to reconstruct images from interferometer data.
    Bins input data based on roll angle, which is important to fill out the uv-plane that will be fed into 
    the 2d inverse fourier transform.

    Args:
        data (interferometer_data class object): The interferometer data to recover an image from.
        instrument (interferometer class object): The interferometer used to record the aforementioned data.
        point_binsize (float): Size of roll angle bins.
        samples (int): N for the NxN matrix that is the uv-plane used for the 2d inverse fourier transform.
        
    Returns:
        array, array: Two arrays, first of which is the recovered image, second of which is the array used in the ifft.
    """    
    # Setting up some necessary parameters
    pos_data = data.pixel_to_pos(instrument)[:, 1]
    time_data = data.discrete_t
    base_ind = data.baseline_indices
    pointing = data.pointing
    
    # Calculating a grid image of the fourier transformed data that can be 2d-inverse fourier transformed.
    f_grid = np.zeros(samples, dtype=np.complex_)
    f_coverage = np.zeros(samples)
    uv = np.zeros(samples, dtype=np.complex_)

    # fft_freqs_u = ft.fftfreq(int((instrument.pos_range[1] - instrument.pos_range[0]) /  instrument.res_pos), instrument.res_pos)
    fft_freqs_u = ft.fftfreq(samples[0], instrument.res_pos)
    fft_freq_ind_u = np.arange(0, fft_freqs_u.size, 1, dtype=np.int_)
    u_conv = spinter.interp1d(fft_freqs_u, fft_freq_ind_u, kind='nearest')

    # fft_freqs_v = ft.fftfreq(int((instrument.pos_range[1] - instrument.pos_range[0]) /  instrument.res_pos), instrument.res_pos)
    fft_freqs_v = ft.fftfreq(samples[1], instrument.res_pos)
    fft_freq_ind_v = np.arange(0, fft_freqs_v.size, 1, dtype=np.int_)
    v_conv = spinter.interp1d(fft_freqs_v, fft_freq_ind_v, kind='nearest')

    for roll in np.arange(0, 2 * np.pi, point_binsize):
        # Binning data based on roll angle.
        ind_in_range = (((pointing[time_data, 2] % (2 * np.pi)) >= roll) * 
                            ((pointing[time_data, 2] % (2 * np.pi)) < roll + point_binsize))

        if ind_in_range.any():
            data_bin = pos_data[ind_in_range]
            base_bin = base_ind[ind_in_range]

            for i in range(len(instrument.baselines)):
                lam_bin = (spc.h * spc.c / (1.2 * spc.eV))

                # Setting up data for the fourier transform, taking only relevant photons from the current baseline
                data_bin_i = data_bin[base_bin == i]
                y_data, edges = np.histogram(data_bin_i, int(np.ceil(instrument.baselines[i].W / instrument.res_pos)) + 1)
                centres = edges[:-1] + (edges[1:] - edges[:-1])/2

                freq_baseline = instrument.baselines[i].D / lam_bin

                # Calculating u for middle of current bin by taking a projection of the current frequency
                u = u_conv(freq_baseline * np.sin(roll + point_binsize / 2))
                # Calculating v for middle of current bin by taking a projection of the current frequency
                v = v_conv(freq_baseline * np.cos(roll + point_binsize / 2))

                # Calculating the frequency we will be doing the fourier transform for, which is the frequency we expect the fringes to appear at.
                freq = 1 / np.sqrt(instrument.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))

                # Calculating magnitude of the fourier transform for the current frequency and bin
                f_grid[u.astype(int), v.astype(int)] += np.sum(y_data * np.exp(-2j * np.pi * freq * centres)) 
                f_grid[-u.astype(int), -v.astype(int)] += np.sum(y_data * np.exp(-2j * np.pi * -freq * centres))
                f_coverage[[u.astype(int), -u.astype(int)], [v.astype(int), -v.astype(int)]] += y_data.size

                if test_data.any():
                    uv[u.astype(int), v.astype(int)] = test_data[u.astype(int), v.astype(int)]
                    uv[-u.astype(int), -v.astype(int)] = test_data[-u.astype(int), -v.astype(int)]       

                # if exvfast == 1:
                #     ft_x_data = ft.fftshift(ft.fftfreq(int(instrument.baselines[i].W // instrument.res_pos) + 1, instrument.res_pos))
                #     ft_y_data = np.array([np.sum(y_data * np.exp(-2j * np.pi * f * centres)) for f in ft_x_data]) / y_data.size
                #     exact_data = np.array([np.sum(y_data * np.exp(-2j * np.pi * freq * centres)) / y_data.size, np.sum(y_data * np.exp(-2j * np.pi * -freq * centres)) / y_data.size])
                #     fft_y_data = ft.fftshift(ft.fft(y_data) / y_data.size)

                #     fig, (ax1, ax2) = plt.subplots(2, 1)

                #     ax1.plot(ft_x_data, np.abs(ft_y_data), label='Exact')
                #     ax1.plot(ft_x_data, np.abs(fft_y_data), label='Fast')
                #     ax1.vlines([freq, -freq], np.amin(ft_y_data) - 2, np.amax(ft_y_data) + 1, color='red', label='Expected frequency')
                #     ax1.plot([freq, -freq], np.abs(exact_data), 'ro', label='Exact point')
                #     ax1.set_title('Exact vs. fast fourier transform of data')
                #     ax1.set_xlabel('Spatial frequency (m$^{-1}$)')
                #     ax1.set_ylabel('Amplitude')
                #     ax1.set_xlim(-freq * 1.2, freq * 1.2)
                #     ax1.legend()

                #     ax2.plot(ft_x_data, np.angle(ft_y_data), label='Exact')
                #     ax2.plot(ft_x_data, np.angle(fft_y_data), label='Fast')
                #     ax2.vlines([freq, -freq], np.amin(np.imag(ft_y_data)) - 1, np.amax(np.imag(ft_y_data)) + 1, color='red', label='Expected frequency')
                #     ax2.plot([freq, -freq], np.angle(exact_data), 'ro', label='Exact point')
                #     ax2.set_title('Exact vs. fast fourier transform of data')
                #     ax2.set_xlabel('Spatial frequency (m$^{-1}$)')
                #     ax2.set_ylabel('Phase')
                #     ax2.set_xlim(-freq * 1.2, freq * 1.2)
                #     ax2.set_ylim(-np.pi, np.pi)
                #     ax2.legend()
                    
                #     plt.show()

                #     exvfast = 0

    f_grid[f_coverage.nonzero()] /= f_coverage[f_coverage.nonzero()]

    # Doing the final inverse fourier transform, and also returning the pre-ifft data, for visualization and testing.
    return ft.ifft2(f_grid), f_grid, uv 

#Old processing
def process_photon_dpos(self, instrument, image, N_f, samples):
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
            I_pdf = I / sum(I)

            return I_pdf, u

        self.actual_pos = np.zeros((self.size, 2))
        # Calculating photon wavelengths and phases from their energies
        # wavelengths = spc.h * spc.c / image.energies

        # Randomly selecting a baseline for the photon to go in. 
        self.baseline_indices = np.random.randint(0, len(instrument.baselines), self.size)

        # Getting data from those baselines that is necessary for further calculations.
        # Must be done this way since baselines[unacc_ind].W doesn't go element-wise, and .W is not an array operation
        # baseline_data = np.array([[instrument.baselines[index].W, 
        #                             instrument.baselines[index].F,
        #                             instrument.baselines[index].L] for index in self.baseline_indices])

        # Calculating the fresnell diffraction pattern for set number of fringes and samples
        # self.inter_pdf = fre_dif(N_f, samples)

        # Defining the pointing, relative position and off-axis angle for each photon over time.
        # Relative position is useful for the calculation of theta, since the off-axis angle is very dependent on where the axis is.
        # There is a 1e-20 factor in a denominator, to prevent divide by zero errors. Typical values for pos_rel are all much larger, 
        # so this does not simply move the problem.
        self.pointing = instrument.gen_pointing(np.max(image.toa))
        pos_rel = self.pointing[image.toa, :2] - image.loc
        theta = np.cos(self.pointing[image.toa, 2] - np.arctan2(pos_rel[:, 0], pos_rel[:, 1])) * np.sqrt(pos_rel[:, 0]**2 + pos_rel[:, 1]**2)

        # Doing an accept/reject method to find the precise location photons impact at.
        # This array records which photons are accepted so far, starting as all False and becoming True when one is accepted
        # accepted_array = np.full(image.size, False, bool)
        # while np.any(accepted_array == False):
        #     # Indices of all unaccepted photons, which are the only ones that need to be generated again each loop.
        #     unacc_ind = np.nonzero(accepted_array == False)[0]

        #     # Generating new photons for all the unaccepted indices with accurate y-locations and random intensities.
        #     photon_y = (np.random.rand(unacc_ind.size) * baseline_data[unacc_ind, 0] - baseline_data[unacc_ind, 0]/2)

        #     # Converting y positions to u positions for scaling the fresnell diffraction to            
        #     photon_u = (photon_y + baseline_data[unacc_ind, 1] * theta[unacc_ind]) * np.sqrt(2 / (wavelengths[unacc_ind] * baseline_data[unacc_ind, 2]))
        #     photon_fresnell = self.inter_pdf(photon_u)
            
        #     photon_I = np.random.rand(unacc_ind.size) * np.amax(photon_fresnell) 

        #     # Checking which photons will be accepted, and updating the accepted_array accordingly
        #     accepted_array[unacc_ind] = photon_I <= photon_fresnell
        #     self.actual_pos[unacc_ind, 1] = photon_y 