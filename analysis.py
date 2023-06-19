"""
This file is intended to contain code that can be used to analyse instrument data, in order to be able to draw meaningful conclusions from it.
""" 

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as ft
import scipy.constants as spc
import scipy.interpolate as spinter


def hist_data(data, binsno, pixs = False, num = 0):
    """
    Function that makes a histogram of direct output data from an interferometer object.

    Parameter:
    data (interferometer_data class object): Data to be plotted.
    binsno (int): Number of bins in the histogram.
    pixs (Boolean): whether the x-axis is in units of pixels or meters. If true, then in pixels. Default is False.
    """

    if pixs:
        plt.hist(data, binsno, label=f'Baseline {num}')
        plt.xlabel('Detector position (pixels)')
    else:
        plt.hist(data, binsno, label=f'Baseline {num}')
        plt.xlabel('Detector position (micrometers)')
    plt.ylabel('Counts')

def ft_data(y_data, samples, spacing):
    """
    Function that fourier transforms given input data from an interferometer.
    Works by first making a histogram of the positional data to then fourier transform that and obtain spatial frequencies.

    Parameters:
    data (interferometer_data class object): Data to be fourier transformed.
    samples (int): Number of samples for the fourier transform to take.
    """
    ft_x_data = ft.fftfreq(samples, spacing)
    ft_y_data = ft.fft(y_data) / y_data.size

    return ft_x_data, ft_y_data

def plot_ft(ft_x_data, ft_y_data, plot_obj, log=0, num= 0):
    """
    Function to plot fourier transformed interferometer data in a anumber of ways.

    Parameters:
    ft_x_data (array): fourier transformed data for the x-axis (so spatial frequencies).
    ft_y_data (array): fourier transformed data for the y-axis.
    log (int in [0,2]): indicates how many axes are to be in log scale, with 1 having only the y-axis in log.
    """
    if log == 0:
        plot_obj.plot(ft.fftshift(ft_x_data), (ft.fftshift(ft_y_data)), label=f'Baseline {num}')
    if log == 1:
        plot_obj.semilogy(ft.fftshift(ft_x_data), (ft.fftshift(ft_y_data)), label=f'Baseline {num}')
    if log == 2:
        plot_obj.loglog(ft.fftshift(ft_x_data), (ft.fftshift(ft_y_data)), label=f'Baseline {num}') 

def image_recon_smooth(data, instrument, point_binsize, fov, test_data=np.zeros((1,1)), samples = 512, progress = 0):
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

    def inverse_fourier(f_values, uv):
        """
        This is a helper function that calculates the inverse fourier transform of the data from all baselines, only to 
        be used at the last step of the parent function. It is sectioned off here for legibility.

        It first defines an image to fill in according to the provided samples size, and then pixel by pixel calculates the sum
        of all fourier components at that location. 
        """
        re_im = np.zeros(samples)
        shape = np.array(samples)

        # This lambda function is the formula for an inverse fourier transform, without the integration.
        # It is included here to make clear that a discrete inverse fourier transform is what is happening, and 
        # to make clear what argument means what.
        inverse_fourier = lambda x, y, u, v, fourier: fourier * np.exp(2j * np.pi * (u * x + v * y))

        for i, x in np.ndenumerate(re_im):
            # This is a diagnostic print that can be switched on to progress updates to ensure the code is still 
            # functioning while it is doing an exceptionally long task.
            if progress and i[1] == 0 and i[0] % samples[0] // 4 == 0:
                print(i[0] / samples[0])

            # j is the pixel coordinate i transformed to the sky coordinates (x,y) that the inverse fourier transform needs.
            j = (np.array(i) / shape.max() - (shape / (2 * shape.max()))) * fov * 2 * np.pi / (3600 * 360)
            re_im[i] = np.abs(np.sum(inverse_fourier(j[0], j[1], uv[:, 0], uv[:, 1], f_values)))

        return re_im

    # These arrays are all copied locally to reduce the amount of cross-referencing to other objects required.  
    pos_data = data.pixel_to_pos(instrument)
    time_data = data.discrete_t
    E_channels = np.unique(data.discrete_E)
    base_ind = data.baseline_indices
    pointing = data.pointing

    # Determing the binning of the roll angle and the number of points to be sampled in the uv-plane for later use.
    roll_bins = np.arange(0, 2 * np.pi, point_binsize)
    uv_points = roll_bins.size * len(instrument.baselines) * 2 * E_channels.size
    
    # Generating the arrays that will contain the uv coordinates and associated fourier values covered by the interferometer.
    uv = np.zeros((uv_points, 2))
    f_values = np.zeros(uv_points, dtype=np.complex_)

    bins = int(np.ceil(abs(instrument.pos_range[0] - instrument.pos_range[1]) / instrument.res_pos))
    edges = np.array([instrument.pos_range[0] + i * instrument.res_pos for i in range(bins + 1)])
    centres = edges[:-1] + (edges[1:] - edges[:-1])/2

    # This is for indexing the above two arrays
    i = 0

    for roll in roll_bins:
        # Binning data based on roll angle.
        # Retrieving all the indices of photons that are in the bin to later use to slice the data.
        ind_in_range = (((pointing[time_data, 2] % (2 * np.pi)) >= roll) * 
                        ((pointing[time_data, 2] % (2 * np.pi)) < roll + point_binsize))

        # If no photons are in range, everything below can be skipped
        if ind_in_range.any():
            # Reducing the data we take with us to the other steps
            data_bin = pos_data[ind_in_range]
            base_bin = base_ind[ind_in_range]

            for k in np.unique(base_bin):
                # Setting up data for the fourier transform, taking only relevant photons from the current baseline
                data_bin_k = data_bin[base_bin == k]
                y_data, _ = np.histogram(data_bin_k, edges)

                for E in E_channels:
                    # Calculating the wavelength of light we are dealing with, and the frequency that this baseline covers in the uv-plane with it.
                    lam_bin = spc.h * spc.c / ((E + .5) * instrument.res_E + instrument.E_range[0])
                    freq_baseline = instrument.baselines[k].D / lam_bin

                    # Calculating u and v for middle of current bin by taking a projection of the current frequency
                    u = freq_baseline * np.sin(roll  + point_binsize / 2)
                    v = freq_baseline * np.cos(roll  + point_binsize / 2)

                    # Calculating the frequency we will be doing the fourier transform for, which is the frequency we expect the fringes to appear at.
                    freq = 1 / (instrument.baselines[k].L * lam_bin / instrument.baselines[k].W)

                    # Calculating value of the fourier transform for the current frequency and bin
                    uv[i] = np.array([u, v])
                    f_values[i] = np.sum(y_data * np.exp(-2j * np.pi * freq * centres)) / y_data.size

                    # Doinng the same with the negative frequency
                    uv[i+1] = np.array([-u, -v])
                    # TODO test this as just complex conjugate of other
                    f_values[i+1] = np.conjugate(f_values[i])

                    i += 2  

    # Doing the final inverse fourier transform, and also returning the pre-ifft data, for visualization and testing.
    return inverse_fourier(f_values, uv), f_values, uv  