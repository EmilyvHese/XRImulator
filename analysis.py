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
        plt.hist(data * 1e6, binsno, label=f'Baseline {num}')
        plt.xlabel('Detector position (micrometers)')
    plt.ylabel('Counts')

def ft_data(data):
    """
    Function that fourier transforms given input data from an interferometer.
    Works by first making a histogram of the positional data to then fourier transform that and obtain spatial frequencies.

    Parameters:
    data (interferometer_data class object): Data to be fourier transformed.
    """
    samples = len(data)
    y_data, edges = np.histogram(data, samples)
    ft_x_data = ft.fftfreq(samples, edges[-1] - edges[-2])
    ft_y_data = ft.fft(y_data)

    return ft_x_data, ft_y_data, edges

def plot_ft(ft_x_data, ft_y_data, log=0, num= 0):
    """
    Function to plot fourier transformed interferometer data in a anumber of ways.

    Parameters:
    ft_x_data (array): fourier transformed data for the x-axis (so spatial frequencies).
    ft_y_data (array): fourier transformed data for the y-axis.
    log (int in [0,2]): indicates how many axes are to be in log scale, with 1 having only the y-axis in log.
    """
    if log == 0:
        plt.plot(ft.fftshift(ft_x_data), abs(ft.fftshift(ft_y_data)), label=f'Baseline {num}')
    if log == 1:
        plt.semilogy(ft.fftshift(ft_x_data), abs(ft.fftshift(ft_y_data)), label=f'Baseline {num}')
    if log == 2:
        plt.loglog(ft.fftshift(ft_x_data), abs(ft.fftshift(ft_y_data)), label=f'Baseline {num}')

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
        ind_in_range = (pointing[time_data - 1, 2] > roll) * (pointing[time_data - 1, 2] <= roll + point_binsize)
        data_bin = pos_data[ind_in_range]

        for i in range(len(instrument.baselines)):
            # Setting up data for the fourier transform, taking only relevant photons from the current baseline
            delta_u = 1 / np.sqrt(instrument.baselines[i].L * spc.h * spc.c / (np.array([1., 1.4]) * 1.602177733e-16 * 10))
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
            f_grid[u.astype(int), v.astype(int)] += ft.fft(y_data)[sampled_freq_range]

    # Doing the final inverse fourier transform, and also returning the pre-ifft data.
    return ft.ifft2(f_grid), f_grid


    