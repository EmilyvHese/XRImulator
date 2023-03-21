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
        histed_data = plt.hist(data, binsno, label=f'Baseline {num}')
        plt.xlabel('Detector position (pixels)')
    else:
        histed_data = plt.hist(data * 1e6, binsno, label=f'Baseline {num}')
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
    ft_y_data = ft.fft(y_data)

    return ft_x_data, ft_y_data / y_data.size

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

def image_recon_smooth(data, instrument, point_binsize, test_data=np.zeros((1,1)), samples = 512):
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
    discrete_data = data.discrete_pos[:, 1]
    time_data = data.tstep_to_t(instrument).astype(int)
    base_ind = data.baseline_indices
    pointing = data.pointing
    
    # Calculating a grid image of the fourier transformed data that can be 2d-inverse fourier transformed.
    f_grid = np.zeros((samples, samples), dtype=np.complex_)
    fft_freqs = ft.fftfreq(samples, instrument.res_pos)
    fft_freq_ind = np.arange(0, fft_freqs.size, 1, dtype=np.int_)
    freqs_conv = spinter.interp1d(fft_freqs, fft_freq_ind, kind='nearest')
    uv = np.zeros((samples, samples), dtype=np.complex_)

    for roll in np.arange(0, pointing[-1, 2] % (2 * np.pi), point_binsize):
        # Binning data based on roll angle.
        ind_in_range = (((pointing[time_data, 2] % (2 * np.pi)) >= (roll % (2 * np.pi))) * 
                            ((pointing[time_data, 2] % (2 * np.pi)) < (((roll + point_binsize) % (2 * np.pi)))))

        data_bin = pos_data[ind_in_range]
        disc_bin = discrete_data[ind_in_range]
        base_bin = base_ind[ind_in_range]


        for i in range(len(instrument.baselines)):

            if disc_bin.any() == False:
                print(disc_bin)
                break

            # Setting up data for the fourier transform, taking only relevant photons from the current baseline
            data_bin_i = data_bin[base_bin == i]
            y_data, edges = np.histogram(data_bin_i, int(np.amax(disc_bin[base_bin == i]) - 
                            np.amin(disc_bin[base_bin == i])) + 1)
            centres = edges[:-1] + (edges[1:] - edges[:-1])/2

            # Calculating the frequency we will be doing the fourier transform for, which is the frequency we expect the fringes to appear at.
            freq = 1 / np.sqrt(instrument.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))

            # Calculating u for middle of current bin by taking a projection of the current frequency
            u = freqs_conv(freq * np.cos(roll + point_binsize / 2))
            # Calculating v for middle of current bin by taking a projection of the current frequency
            v = freqs_conv(freq * np.sin(roll + point_binsize / 2))
            # Calculating magnitude of the fourier transform for the current frequency and bin

            f_grid[v.astype(int), u.astype(int)] += np.sum(y_data * np.exp(-2j * np.pi * freq * centres)) / y_data.size
            f_grid[-v.astype(int), -u.astype(int)] += np.sum(y_data * np.exp(-2j * np.pi * -freq * centres)) / y_data.size

            # if roll == 0 and i == 0:
            #     freqs_test = np.linspace(-50000, 50000, int(1e4))
            #     four_test = np.zeros(freqs_test.size)
            #     for j, freq_test in enumerate(freqs_test):
            #         four_test[j] = np.sum(y_data * np.exp(-2j * np.pi * freq_test * centres))
            #     plt.plot(freqs_test, four_test)
            #     plt.vlines([-freq, freq], [0, 0], [1e4, 1e4], color='r')
            #     plt.show()

            if test_data.any():
                uv[v.astype(int), u.astype(int)] = test_data[v.astype(int), u.astype(int)]
                uv[-v.astype(int), -u.astype(int)] = test_data[-v.astype(int), -u.astype(int)]       

        # y_data, edges = np.histogram(data_bin, int(np.amax(disc_bin) - 
        #                     np.amin(disc_bin)) + 1)
        # hist_data(data_bin, int(np.amax(disc_bin) - 
        #                     np.amin(disc_bin)) + 1)
        # plt.title(roll)
        # plt.show()

    # Doing the final inverse fourier transform, and also returning the pre-ifft data, for visualization and testing.
    return ft.ifft2(f_grid), f_grid, uv   