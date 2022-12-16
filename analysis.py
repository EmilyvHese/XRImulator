"""
This file is intended to contain code that can be used to analyse instrument data, in order to be able to draw meaningful conclusions from it.
""" 

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as ft
import scipy.constants as spc


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
    # plt.show()

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
    # print(ft_y_data)

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

    # plt.ylim(bottom=samples/10)
    # plt.show()

def image_recon_smooth(data, instrument, pointing, point_binsize):
    pos_data = data.pixel_to_pos(instrument)[:, 1]
    time_data = data.discrete_t
    base_ind = data.baseline_indices
    
    fourier_data = np.zeros((pos_data.size, 4), dtype=np.complex_)
    index = 0

    for roll in np.arange(0, pointing[-1, 2], point_binsize):
        ind_in_range = (pointing[time_data - 1, 2] > roll) * (pointing[time_data - 1, 2] <= roll + point_binsize)
        data_bin = pos_data[ind_in_range]

        for i in range(len(instrument.baselines)):
            data_bin_i = data_bin[base_ind[ind_in_range] == i]
            samples = data_bin_i.size
            y_data, edges = np.histogram(data_bin_i, samples)
            delta_u = 1 / np.sqrt(instrument.baselines[i].L * spc.h * spc.c / (np.array([1.2, 1.6]) * 1.602177733e-16 * 10))

            ft_x_data = ft.fftfreq(samples, edges[-1] - edges[-2])
            sampled_freq_range = ((ft_x_data > delta_u[0]) * (ft_x_data <= delta_u[1]))
            sliced_ft_x = ft_x_data[sampled_freq_range]
            actual_samples = sliced_ft_x.size

            # Calculating magnitudes of fourier components
            fourier_data[index:index+actual_samples, 0] = ft.fft(y_data)[sampled_freq_range]
            # Calculating u for middle of current bin
            fourier_data[index:index+actual_samples, 1] = sliced_ft_x * np.cos(roll + point_binsize / 2)
            # Calculating v for middle of current bin
            fourier_data[index:index+actual_samples, 2] = sliced_ft_x * np.sin(roll + point_binsize / 2)

            fourier_data[index:index+actual_samples, 3] = i

            index += samples

    # u_grid, v_grid = np.meshgrid(fourier_data[:, 1], fourier_data[:, 2], indexing='ij')
    u_max = 1 / np.sqrt(instrument.baselines[0].L * spc.h * spc.c / (10 * 1.602177733e-16 * 10))
    f_grid = np.zeros((int(u_max/50), int(u_max/50)), dtype=np.complex_)
    for i in range(pos_data.size):
        f_grid[int(fourier_data[i, 1]/100), int(fourier_data[i, 2]/100)] += fourier_data[i, 0]

    recon_image = ft.ifft2(f_grid)

    return fourier_data, recon_image, f_grid


    