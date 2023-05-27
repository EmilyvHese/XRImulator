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

    fft_freqs_u = ft.fftfreq(samples[0], instrument.res_pos)
    fft_freq_ind_u = np.arange(0, fft_freqs_u.size, 1, dtype=np.int_)
    u_conv = spinter.interp1d(fft_freqs_u, fft_freq_ind_u, kind='nearest')

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

                #TODO look at fractionalizing, normalize by number of counts

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