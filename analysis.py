"""
This file is intended to contain code that can be used to analyse instrument data, in order to be able to draw meaningful conclusions from it.
""" 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.fft as ft

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
    samples = len(data) // 10
    y_data, edges = np.histogram(data, samples)
    ft_x_data = ft.fftfreq(samples, edges[-1] - edges[-2])
    ft_y_data = ft.fft(y_data)

    return ft_x_data, ft_y_data

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
