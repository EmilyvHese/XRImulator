from cProfile import label
import numpy as np
import scipy.special as sps
import scipy.interpolate as spinter
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

import images
import instrument
import process
import analysis

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

def ps_test():
    image = images.point_source(int(1e5), 0.0001, 0.00, 1.2)

    test_I = instrument.interferometer(0,0,0,0,0)
    test_I.add_baseline(1, 10, 300, 17000, 2, 1)

    start = time.time()
    test_data = process.process_image(test_I, image, 0, 10, int(1e4))
    print('Processing this image took ', time.time() - start, ' seconds')

    analysis.hist_data(test_data.discrete_pos, 100)
    ft_x_data, ft_y_data = analysis.ft_data(test_data)
    analysis.plot_ft(ft_x_data, ft_y_data, 2)

def dps_test():
    image = images.double_point_source(10000, [-.001, .001], [0, 0], [1.2, 6])

    test_I = instrument.interferometer(0,0,0,0,0)
    test_I.add_baseline(1, 10, 300, 17000, 2, 1)
    # print(image.loc[2,0])

    start = time.time()
    test_data = process.process_image(test_I, image, 0, 10, int(1e4))
    print('Processing this image took ', time.time() - start, ' seconds')

    analysis.hist_data(test_data.discrete_pos, 100)
    ft_x_data, ft_y_data = analysis.ft_data(test_data)
    analysis.plot_ft(ft_x_data, ft_y_data, 0)

def psmc_test():
    image = images.point_source_multichromatic(int(1e6), 0.0001, 0, [1.2, 1.6])

    # TODO 10 micron is better pixel size
    test_I = instrument.interferometer(.1, 0, .1, np.array([1.2, 6]), np.array([-1500, 1500]))
    test_I.add_baseline(1, 10, 300, 17000, 2, 1)

    start = time.time()
    test_data = process.process_image(test_I, image, 0, 10, int(1e4))
    print('Processing this image took ', time.time() - start, ' seconds')
    test_data.discretize_E(test_I)
    test_data.discretize_pos(test_I)

    analysis.hist_data(test_data.pixel_to_pos(test_I), 100)
    ft_x_data, ft_y_data = analysis.ft_data(test_data.pixel_to_pos(test_I))
    analysis.plot_ft(ft_x_data, ft_y_data, 2)

def Fre_test():
    # u_0 = np.linspace(4, 5, 100)
    u_0 = 4.5
    u_1 = lambda u, u_0: u + u_0/2
    u_2 = lambda u, u_0: u - u_0/2
    u = np.linspace(-5, 5, 1000) 

    u, u_0 = np.meshgrid(u, u_0, indexing='ij') 
    S_1, C_1 = sps.fresnel(u_1(u, u_0))
    S_2, C_2 = sps.fresnel(u_2(u, u_0))

    A = (C_2 - C_1 + 1j*(S_2 - S_1)) * (1 + np.exp(np.pi * 1j * u_0 * u))
    A_star = np.conjugate(A)

    I = A * A_star

    fig = plt.figure(figsize=(8, 6))
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(u, u_0, np.real(I))
    # ax.set_xlabel('u')
    # ax.set_ylabel('u_0')
    # ax.set_zlabel('I')

    plt.plot(u, np.real(I))

    plt.show()

def scale_test():
    func = lambda k, x: 2 + 2 * np.cos(k * x)
    x = np.linspace(-5, 5, 10000)
    k = 2 * np.pi / np.linspace(1, 10, 10000)
    x_grid, k_grid = np.meshgrid(x, k)
    I = func(k_grid, x_grid)

    fig = plt.figure(figsize=(8, 6))
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(x_grid, k_grid, I)
    # ax.set_xlabel('x')
    # ax.set_ylabel('k')
    # ax.set_zlabel('I')

    plt.plot(x, I[0,:])
    plt.plot(x, I[50,:])
    plt.show()

def scale_test2():
    func = lambda k, x: 2 + 2 * np.cos(k * x)
    x = np.linspace(0, 2, 1000)
    plt.plot(x, func(1, x), label="1, x")
    plt.plot(x, func(2, x), label="2, x")
    plt.plot(2*x, func(2, x), label="2, 2x")
    plt.legend()
    plt.show()

def discretize_E_test(E_range, res_E, data):
    """
    Function that discretizes energies of incoming photons into energy channels.

    Parameters:
    data (interferometer-class object): data object containing the energy data to discretize.
    """
    E_edges = np.arange(E_range[0], E_range[1], res_E)
    E_binner = spinter.interp1d(E_edges, E_edges, 'nearest', bounds_error=False)
    return E_binner(data.energies)

def discretize_test():
    data = process.interferometer_data(10)
    data.energies = np.array([1.6, 2.3, 3.1, 4.2, 5.5, 6.0, 7.3, 8.1, 9.9, 10.6])

    print(discretize_E_test([0, 11], 1, data))


if __name__ == "__main__":
    psmc_test()
    # Fre_test()
    # scale_test2()
    # discretize_test()