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
    image = images.point_source_multichromatic(int(1e5), 0.000, 0, [1.2, 1.6])

    # TODO 10 micron is better pixel size
    test_I = instrument.interferometer(.1, 1, 10, np.array([1.2, 6]), np.array([-1500, 1500]))
    test_I.add_baseline(1, 10, 300, 17000, 2, 1)

    start = time.time()
    test_data = process.process_image(test_I, image, 0, 10, int(1e4))
    print('Processing this image took ', time.time() - start, ' seconds')
    test_data.discretize_E(test_I)
    test_data.discretize_pos(test_I)

    print(int(np.amax(test_data.discrete_pos) - np.amin(test_data.discrete_pos)))
    analysis.hist_data(test_data.discrete_pos, int(np.amax(test_data.discrete_pos) - np.amin(test_data.discrete_pos)))
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

#wobble point source
def w_ps_test():
    image = images.point_source_multichromatic(int(1e5), 0.0001, 0, [1.2, 1.6])

    # TODO 10 micron is better pixel size
    test_I = instrument.interferometer(.1, .01, 10, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.001, None, instrument.interferometer.smooth_roller, 
                                        .01 * 2 * np.pi, 10, np.pi/4)
    test_I.add_baseline(1, 10, 300, 17000, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, int(1e4))
    print('Processing this image took ', time.time() - start, ' seconds')

    analysis.hist_data(test_data.pixel_to_pos(test_I)[:, 1], int(np.amax(test_data.discrete_pos[:, 1]) - np.amin(test_data.discrete_pos[:, 1])) + 1, False)
    ft_x_data, ft_y_data = analysis.ft_data(test_data.pixel_to_pos(test_I)[:, 1])
    analysis.plot_ft(ft_x_data, ft_y_data, 2)

def willingale_test():
    image = images.point_source_multichromatic(int(1e5), 0.001, 0, [1.2, 1.6])

    test_I = instrument.interferometer(.1, .01, 10, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.001, None, instrument.interferometer.smooth_roller, 
                                        .01 * 2 * np.pi, 10, np.pi/4)
    test_I.add_baseline(.035, 10, 300, 1200, 2, 1)
    test_I.add_baseline(.105, 10, 300, 3700, 2, 1)
    test_I.add_baseline(.315, 10, 300, 11100, 2, 1)
    test_I.add_baseline(.945, 10, 300, 33400, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, int(1e4))
    print('Processing this image took ', time.time() - start, ' seconds')

    for i in range(4):
        analysis.hist_data(test_data.pixel_to_pos(test_I)[:, 1][test_data.baseline_indices == i], 
                            int(np.amax(test_data.discrete_pos[:, 1][test_data.baseline_indices == i]) - 
                            np.amin(test_data.discrete_pos[:, 1][test_data.baseline_indices == i])) + 1, False, i)
    plt.legend()
    plt.show()

    for i in range(4):
        ft_x_data, ft_y_data = analysis.ft_data(test_data.pixel_to_pos(test_I)[:, 1][test_data.baseline_indices == i])
        analysis.plot_ft(ft_x_data, ft_y_data, 2, i)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # w_ps_test()
    # Fre_test()
    # scale_test2()
    # discretize_test()
    willingale_test()