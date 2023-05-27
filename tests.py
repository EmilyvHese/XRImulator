import numpy as np
import scipy.special as sps
import scipy.constants as spc
import scipy.interpolate as spinter
import scipy.stats as spst
import scipy.fft as ft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.cm as cm
from PIL import Image


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
    image = images.point_source_multichromatic(int(1e5), 0.00, 0, [1.2, 1.6])

    test_I = instrument.interferometer(.1, .01, 4, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.001, None, instrument.interferometer.smooth_roller, 
                                        .00001 * 2 * np.pi, 10, np.pi/4)
    test_I.add_baseline(.035, 10, 300, 1200, 2, 1)
    test_I.add_baseline(.105, 10, 300, 3700, 2, 1)
    test_I.add_baseline(.315, 10, 300, 11100, 2, 1)
    test_I.add_baseline(.945, 10, 300, 33400, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, int(1e4))
    print('Processing this image took ', time.time() - start, ' seconds')

    # for i in range(4):
    #     analysis.hist_data(test_data.pixel_to_pos(test_I)[:, 1][test_data.baseline_indices == i], 
    #                         int(np.amax(test_data.discrete_pos[:, 1][test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[:, 1][test_data.baseline_indices == i])) + 1, False, i)
    # plt.legend()
    # plt.show()

    # for i in range(4):
    #     ft_x_data, ft_y_data, edges = analysis.ft_data(test_data.pixel_to_pos(test_I)[:, 1][test_data.baseline_indices == i])
    #     analysis.plot_ft(ft_x_data, ft_y_data, 0, i)
    # delta_u = 1 / np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (np.array([1.2, 1.6]) * 1.602177733e-16 * 10))
    # plt.axvline(delta_u[0], 1e-5, 1e4)
    # plt.axvline(delta_u[1], 1e-5, 1e4)
    # plt.legend()
    # plt.xlim(-2 * delta_u[1], 2 * delta_u[1])
    # plt.show()

    # test = np.linspace(-4, 4, 1000)
    # plt.plot(test, test_data.inter_pdf(test))
    # plt.show()

    ft_data, re_im, f_grid = analysis.image_recon_smooth(test_data, test_I, test_data.pointing, .01 * 2 * np.pi)
    # for i in range(4):
    #     ft_base = ft_data[ft_data[:, 3] == i]
    #     plt.plot(ft_base[:, 1], ft_base[:, 2], '.', label=f'baseline {i}')
    # plt.legend()
    # plt.show()

    plt.imshow(abs(f_grid), cmap=cm.Reds)
    plt.show()

    plt.imshow(abs(re_im), cmap=cm.Greens)
    # plt.plot(re_im[:,0], re_im[:,2], label='0-2')
    # plt.legend()
    plt.show()

    # plt.plot(re_im[:,0], re_im[:,1])
    # plt.show()

def image_re_test():
    # TODO Try without diffraction, single baseline, with something with a peak at the centre
    # Try starting simple, add effects until divergence point
    # Maybe don't even throw it through the sampling

    # TODO check fringe generation to be sure

    # TODO check with basic sine wave outside normal code

    # TODO Echt echt echt verder kijken naar de fases

    # TODO kijk naar diagnostic codes
    image = images.double_point_source(int(1e5), [0.0001, -.0001], [0.0001, -.0001], [1.2, 1.2])

    test_I = instrument.interferometer(.1, .01, 4, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        .0005 * 2 * np.pi)
    # test_I.add_baseline(.035, 300, 1200, 2, 1)
    # test_I.add_baseline(.105, 300, 3700, 2, 1)
    # test_I.add_baseline(.315, 300, 11100, 2, 1)
    test_I.add_baseline(.945, 10, 300, 33400, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, 512)
    print('Processing this image took ', time.time() - start, ' seconds')

    exp = test_I.baselines[0].F * np.cos(-np.arctan(image.loc[0, 0] / (image.loc[0, 1] + 1e-20))) * np.sqrt(image.loc[0, 0]**2 + image.loc[0, 1]**2) * 1e6
    for i in range(len(test_I.baselines)):
        analysis.hist_data(test_data.actual_pos[:, 1][test_data.baseline_indices == i], 
                            int(np.ceil(test_I.baselines[i].W / test_I.res_pos)) + 1, False, i)
        # print(test_I.baselines[i].F / test_I.baselines[i].D)
    # print(exp /33400 * 1e-6)
    plt.vlines(-exp, -100, 10000)
    plt.title('Photon impact positions on detector')
    plt.ylim(0, 5000)
    plt.legend()
    plt.show()

    colourlist = ['b', 'orange', 'g', 'r']
    for i in range(len(test_I.baselines)):
        samples = int(np.ceil(test_I.baselines[i].W / test_I.res_pos)) + 1
        ft_x_data, ft_y_data, edges = analysis.ft_data(test_data.pixel_to_pos(test_I)[:, 1][test_data.baseline_indices == i], samples)
        analysis.plot_ft(ft_x_data, ft_y_data, 0, i)
        delta_u = 1 / np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * 1.602177733e-16 * 10))
        plt.axvline(delta_u, 1e-5, 1e4, color = colourlist[i])
        plt.axvline(-delta_u, 1e-5, 1e4, color = colourlist[i])
    plt.xlim(-4 * delta_u, 4 * delta_u)
    plt.title('Fourier transform of photon positions')
    plt.xlabel('Spatial frequency ($m^{-1}$)')
    plt.ylabel('Fourier magnitude')
    plt.legend()
    plt.show()

    start = time.time()
    test_data_imre = np.zeros((512, 512))
    test_data_imre[256, 256] = 1
    test_data_imre = ft.fft2(test_data_imre)
    re_im, f_grid, test_data_masked = analysis.image_recon_smooth(test_data, test_I, .01 * 2 * np.pi, samples=512, test_data=test_data_imre)
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    test_image = ft.ifft2(test_data_imre)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(abs(ft.fftshift(f_grid)), cmap=cm.Reds)
    ax1.set_title('UV-plane (amplitude)')
    ax2.imshow(abs(1j * ft.fftshift(f_grid)), cmap=cm.Blues)
    ax2.set_title('UV-plane (phase)')
    ax3.imshow(abs(re_im), cmap=cm.Greens)
    ax3.set_title('Reconstructed image')
    ax4.imshow(abs(ft.fftshift(test_data_masked)), cmap=cm.Blues)
    ax4.set_title('UV-plane (amplitude)')
    ax5.imshow(abs(1j * ft.fftshift(test_data_masked)), cmap=cm.Greens)
    ax5.set_title('UV-plane (phase)')
    ax6.imshow(abs(test_image), cmap=cm.Greens)
    ax6.set_title('Reconstructed image')
    plt.show()

def stats_test():
    """This test exists to do some statistical testing"""
    offset = 0e-6
    image = images.point_source(int(1e5), 0.000, offset, 1.2)

    no_sims = 1000
    simulated = np.zeros((no_sims, 4), dtype=np.complex_)
    masked = np.zeros((no_sims, 4), dtype=np.complex_)

    for sim in range(no_sims):
        test_I = instrument.interferometer(.1, 1, .5, np.array([1.2, 6]), np.array([-400, 400]), 
                                            0.00, None, instrument.interferometer.smooth_roller, 
                                            .00000 * 2 * np.pi, roll_init=0/4 * np.pi)
        test_I.add_baseline(.035, 10, 300, 1200, 2, 1)
        test_I.add_baseline(.105, 10, 300, 3700, 2, 1)
        # test_I.add_baseline(.315, 10, 300, 11100, 2, 1)
        # test_I.add_baseline(.945, 10, 300, 33400, 2, 1)

        test_data = process.interferometer_data(test_I, image, 10, 512)

        test_data_imre = np.zeros((512, 512))
        test_data_imre[256, 256] = 1
        test_data_imre_fft = ft.fft2(test_data_imre)
        re_im, f_grid, test_data_masked = analysis.image_recon_smooth(test_data, test_I, .02 * 2 * np.pi, samples=test_data_imre.shape, test_data=test_data_imre_fft, exvfast=0)

        simulated[sim] = f_grid[f_grid.nonzero()] / np.sum(np.abs(f_grid[f_grid.nonzero()]))
        masked[sim] = test_data_masked[f_grid.nonzero()] / np.sum(np.abs(test_data_masked[f_grid.nonzero()]))

        if sim % 10 == 0:
            print(f'Done with sim {sim}')

    print(f'Average amplitude of {no_sims} normalised nonzero points in interferometer plane: {np.mean(np.abs(simulated), axis=0)} +/- {np.std(np.abs(simulated), axis=0) / np.sqrt(no_sims)}')
    print(f'Average phase of {no_sims} normalised nonzero points in interferometer plane: {np.mean(np.angle(simulated), axis=0)} +/- {np.std(np.angle(simulated), axis=0) / np.sqrt(no_sims)}')
    print(f'Average amplitude of {no_sims} normalised nonzero points in masked plane: {np.mean(np.abs(masked), axis=0)}  +/- {np.std(np.abs(masked), axis=0) / np.sqrt(no_sims)}')
    print(f'Average phase of {no_sims} normalised nonzero points in masked plane: {np.mean(np.angle(masked), axis=0)}  +/- {np.std(np.angle(masked), axis=0) / np.sqrt(no_sims)}')
    
def image_re_test_uv():
    # m = 10
    # locs = np.linspace(0, 2 * np.pi, m)
    # image = images.m_point_sources(int(1e6), m, [0.000 * np.sin(x) for x in locs], [0.0005 * np.cos(x) for x in locs], [1.2 for x in locs])

    offset = 0e-6
    image = images.point_source(int(1e6), 0.000, offset, 1.2)
    # image = images.double_point_source(int(1e6), [0.000, 0.000], [0.0005, -0.0005], [1.2, 1.2])

    test_I = instrument.interferometer(.1, 1, .5, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        .00000 * 2 * np.pi, roll_init=0/4 * np.pi)
    test_I.add_baseline(.035, 10, 300, 1200, 2, 1)
    # test_I.add_baseline(.105, 10, 300, 3700, 2, 1)
    # test_I.add_baseline(.315, 10, 300, 11100, 2, 1)
    # test_I.add_baseline(.945, 10, 300, 33400, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, 100000)
    print('Processing this image took ', time.time() - start, ' seconds')

    colourlist = ['b', 'orange', 'g', 'r']
    for i in range(len(test_I.baselines)):
        exp = test_I.baselines[i].F * np.cos(test_data.pointing[0, 2] - np.arctan2(image.loc[0, 0], (image.loc[0, 1] + 1e-20))) * np.sqrt(image.loc[0, 0]**2 + image.loc[0, 1]**2) * 1e6
        delta_y = np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
        # analysis.hist_data(test_data.actual_pos[:, 1][(test_data.pointing[test_data.discrete_t, 2] >= test_I.roll_init - .01 * np.pi) * (test_data.pointing[test_data.discrete_t, 2] < test_I.roll_init + .01 * np.pi) * (test_data.baseline_indices == i)], 
        #                     int(np.amax(test_data.discrete_pos[:, 1][test_data.baseline_indices == i]) - 
        #                     np.amin(test_data.discrete_pos[:, 1][test_data.baseline_indices == i])) + 1, False, i)
        print(int(np.amax(test_data.discrete_pos[:, 1][test_data.baseline_indices == i]) - np.amin(test_data.discrete_pos[:, 1][test_data.baseline_indices == i])) + 1)
        analysis.hist_data(test_data.pixel_to_pos(test_I)[:, 1][(test_data.pointing[test_data.discrete_t, 2] >= test_I.roll_init - .01 * np.pi) * (test_data.pointing[test_data.discrete_t, 2] < test_I.roll_init + .01 * np.pi) * (test_data.baseline_indices == i)], 
                            int(np.ceil(test_I.baselines[i].W / test_I.res_pos)) + 1, False, i)
        plt.vlines(exp, -100, 10000, color=colourlist[i])
        plt.vlines(exp + (delta_y * np.arange(-5, 5, 1))*1e6, -100, 10000, color=colourlist[i])
        plt.title(f'Photon impact positions on detector at roll of {test_I.roll_init / np.pi} pi rad')
        plt.ylim(0, 8000)
        plt.xlim(-200, 200)
        plt.legend()
        plt.show()

    # test_freq = np.fft.fftfreq(test_data.size)
    # plt.plot(test_freq, np.fft.fftshift(np.fft.fft(test_data.test_data)))
    # plt.show()

    for i in range(len(test_I.baselines)):
        samples = int(np.ceil(test_I.baselines[i].W / test_I.res_pos)) + 1
        binned_data, edges = np.histogram(test_data.actual_pos[:,1][test_data.baseline_indices == i], samples)
        centres = edges[:-1] + (edges[1:] - edges[:-1])/2
        print(centres)
        ft_x_data, ft_y_data = analysis.ft_data(binned_data, samples, edges[1] - edges[0])

        fig, (ax1, ax2) = plt.subplots(2,1)
        delta_u = 1 / np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
        exact = (np.sum(binned_data * np.exp(-2j * np.pi * -delta_u * centres)) / binned_data.size, np.sum(binned_data * np.exp(-2j * np.pi * delta_u * centres)) / binned_data.size)
        fig.suptitle(f'Exact values for amplitude: {np.round(np.abs(exact), 5)}, for phase: {np.round(np.angle(exact), 5)} at offset {np.round(offset, 6)} \"')

        print(exact, np.abs(exact), np.angle(exact))

        analysis.plot_ft(ft_x_data, np.abs(ft_y_data), ax1, 0, i)
        ax1.axvline(delta_u, 1e-5, 1e4, color = 'k', label='Expected frequency')
        ax1.axvline(-delta_u, 1e-5, 1e4, color = 'k')
        ax1.plot([-delta_u, delta_u], np.abs(exact), 'ro', label='Exact value')
        ax1.set_xlim(-4 * delta_u, 4 * delta_u)
        ax1.set_title(f'Fourier transform of photon positions at roll of {test_I.roll_init / (np.pi)} pi rad')
        ax1.set_xlabel('Spatial frequency ($m^{-1}$)')
        ax1.set_ylabel('Fourier magnitude')
        ax1.legend()

        analysis.plot_ft(ft_x_data, np.angle(ft_y_data), ax2, 0, i)
        ax2.axvline(delta_u, 1e-5, 1e4, color = 'k', label='Expected frequency')
        ax2.axvline(-delta_u, 1e-5, 1e4, color = 'k')
        ax2.plot([-delta_u, delta_u], np.angle(exact), 'ro', label='Exact value')
        ax2.set_xlim(-4 * delta_u, 4 * delta_u)
        ax2.set_title(f'Fourier transform of photon positions at roll of {test_I.roll_init / (np.pi)} pi rad')
        ax2.set_xlabel('Spatial frequency ($m^{-1}$)')
        ax2.set_ylabel('Fourier phase')
        ax2.legend()

    # plt.show()

    start = time.time()
    test_data_imre = np.zeros((512, 512))
    test_data_imre[256, 256] = 1
    # test_data_imre[12, 12] = 1
    test_data_imre_fft = ft.fft2(test_data_imre)
    re_im, f_grid, test_data_masked = analysis.image_recon_smooth(test_data, test_I, .02 * 2 * np.pi, samples=test_data_imre.shape, test_data=test_data_imre_fft, exvfast=0)
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    test_image = ft.ifft2(test_data_masked)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    ax1.imshow(np.abs(ft.fftshift(f_grid)), cmap=cm.Reds)
    ax1.set_title('UV-plane (amplitude)')
    ax2.imshow(np.angle(ft.fftshift(f_grid)), cmap=cm.Blues)
    ax2.set_title('UV-plane (phase)')
    ax3.imshow(np.abs(ft.fftshift(re_im)), cmap=cm.Greens)
    ax3.set_title('Reconstructed image')
    # ax3.plot(256, 256, 'r.')
    ax4.imshow(np.abs(ft.fftshift(test_data_masked)), cmap=cm.Blues)
    ax4.set_title('UV-plane (amplitude)')
    ax5.imshow(np.angle(ft.fftshift(test_data_masked)), cmap=cm.Greens)
    ax5.set_title('UV-plane (phase)')
    ax6.imshow(np.abs(test_image), cmap=cm.Greens)
    ax6.set_title('Reconstructed image')

    ax7.imshow(np.abs(ft.fftshift(test_data_imre_fft)), cmap=cm.Blues)
    ax7.set_title('Full UV-plane (amplitude)')
    # ax7.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax7.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax8.imshow(np.angle(ft.fftshift(test_data_imre_fft)), cmap=cm.Greens)
    ax8.set_title('Full UV-plane (phase)')
    # ax8.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax8.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax9.imshow(np.abs(ft.ifft2(test_data_imre_fft)), cmap=cm.Greens)
    ax9.set_title('Image')
    plt.show()

    print(f'Normalised nonzero points in interferometer plane: {f_grid[f_grid.nonzero()] / np.sum(np.abs(f_grid[f_grid.nonzero()]))}')
    print(f'Normalised nonzero points in masked plane: {test_data_masked[f_grid.nonzero()] / np.sum(np.abs(test_data_masked[f_grid.nonzero()]))}')

def image_re_test_parts():
    # m = 10
    # locs = np.linspace(0, 2 * np.pi, m)
    # image = images.m_point_sources(int(1e6), m, [0.000 * np.sin(x) for x in locs], [0.0005 * np.cos(x) for x in locs], [1.2 for x in locs])

    #TODO look at digitize function

    #TODO look at testing with sinusoid instead of point source

    offset = 0e-6
    image = images.point_source(int(1e6), 0.000, offset, 1.2)
    # image = images.double_point_source(int(1e6), [0.000, 0.000], [0.0005, -0.0005], [1.2, 1.2])

    test_I = instrument.interferometer(.1, 1, .5, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        .00000 * 2 * np.pi, roll_init=0/4 * np.pi)
    # test_I.add_baseline(.035, 10, 300, 1200, 2, 1)
    # test_I.add_baseline(.105, 10, 300, 3700, 2, 1)
    # test_I.add_baseline(.315, 10, 300, 11100, 2, 1)
    test_I.add_baseline(.945, 10, 300, 33400, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, 100000)
    print('Processing this image took ', time.time() - start, ' seconds')

    colourlist = ['b', 'orange', 'g', 'r']
    for i in range(len(test_I.baselines)):
        exp = test_I.baselines[i].F * np.cos(test_data.pointing[0, 2] - np.arctan2(image.loc[0, 0], (image.loc[0, 1] + 1e-20))) * np.sqrt(image.loc[0, 0]**2 + image.loc[0, 1]**2) * 1e6
        delta_y = np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
        analysis.hist_data(test_data.actual_pos[:, 1][(test_data.pointing[test_data.discrete_t, 2] >= test_I.roll_init - .01 * np.pi) * (test_data.pointing[test_data.discrete_t, 2] < test_I.roll_init + .01 * np.pi) * (test_data.baseline_indices == i)], 
                            int(np.ceil(test_I.baselines[i].W / test_I.res_pos)), False, i)
        plt.vlines(exp, -100, 10000, color=colourlist[i])
        plt.vlines(exp + (delta_y * np.arange(-5, 5, 1))*1e6, -100, 10000, color=colourlist[i])
        plt.title(f'Photon impact positions on detector at roll of {test_I.roll_init / np.pi} pi rad')
        plt.ylim(0, 8000)
        plt.xlim(-200, 200)
        plt.legend()
        plt.show()

    # test_freq = np.fft.fftfreq(test_data.size)
    # plt.plot(test_freq, np.fft.fftshift(np.fft.fft(test_data.test_data)))
    # plt.show()

    for i in range(len(test_I.baselines)):
        samples = int(np.ceil(test_I.baselines[i].W / test_I.res_pos)) + 1
        binned_data, edges = np.histogram(test_data.actual_pos[:,1][test_data.baseline_indices == i], samples)
        centres = edges[:-1] + (edges[1:] - edges[:-1])/2
        ft_x_data, ft_y_data = analysis.ft_data(binned_data, samples, edges[1] - edges[0])

        fig, (ax1, ax2) = plt.subplots(2,1)
        delta_u = 1 / np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
        exact = (np.sum(binned_data * np.exp(-2j * np.pi * -delta_u * centres)) / binned_data.size, np.sum(binned_data * np.exp(-2j * np.pi * delta_u * centres)) / binned_data.size)
        fig.suptitle(f'Exact values for amplitude: {np.round(np.abs(exact), 5)}, for phase: {np.round(np.angle(exact), 5)} at offset {np.round(offset, 6)} \"')

        print(exact, np.abs(exact), np.angle(exact))

        analysis.plot_ft(ft_x_data, np.abs(ft_y_data), ax1, 0, i)
        ax1.axvline(delta_u, 1e-5, 1e4, color = 'k', label='Expected frequency')
        ax1.axvline(-delta_u, 1e-5, 1e4, color = 'k')
        ax1.plot([-delta_u, delta_u], np.abs(exact), 'ro', label='Exact value')
        ax1.set_xlim(-4 * delta_u, 4 * delta_u)
        ax1.set_title(f'Fourier transform of photon positions at roll of {test_I.roll_init / (np.pi)} pi rad')
        ax1.set_xlabel('Spatial frequency ($m^{-1}$)')
        ax1.set_ylabel('Fourier magnitude')
        ax1.legend()

        analysis.plot_ft(ft_x_data, np.angle(ft_y_data), ax2, 0, i)
        ax2.axvline(delta_u, 1e-5, 1e4, color = 'k', label='Expected frequency')
        ax2.axvline(-delta_u, 1e-5, 1e4, color = 'k')
        ax2.plot([-delta_u, delta_u], np.angle(exact), 'ro', label='Exact value')
        ax2.set_xlim(-4 * delta_u, 4 * delta_u)
        ax2.set_title(f'Fourier transform of photon positions at roll of {test_I.roll_init / (np.pi)} pi rad')
        ax2.set_xlabel('Spatial frequency ($m^{-1}$)')
        ax2.set_ylabel('Fourier phase')
        ax2.legend()

    plt.show()

    # start = time.time()
    # test_data_imre = np.zeros((512, 512))
    # test_data_imre[256, 256] = 1
    # test_data_imre = ft.fft2(test_data_imre)
    # re_im, f_grid, test_data_masked = analysis.image_recon_smooth(test_data, test_I, .02 * 2 * np.pi, samples=[512,512], test_data=test_data_imre, exvfast=0)
    # print('Reconstructing this image took ', time.time() - start, ' seconds')

    # test_image = ft.ifft2(test_data_masked)

    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    # ax1.imshow(np.abs(ft.fftshift(f_grid)), cmap=cm.Reds)
    # ax1.set_title('UV-plane (amplitude)')
    # ax2.imshow(np.angle(ft.fftshift(f_grid)), cmap=cm.Blues)
    # ax2.set_title('UV-plane (phase)')
    # ax3.imshow(np.abs(ft.fftshift(re_im)), cmap=cm.Greens)
    # ax3.set_title('Reconstructed image')
    # # ax3.plot(256, 256, 'r.')
    # ax4.imshow(np.abs(ft.fftshift(test_data_masked)), cmap=cm.Blues)
    # ax4.set_title('UV-plane (amplitude)')
    # ax5.imshow(np.angle(ft.fftshift(test_data_masked)), cmap=cm.Greens)
    # ax5.set_title('UV-plane (phase)')
    # ax6.imshow(np.abs(test_image), cmap=cm.Greens)
    # ax6.set_title('Reconstructed image')
    # plt.show()

def image_re_test_point():
    image = images.point_source(int(1e6), 0.0005, 0.000, 1.2)

    test_I = instrument.interferometer(.1, 1, .5, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        .015 * 2 * np.pi)
    test_I.add_baseline(.035, 10, 300, 1200, 2, 1)
    test_I.add_baseline(.105, 10, 300, 3700, 2, 1)
    test_I.add_baseline(.315, 10, 300, 11100, 2, 1)
    test_I.add_baseline(.945, 10, 300, 33400, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, 512)
    print('Processing this image took ', time.time() - start, ' seconds')

    # plt.plot(test_data.test_data)
    # plt.show()

    # colourlist = ['b', 'orange', 'g', 'r']
    # for i in range(len(test_I.baselines)):
    #     exp = test_I.baselines[i].F * np.cos(-np.arctan2(image.loc[0, 0], (image.loc[0, 1] + 1e-20))) * np.sqrt(image.loc[0, 0]**2 + image.loc[0, 1]**2) * 1e6
    #     delta_y = np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
    #     analysis.hist_data(test_data.actual_pos[:, 1][(test_data.pointing[test_data.discrete_t, 2] > 0) * (test_data.pointing[test_data.discrete_t, 2] < .1 * np.pi) * (test_data.baseline_indices == i)], 
    #                         int(np.amax(test_data.discrete_pos[:, 1][test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[:, 1][test_data.baseline_indices == i])) + 1, False, i)
    #     plt.vlines(exp, -100, 10000, color=colourlist[i])
    #     plt.vlines(exp + (delta_y * np.arange(-5, 5, 1))*1e6, -100, 10000, color=colourlist[i])
    #     # print(test_I.baselines[i].F / test_I.baselines[i].D)
    # # print(exp /33400 * 1e-6)
    # plt.title('Photon impact positions on detector')
    # plt.ylim(0, 200)
    # plt.xlim(-200, 200)
    # plt.legend()
    # plt.show()

    # test_freq = np.fft.fftfreq(test_data.size)
    # plt.plot(test_freq, np.fft.fftshift(np.fft.fft(test_data.test_data)))
    # plt.show()

    # for i in range(len(test_I.baselines)):
    #     samples = int(np.amax(test_data.discrete_pos[:, 1][test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[:, 1][test_data.baseline_indices == i])) + 1
    #     binned_data, edges = np.histogram(test_data.actual_pos[:,1][test_data.baseline_indices == i], samples)
    #     ft_x_data, ft_y_data = analysis.ft_data(binned_data, samples, edges[1] - edges[0])
    #     analysis.plot_ft(ft_x_data, ft_y_data, 0, i)
    #     delta_u = 1 / np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
    #     # delta_guess = 10**6 / 30.6
    #     plt.axvline(delta_u, 1e-5, 1e4, color = colourlist[i])
    #     plt.axvline(-delta_u, 1e-5, 1e4, color = colourlist[i])
    #     # plt.axvline(delta_guess, 1e-5, 1e4, color = 'r')
    #     # plt.axvline(-delta_guess, 1e-5, 1e4, color = 'r')
    # plt.xlim(-4 * delta_u, 4 * delta_u)
    # plt.title('Fourier transform of photon positions')
    # plt.xlabel('Spatial frequency ($m^{-1}$)')
    # plt.ylabel('Fourier magnitude')
    # plt.legend()
    # plt.show()

    start = time.time()
    test_data_imre = np.zeros((512, 512))
    test_data_imre[264, 256] = 1
    # test_data_imre[261, 261] = 1
    # test_data_imre[12, 12] = 1
    test_data_imre = ft.fft2(test_data_imre)
    re_im, f_grid, test_data_masked = analysis.image_recon_smooth(test_data, test_I, .02 * 2 * np.pi, samples=[512, 512], test_data=test_data_imre)
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    test_image = ft.ifft2(test_data_masked)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(abs(ft.fftshift(f_grid)), cmap=cm.Reds)
    ax1.set_title('UV-plane (amplitude)')
    ax2.imshow(np.imag(ft.fftshift(f_grid)), cmap=cm.Blues)
    ax2.set_title('UV-plane (phase)')
    ax3.imshow(abs(ft.fftshift(re_im)), cmap=cm.Greens)
    ax3.set_title('Reconstructed image')
    # ax3.plot(256, 256, 'r.')
    ax4.imshow(abs(ft.fftshift(test_data_masked)), cmap=cm.Blues)
    ax4.set_title('UV-plane (amplitude)')
    ax5.imshow(np.imag(ft.fftshift(test_data_masked)), cmap=cm.Greens)
    ax5.set_title('UV-plane (phase)')
    ax6.imshow(abs(test_image), cmap=cm.Greens)
    ax6.set_title('Reconstructed image')
    plt.show()

def sinety_test():
    f = 4.387
    phase = np.pi / 4.5
    xend = 3.17
    samples = 1000

    x = np.linspace(0, xend, samples)
    y = np.cos(f * 2 * np.pi * x + phase)
    four_x, fast_y = analysis.ft_data(y, samples, xend/samples)
    four_x = np.fft.fftshift(four_x)
    
    exact_y = np.zeros(four_x.size, dtype=np.complex_)
    for i, freq in enumerate(four_x):
        exact_y[i] = np.sum(y * np.exp(-2j * np.pi * freq * x)) / y.size

    fast_y = np.fft.fftshift(fast_y)
    # y_data, edges = np.histogram(data, samples)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(x, y)
    ax1.set_ylim(-3, 3)
    ax1.set_title(f'Cosine with frequency pi and phase {phase / np.pi} pi')

    ax2.plot(four_x, np.abs(fast_y), '--', label='Fast')
    ax2.plot(four_x, np.abs(exact_y), '-.', label='Exact')
    ax2.plot([-f, f], [np.abs(np.sum(y * np.exp(-2j * np.pi * -f * x)) / y.size), np.abs(np.sum(y * np.exp(-2j * np.pi * f * x)) / y.size)], 'ro', label='Extra exact')

    # ax2.vlines([-5, 5], [-2, -2], [2, 2], colors='r')
    ax2.set_ylim(-1, 1)
    ax2.set_xlim(-6, 6)
    ax2.set_title('Amplitude of fourier transform')
    ax2.legend()

    ax3.plot(four_x, np.angle(fast_y), '--', label='Fast')
    ax3.plot(four_x, np.angle(exact_y), '-.', label='Exact')
    ax3.plot(four_x, np.angle(exact_y), '.', label='Exact')
    ax3.plot(four_x, [phase for i in four_x], 'r:', label='Phase')
    ax3.plot(four_x, [-phase for i in four_x], 'r:')
    ax3.plot([-f, f], [np.angle(np.sum(y * np.exp(-2j * np.pi * -f * x)) / y.size), np.angle(np.sum(y * np.exp(-2j * np.pi * f * x)) / y.size)], 'ro', label='Extra exact')
    ax3.set_ylim(-np.pi, np.pi)
    ax3.set_xlim(-6, 6)
    ax3.set_title('Phase of fourier transform')
    ax3.legend()
    plt.show()


def sinetier_test():
    f = 4.387
    phase = np.pi / 4.5
    xend = 3.17
    samples = 1000

    x = np.linspace(0, xend, samples)
    y = np.cos(f * 2 * np.pi * x + phase)
    four_x, fast_y = analysis.ft_data(y, samples, xend/samples)
    four_x = np.fft.fftshift(four_x)
    
    exact_y = np.zeros(four_x.size, dtype=np.complex_)
    for i, freq in enumerate(four_x):
        exact_y[i] = np.sum(y * np.exp(-2j * np.pi * freq * x)) / y.size

    fast_y = np.fft.fftshift(fast_y)
    # y_data, edges = np.histogram(data, samples)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(x, y)
    ax1.set_ylim(-3, 3)
    ax1.set_title(f'Cosine with frequency pi and phase {phase / np.pi} pi')

    ax2.plot(four_x, np.abs(fast_y), '--', label='Fast')
    ax2.plot(four_x, np.abs(exact_y), '-.', label='Exact')
    ax2.plot([-f, f], [np.abs(np.sum(y * np.exp(-2j * np.pi * -f * x)) / y.size), np.abs(np.sum(y * np.exp(-2j * np.pi * f * x)) / y.size)], 'ro', label='Extra exact')

    # ax2.vlines([-5, 5], [-2, -2], [2, 2], colors='r')
    ax2.set_ylim(-1, 1)
    ax2.set_xlim(-6, 6)
    ax2.set_title('Amplitude of fourier transform')
    ax2.legend()

    ax3.plot(x, np.fft.ifft(np.fft.fftshift(fast_y)), '--', label='Fast')
    ax3.plot(x, np.fft.ifft(np.fft.fftshift(exact_y)), '-.', label='Exact')
    ax3.set_title('Reconstructed cosine')
    ax3.legend()

    ax4.plot(four_x, np.angle(fast_y), '--', label='Fast')
    ax4.plot(four_x, np.angle(exact_y), '-.', label='Exact')
    ax4.plot(four_x, np.angle(exact_y), '.', label='Exact')
    ax4.plot(four_x, [phase for i in four_x], 'r:', label='Phase')
    ax4.plot(four_x, [-phase for i in four_x], 'r:')
    ax4.plot([-f, f], [np.angle(np.sum(y * np.exp(-2j * np.pi * -f * x)) / y.size), np.angle(np.sum(y * np.exp(-2j * np.pi * f * x)) / y.size)], 'ro', label='Extra exact')
    ax4.set_ylim(-np.pi, np.pi)
    ax4.set_xlim(-6, 6)
    ax4.set_title('Phase of fourier transform')
    ax4.legend()
    plt.show()

def image_re_test_multiple():
    # image = images.double_point_source(int(1e6), [0.0002, -0.0002], [0.0002, -0.0002], [1.2, 1.2])
    image1 = images.point_source(int(1e5), 0.0002, 0.0002, 1.2)
    image2 = images.point_source(int(1e5), -0.0002, -0.0002, 1.2)

    test_I = instrument.interferometer(.1, .01, .5, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        .0005 * 2 * np.pi)
    # test_I.add_baseline(.035, 10, 300, 1200, 2, 1)
    # test_I.add_baseline(.105, 10, 300, 3700, 2, 1)
    # test_I.add_baseline(.315, 10, 300, 11100, 2, 1)
    test_I.add_baseline(.945, 10, 300, 33400, 2, 1)

    start = time.time()
    test_data1 = process.interferometer_data(test_I, image1, 10, 512)
    print('Processing this first image took ', time.time() - start, ' seconds')

    start = time.time()
    test_data2 = process.interferometer_data(test_I, image2, 10, 512)
    print('Processing this second image took ', time.time() - start, ' seconds')

    # plt.plot(test_data.test_data)
    # plt.show()

    # exp = test_I.baselines[0].F * np.cos(-np.arctan(image.loc[0, 0] / (image.loc[0, 1] + 1e-20))) * np.sqrt(image.loc[0, 0]**2 + image.loc[0, 1]**2) * 1e6
    # for i in range(len(test_I.baselines)):
    #     analysis.hist_data(test_data.actual_pos[:, 1][test_data.baseline_indices == i], 
    #                         int(np.amax(test_data.discrete_pos[:, 1][test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[:, 1][test_data.baseline_indices == i])) + 1, False, i)
    #     # print(test_I.baselines[i].F / test_I.baselines[i].D)
    # # print(exp /33400 * 1e-6)
    # # plt.vlines(-exp, -100, 10000)
    # plt.title('Photon impact positions on detector')
    # # plt.ylim(0, 500)
    # plt.legend()
    # plt.show()

    # test_freq = np.fft.fftfreq(test_data.size)
    # plt.plot(test_freq, np.fft.fftshift(np.fft.fft(test_data.test_data)))
    # plt.show()

    # colourlist = ['b', 'orange', 'g', 'r']
    # for i in range(len(test_I.baselines)):
    #     samples = int(np.amax(test_data.discrete_pos[:, 1][test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[:, 1][test_data.baseline_indices == i])) + 1
    #     binned_data, edges = np.histogram(test_data.actual_pos[:,1][test_data.baseline_indices == i], samples)
    #     ft_x_data, ft_y_data = analysis.ft_data(binned_data, samples, edges[1] - edges[0])
    #     analysis.plot_ft(ft_x_data, ft_y_data, 0, i)
    #     delta_u = 1 / np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
    #     # delta_guess = 10**6 / 30.6
    #     plt.axvline(delta_u, 1e-5, 1e4, color = colourlist[i])
    #     plt.axvline(-delta_u, 1e-5, 1e4, color = colourlist[i])
    #     # plt.axvline(delta_guess, 1e-5, 1e4, color = 'r')
    #     # plt.axvline(-delta_guess, 1e-5, 1e4, color = 'r')
    # plt.xlim(-4 * delta_u, 4 * delta_u)
    # plt.title('Fourier transform of photon positions')
    # plt.xlabel('Spatial frequency ($m^{-1}$)')
    # plt.ylabel('Fourier magnitude')
    # plt.legend()
    # plt.show()

    re_im1, f_grid1, test_data_masked = analysis.image_recon_smooth(test_data1, test_I, .01 * 2 * np.pi, samples=512)
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    re_im2, f_grid2, test_data_masked = analysis.image_recon_smooth(test_data1, test_I, .01 * 2 * np.pi, samples=512)
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(abs(ft.fftshift(f_grid1)), cmap=cm.Reds)
    ax1.set_title('UV-plane (amplitude)')
    ax2.imshow(np.imag(ft.fftshift(f_grid1)), cmap=cm.Blues)
    ax2.set_title('UV-plane (phase)')
    ax3.imshow(abs(ft.fftshift(re_im1)), cmap=cm.Greens)
    ax3.set_title('Reconstructed image')
    ax4.imshow(abs(ft.fftshift(f_grid2)), cmap=cm.Reds)
    ax4.set_title('UV-plane (amplitude)')
    ax5.imshow(np.imag(ft.fftshift(f_grid2)), cmap=cm.Blues)
    ax5.set_title('UV-plane (phase)')
    ax6.imshow(abs(ft.fftshift(re_im2)), cmap=cm.Greens)
    ax6.set_title('Reconstructed image')

    plt.show()

def full_image_test(test_code):
    image_path = r"C:\Users\nielz\Documents\Uni\Master\Thesis\Simulator\vri\models\galaxy_lobes.png"
    # img_scale = 2.2 * .75 * 6.957 * 1e8 / (9.714 * spc.parsec)
    img_scale = .025
    image, pix_scale = images.generate_from_image(image_path, int(1e6), img_scale)
    # image, pix_scale = images.generate_from_image(image_path, int(1e6), img_scale)

    #TODO find total FOV of instrument

    histedimage, _, __ = np.histogram2d(image.loc[:,0], image.loc[:,1], np.array([np.linspace(-img_scale/2, img_scale/2, pix_scale[0]), 
                                                                                  np.linspace(-img_scale/2, img_scale/2, pix_scale[1])]) * 2 * np.pi / (3600 * 360)) 
    plt.imshow(histedimage, cmap=cm.Greens)
    plt.xlabel('x-axis angular offset from optical axis (arcsec)')
    plt.ylabel('y-axis angular offset from optical axis (arcsec)')
    plt.show()

    # image = images.point_source(int(1e5), 0.000, 0.0005, 1.2)

    test_I = instrument.interferometer(.1, 1, .5, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        0.003 * 2 * np.pi, roll_init=0.)
    test_I.add_baseline(.035, 10, 300, 1200, 2, 1)
    test_I.add_baseline(.05, 10, 300, 1800, 2, 1)
    test_I.add_baseline(.001, 10, 300, 20, 2, 1)
    test_I.add_baseline(.010, 10, 300, 400, 2, 1)
    test_I.add_baseline(.020, 10, 300, 800, 2, 1)
    test_I.add_baseline(.005, 10, 300, 100, 2, 1)
    test_I.add_baseline(.5, 10, 300, 18000, 2, 1)
    test_I.add_baseline(.75, 10, 300, 26000, 2, 1)
    test_I.add_baseline(.105, 10, 300, 3700, 2, 1)
    test_I.add_baseline(.315, 10, 300, 11100, 2, 1)
    test_I.add_baseline(.945, 10, 300, 33400, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, 100000)
    print('Processing this image took ', time.time() - start, ' seconds')

    # exp = test_I.baselines[0].F * np.cos(-np.arctan(image.loc[0, 0] / (image.loc[0, 1] + 1e-20))) * np.sqrt(image.loc[0, 0]**2 + image.loc[0, 1]**2) * 1e6
    # for i in range(len(test_I.baselines)):
    #     analysis.hist_data(test_data.actual_pos[:, 1][test_data.baseline_indices == i], 
    #                         int(np.amax(test_data.discrete_pos[:, 1][test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[:, 1][test_data.baseline_indices == i])) + 1, False, i)
    #     # print(test_I.baselines[i].F / test_I.baselines[i].D)
    # # print(exp /33400 * 1e-6)
    # # plt.vlines(-exp, -100, 10000)
    # plt.title('Photon impact positions on detector')
    # # plt.ylim(0, 500)
    # plt.legend()
    # plt.show()

    # test_freq = np.fft.fftfreq(test_data.size)
    # plt.plot(test_freq, np.fft.fftshift(np.fft.fft(test_data.test_data)))
    # plt.show()

    # colourlist = ['b', 'orange', 'g', 'r']
    # for i in range(len(test_I.baselines)):
    #     samples = int(np.amax(test_data.discrete_pos[:, 1][test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[:, 1][test_data.baseline_indices == i])) + 1
    #     binned_data, edges = np.histogram(test_data.actual_pos[:,1][test_data.baseline_indices == i], samples)
    #     ft_x_data, ft_y_data = analysis.ft_data(binned_data, samples, edges[1] - edges[0])
    #     analysis.plot_ft(ft_x_data, ft_y_data, 0, i)
    #     delta_u = 1 / np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
    #     # delta_guess = 10**6 / 30.6
    #     plt.axvline(delta_u, 1e-5, 1e4, color = colourlist[i])
    #     plt.axvline(-delta_u, 1e-5, 1e4, color = colourlist[i])
    #     # plt.axvline(delta_guess, 1e-5, 1e4, color = 'r')
    #     # plt.axvline(-delta_guess, 1e-5, 1e4, color = 'r')
    # plt.xlim(-4 * delta_u, 4 * delta_u)
    # plt.title('Fourier transform of photon positions')
    # plt.xlabel('Spatial frequency ($m^{-1}$)')
    # plt.ylabel('Fourier magnitude')
    # plt.legend()
    # plt.show()

    start = time.time()
    test_data_imre = np.array(Image.open(image_path).convert('L'))
    # plt.imshow(test_data_imre)
    # plt.show()
    shap = np.array(test_data_imre.shape)
    # test_data_imre[np.array(test_data_imre.shape)//2] = 1
    # test_data_imre[261, 261] = 1
    # test_data_imre[12, 12] = 1
    test_data_imre_fft = ft.fft2(test_data_imre)
    re_im, f_grid, test_data_masked = analysis.image_recon_smooth(test_data, test_I, .01 * 2 * np.pi, 
                                                                  samples=np.array(test_data_imre.shape), test_data=test_data_imre_fft)
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    test_image = ft.ifft2(test_data_masked)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    ax1.imshow(np.abs(ft.fftshift(f_grid)), cmap=cm.Reds)
    ax1.set_title('UV-plane (amplitude)')
    # ax1.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax1.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax2.imshow(np.angle(ft.fftshift(f_grid)), cmap=cm.Blues)
    ax2.set_title('UV-plane (phase)')
    # ax2.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax2.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax3.imshow(np.abs(ft.fftshift(re_im)), cmap=cm.Greens)
    ax3.set_title('Reconstructed image')

    ax4.imshow(np.abs(ft.fftshift(test_data_masked)), cmap=cm.Blues)
    ax4.set_title('UV-plane (amplitude)')
    # ax4.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax4.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax5.imshow(np.angle(ft.fftshift(test_data_masked)), cmap=cm.Greens)
    ax5.set_title('UV-plane (phase)')
    # ax5.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax5.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax6.imshow(np.abs(test_image), cmap=cm.Greens)
    ax6.set_title('Ifft of masked image')

    ax7.imshow(np.abs(ft.fftshift(test_data_imre_fft)), cmap=cm.Blues)
    ax7.set_title('Full UV-plane (amplitude)')
    # ax7.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax7.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax8.imshow(np.angle(ft.fftshift(test_data_imre_fft)), cmap=cm.Greens)
    ax8.set_title('Full UV-plane (phase)')
    # ax8.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax8.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax9.imshow(np.abs(ft.ifft2(test_data_imre_fft)), cmap=cm.Greens)
    ax9.set_title('Image')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Look at different normalisations`
    rel_grid = test_data_masked / np.sum(test_data_masked) - f_grid / np.sum(f_grid)

    ax1.imshow(np.abs(ft.fftshift(re_im)), cmap=cm.Reds)
    ax1.set_title('Recreated image')
    # ax1.set_xlim(shap[0] // 2 - 25, shap[0] // 2 + 25)
    # ax1.set_ylim(shap[1] // 2 - 25, shap[1] // 2 + 25)

    ax2.imshow(np.abs(test_image), cmap=cm.Greens)
    ax2.set_title('theoretical image')
    # ax2.set_xlim(shap[0] // 2 - 25, shap[0] // 2 + 25)
    # ax2.set_ylim(shap[1] // 2 - 25, shap[1] // 2 + 25)
    # ax3.imshow(np.abs(ft.ifft2(rel_grid)))
    # ax3.set_title('Image difference')

    plt.show()
    # print('Relative non-zero data points in test data: \n', np.abs(test_data_masked[test_data_masked.nonzero()] / np.amax(test_data_masked[test_data_masked.nonzero()])).astype(float))
    # print('Relative non-zero data points in interferometer data: \n', np.abs(f_grid[f_grid.nonzero()] / np.amax(f_grid[f_grid.nonzero()])).astype(float))

if __name__ == "__main__":
    #TODO fix times of arrival
    # willingale_test()
    # image_re_test()
    # image_re_test_multiple()
    # w_ps_test()
    # Fre_test()
    # scale_test2()
    # discretize_test()
    # sinety_test()
    # sinetier_test()
    # full_image_test(0)
    # image_re_test_point()
    # image_re_test_parts()
    # image_re_test_uv()
    full_image_test(0)
    # stats_test()
    pass