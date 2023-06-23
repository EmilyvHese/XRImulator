"""
This file contains functions and classes related to the generation images in the form of a number of X-ray photons, 
each with location and time of arrival and energy. 
"""

from PIL import Image
import numpy as np
import scipy.constants as spc


class image():
    """ 
    Class that defines a data format that all images to be processed by this package should follow. 
    It consists of a number of arrrays of specified size which should contain the energy, arrival time, and ... #TODO define further
    Note that this class only generates an empty image class of specified size.
    Generating the actual photons to fill it up should happen in a seperate function that manipulates an image class object. 
    """

    def __init__(self, size):
        """ Initiation function for the class. Generates arrays of the specified size for each parameter specified in the class docstring. """
        # Abbreviation of 'Times Of Arrival'.
        self.toa = np.zeros(size, dtype=int)
        self.energies = np.zeros(size)

        # Array containing coordinates of origin for each photon.
        self.loc = np.zeros((size, 2))

        # Useful to have as shorthand
        self.size = size

def point_source(size, alpha, beta, energy):
    """
    Function that generates an image of a monochromatic point source according to some specifications.

    Parameters:

    size (int) = number of photons to generate from this source.\n
    alpha (float) = coordinate offset from zero pointing in x-direction (arcsec)\n
    beta (float) = coordinate offset from zero pointing in y-direction (arcsec)\n
    energy (float) = energy of photons to generate (KeV)\n
    """
    im = image(size)
    im.energies[:] = energy * spc.eV * 1e3
    im.loc[:] = np.array([alpha, beta]) * 2 * np.pi / (3600 * 360)
    im.toa = np.array([i for i in range(size)])

    return im

def double_point_source(size, alpha, beta, energy):
    """
    Function that generates an image of two monochromatic point sources according to some specifications.

    Parameters:

    size (int) = number of photons to generate from the sources.\n
    alpha (list-like of floats) = coordinate offsets from zero pointing in x-direction (arcsec)\n
    beta (list-like of floats) = coordinate offsets from zero pointing in y-direction (arcsec)\n
    energy (list-like of floats) = energies of photons to generate (KeV)\n
    """
    im = image(size)
    for i in range(0, size):
        source = np.random.randint(0,2)
        im.energies[i] = energy[source] * spc.eV * 1e3
        im.loc[i] = np.array([alpha[source], beta[source]]) * 2 * np.pi / (3600 * 360)
        im.toa[i] = i

    return im

def m_point_sources(size, m, alpha, beta, energy):
    """
    Function that generates an image of two monochromatic point sources according to some specifications.

    Parameters:

    size (int) = number of photons to generate from the sources.\n
    m (int) = number of point sources to generate.\n
    alpha (list-like of floats) = coordinate offsets from zero pointing in x-direction (arcsec)\n
    beta (list-like of floats) = coordinate offsets from zero pointing in y-direction (arcsec)\n
    energy (list-like of floats) = energies of photons to generate (KeV)\n
    """
    im = image(size)
    for i in range(0, size):
        source = np.random.randint(0,m)
        im.energies[i] = energy[source] * spc.eV * 1e3
        im.loc[i] = np.array([alpha[source], beta[source]]) * 2 * np.pi / (3600 * 360)
        im.toa[i] = i

    return im

def point_source_multichromatic_range(size, alpha, beta, energy):
    """
    Function that generates an image of a multichromatic point source according to some specifications.

    Parameters:

    size (int) = number of photons to generate from this source.\n
    alpha (float) = coordinate offset from zero pointing in x-direction (arcsec)\n
    beta (float) = coordinate offset from zero pointing in y-direction (arcsec)\n
    energy (list-like of floats) = upper and lower bounds for energy of photons to generate (KeV)\n
    """
    im = image(size)
    for i in range(0, size):
        im.energies[i] = (np.random.random() * (energy[1] - energy[0]) + energy[0]) * spc.eV * 1e3
        im.loc[i] = np.array([alpha, beta]) * 2 * np.pi / (3600 * 360)
        im.toa[i] = i

    return im

def point_source_multichromatic_gauss(size, alpha, beta, energy, energy_spread):
    """
    Function that generates an image of a multichromatic point source according to some specifications.

    Parameters:

    size (int) = number of photons to generate from this source.\n
    alpha (float) = coordinate offset from zero pointing in x-direction (arcsec)\n
    beta (float) = coordinate offset from zero pointing in y-direction (arcsec)\n
    energy (float) = mean energy of photons to generate (KeV)\n
    energy_spread (float) = spread in energy of photons to generate (KeV)\n
    """
    im = image(size)
    im.energies = np.random.normal(energy, energy_spread, size) * spc.eV * 1e3
    for i in range(0, size):
        im.loc[i] = np.array([alpha, beta]) * 2 * np.pi / (3600 * 360)
        im.toa[i] = i

    return im

def disc(size, alpha, beta, energy, radius, energy_spread=0.):
    """
    A function that generates photons in the shape of a continuous disk.
    """
    im = image(size)
    for i in range(0, size):
        im.energies[i] = energy * spc.eV * 1e3
        im.toa[i] = i
        r = (np.random.random() * 2 - 1) * radius
        theta = np.random.random() * 2 * np.pi
        im.loc[i] = np.array([alpha + r * np.cos(theta), beta + r * np.sin(theta)]) * 2 * np.pi / (3600 * 360) 

    if energy_spread > 0.:
        im.energies += np.random.normal(0, energy_spread, size)

    return im

def generate_from_image(image_path, no_photons, img_scale, energy, energy_spread=0., offset=[0,0]):
    """
    Function that generates an image object from any arbitrary input image. 
    useful for testing realistic astrophysical sources without having to include code to simulate them here.
    Just have some other simulator generate an image and use this function to read that out.
    This function uses relative brightness of each part of the input image to generate a pmf defined at each pixel location of the image.
    This pmf is then sampled for however many photons are required. 
    #TODO This function could be adapted so that some colour scale on the input image could indicate relative energy of photons coming from
    each pixel, with an energy scale given so that a more realistic energy distribution is modeled. As it stands, the function gives each
    photon the same given energy, possibly with a gaussian spread.
    """
    photon_img = image(no_photons)
    # Load the image and convert it to grayscale
    img = Image.open(image_path).convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(img)

    pix_scale = np.array(img_array.shape)

    # Generate a probability mass function from the image
    pmf = img_array / np.sum(img_array)
    
    # Draw N samples from the probability mass function
    pixel_locations = np.random.choice(
        np.arange(img_array.size),
        size=no_photons,
        p=pmf.flatten()    
    )
    
    # Convert the flattened indices back into (x,y) coordinates
    pixel_locations = np.column_stack(np.unravel_index(pixel_locations, img_array.shape))

    # Convert the sampled pixel locations to points of origin on the sky
    photon_img.loc = ((pixel_locations - (pix_scale/2)) * img_scale / pix_scale.max() + np.array(offset)) * 2 * np.pi / (3600 * 360)

    # Generating photon energies and times of arrival
    for i in range(no_photons):
        photon_img.energies[i] = energy * spc.eV * 1e3
        photon_img.toa[i] = i 

    # Adds a spread to the energies if given.
    if energy_spread > 0.:
        photon_img.energies += np.random.normal(0, energy_spread * spc.eV * 1e3, no_photons)

    return photon_img, pix_scale