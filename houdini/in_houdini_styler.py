'''
This is the smoke stylizer for Linux users with Python 3,
or for Windows users using the Python 3 VERSION of Houdini
'''
import hou
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.colors
from PIL import Image
from tqdm import trange
import platform
import subprocess as sp
import numpy as np
import tensorflow as tf
import sys
from datetime import datetime
import imageio
from glob import glob
import shutil
import logging
import json
from scipy.ndimage import gaussian_filter, zoom
import skimage.transform
from functools import partial

def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
  else:
      raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument("--single_frame", type=str2bool, default=False)
parser.add_argument("--iter_seg", type=int, default=0)
parser.add_argument("--style_path", type=str, default="C:/Users/Ozeuth/Houdini-Plugin-for-Tensorflow-Smoke-Stylization")

parser.add_argument("--data_dir", type=str, default="C:/Users/Ozeuth/Houdini-Plugin-for-Tensorflow-Smoke-Stylization/data/smoke_gun")
parser.add_argument("--log_dir", type=str, default="C:/Users/Ozeuth/Houdini-Plugin-for-Tensorflow-Smoke-Stylization/log/smoke_gun")
parser.add_argument("--npz2vdb_dir", type=str, default='data\\npz2vdb')
parser.add_argument("--tag", type=str, default='net')
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--model_path", type=str, default='data/model/tensorflow_inception_graph.pb')
parser.add_argument("--pool1", type=str2bool, default=True)

parser.add_argument("--transmit", type=float, default=0.1)
parser.add_argument("--rotate", type=str2bool, default=True)
parser.add_argument('--phi0', type=int, default=-5) # latitude (elevation) start
parser.add_argument('--phi1', type=int, default=5) # latitude end
parser.add_argument('--phi_unit', type=int, default=5)
parser.add_argument('--theta0', type=int, default=-10) # longitude start
parser.add_argument('--theta1', type=int, default=10) # longitude end
parser.add_argument('--theta_unit', type=int, default=10)
parser.add_argument('--v_batch', type=int, default=1, help='# of rotation matrix for batch process')
parser.add_argument('--n_views', type=int, default=9, help='# of view points')
parser.add_argument('--sample_type', type=str, default='poisson',
                    choices=['uniform', 'poisson', 'both'])

parser.add_argument("--target_frame", type=int, default=70)
parser.add_argument("--num_frames", type=int, default=1)
parser.add_argument("--window_size", type=int, default=1)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--mask", type=str2bool, default=True)

parser.add_argument("--field_type", type=str, default='field',
                    choices=['field', 'velocity', 'density'])
parser.add_argument("--w_field", type=float, default=1, help='weight between pot. and str.')
parser.add_argument("--adv_order", type=int, default=2, choices=[1,2], help='SL or MACCORMACK')
parser.add_argument("--resize_scale", type=float, default=1.0)

parser.add_argument("--content_layer", type=str, default='mixed4d_3x3_bottleneck_pre_relu')
parser.add_argument("--content_channel", type=int, default=44)
parser.add_argument("--style_layer", type=str, default='conv2d2,mixed3b,mixed4b') # Weight layer   # Layer list
parser.add_argument("--w_content", type=float, default=1)
parser.add_argument("--w_content_amp", type=float, default=100)
parser.add_argument("--w_style", type=float, default=0)
parser.add_argument("--w_style_layer", type=str, default='1,1,1') # Weight Ratio
parser.add_argument("--content_target", type=str, default='')
parser.add_argument("--style_target", type=str, default='')
parser.add_argument("--top_k", type=int, default=5)

parser.add_argument("--iter", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--lap_n", type=int, default=3)
parser.add_argument("--octave_n", type=int, default=3)
parser.add_argument("--octave_scale", type=float, default=1.8)
parser.add_argument("--g_sigma", type=float, default=1.2)

node = hou.pwd()
geo = node.geometry()

if (hou.parent() != None):
  density = geo.prims()[0]
  velx = geo.prims()[1]
  vely = geo.prims()[2]
  velz = geo.prims()[3] 
  resolution = velx.resolution()

  # First step: We gather the voxel data for the current frame
  frame_d = np.zeros((resolution[2], resolution[1], resolution[0]-1))
  frame_v = np.zeros((resolution[2], resolution[1], resolution[0]-1, 3))
  for z in range(resolution[2]):
    for y in range(resolution[1]):
      for x in range(resolution[0] - 1):  # Strange Bug: There seems to be one less width than Houdini claims
        frame_d[z][resolution[1] - y - 1][x] = density.voxel((x, y, z))
        frame_v[z][resolution[1] - y - 1][x][0] = velx.voxel((x, y, z))
        frame_v[z][resolution[1] - y - 1][x][1] = vely.voxel((x, y, z))
        frame_v[z][resolution[1] - y - 1][x][2] = velz.voxel((x, y, z))

# ----------------------------- util.py -------------------------------

k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = {
    1: k[:,:,None,None]/k.sum(),
    2: k[:,:,None,None]/k.sum()*np.eye(2, dtype=np.float32),
    3: k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32),
    4: k[:,:,None,None]/k.sum()*np.eye(4, dtype=np.float32),
    6: k[:,:,None,None]/k.sum()*np.eye(6, dtype=np.float32),
}
k_ = []
k2_ = [1,16**(1/3),36**(1/3),16**(1/3),1]
k2 = np.float32([1,16**(1/3),36**(1/3),16**(1/3),1])
k2 = np.outer(k2, k2)
for i in k2_:
    k_.append(k2*i)
k_ = np.floor(np.array(k_))
k5x5x5 = {
    1: k_[:,:,:,None,None]/k_.sum(),
    3: k_[:,:,:,None,None]/k_.sum()*np.eye(3, dtype=np.float32),
    5: k_[:,:,:,None,None]/k_.sum()*np.eye(5, dtype=np.float32),
}

def lap_split(img, is_3d, k):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        if is_3d:
            lo = tf.nn.conv3d(img, k, [1,2,2,2,1], 'SAME')
            lo2 = tf.nn.conv3d_transpose(lo, k*5, tf.shape(img), [1,2,2,2,1])
        else:
            lo = tf.nn.conv2d(img, k, [1,2,2,1], 'SAME')
            lo2 = tf.nn.conv2d_transpose(lo, k*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n, is_3d, k):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img, is_3d, k)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels, is_3d, k):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            if is_3d:
                img = tf.nn.conv3d_transpose(img, k*5, tf.shape(hi), [1,2,2,2,1]) + hi
            else:
                img = tf.nn.conv2d_transpose(img, k*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=3, is_3d=False, c=1):
    '''Perform the Laplacian pyramid normalization.'''
    if scale_n == 0:
        m = tf.reduce_mean(tf.abs(img))
        return img/tf.maximum(m, 1e-7)
    else:
        if is_3d:
            k = k5x5x5[c]
        else:
            k = k5x5[c]

        img = tf.expand_dims(img, 0)
        tlevels = lap_split_n(img, scale_n, is_3d, k)
        tlevels = list(map(normalize_std, tlevels))
        out = lap_merge(tlevels, is_3d, k)
        return out[0]

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.compat.v1.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

def denoise(img, sigma):
    if sigma > 0:
        return gaussian_filter(img, sigma=sigma)
    else:
        return img

def crop_ratio(img, ratio):
    hw_t = img.shape[:2]
    ratio_t = hw_t[1] / float(hw_t[0])
    if ratio_t > ratio:
        hw_ = [hw_t[0], int(hw_t[0]*ratio)]
    else:
        hw_ = [int(hw_t[1]/ratio), hw_t[1]]
    assert(hw_[0] <= hw_t[0] and hw_[1] <= hw_t[1])
    o = [int((hw_t[0]-hw_[0])*0.5), int((hw_t[1]-hw_[1])*0.5)]
    return img[o[0]:o[0]+hw_[0], o[1]:o[1]+hw_[1]]

def resize(img, size=None, f=None, order=3):
    vmin, vmax = img.min(), img.max()
    if vmin < -1 or vmax > 1:
        img = (img - vmin) / (vmax-vmin) # [0,1]
    if size is not None:
        if img.ndim == 4:
            if len(size) == 4: size = size[:-1]
            img_ = []
            for i in range(img.shape[-1]):
                img_.append(skimage.transform.resize(img[...,i], size, order=order).astype(np.float32))
            img = np.stack(img_, axis=-1)
        elif img.ndim < 4:
            img = skimage.transform.resize(img, size, order=order).astype(np.float32)
        else:
            assert False
    else:
        img = skimage.transform.rescale(img, f, order=order).astype(np.float32)
    if vmin < -1 or vmax > 1:
        return img * (vmax-vmin) + vmin
    else:
        return img

def save_density(d, d_path):
    im = d*255
    im = np.stack((im,im,im), axis=-1).astype(np.uint8)
    im = Image.fromarray(im)
    im.save(d_path)

def yuv2rgb(y,u,v):
    # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/image_ops_impl.py
    r = y + 1.13988303*v
    g = y - 0.394642334*u - 0.58062185*v
    b = y + 2.03206185*u
    # r = y + 1.4746*v
    # g = y - 0.16455*u - 0.57135*v
    # b = y + 1.8814*u
    # ## JPEG
    # r = y + 1.402*v
    # g = y - 0.344136*u - 0.714136*v
    # b = y + 1.772*u
    return r,g,b

def rgb2yuv(r,g,b):
    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.14714119*r - 0.28886916*g + 0.43601035*b
    v = 0.61497538*r - 0.51496512*g - 0.10001026*b
    return y,u,v

def hsv2rgb(h,s,v):
    c = s * v
    m = v - c
    dh = h * 6
    h_category = tf.cast(dh, tf.int32)
    fmodu = tf.mod(dh, 2)
    x = c * (1 - tf.abs(fmodu - 1))
    component_shape = tf.shape(h)
    dtype = h.dtype
    rr = tf.zeros(component_shape, dtype=dtype)
    gg = tf.zeros(component_shape, dtype=dtype)
    bb = tf.zeros(component_shape, dtype=dtype)
    h0 = tf.equal(h_category, 0)
    rr = tf.where(h0, c, rr)
    gg = tf.where(h0, x, gg)
    h1 = tf.equal(h_category, 1)
    rr = tf.where(h1, x, rr)
    gg = tf.where(h1, c, gg)
    h2 = tf.equal(h_category, 2)
    gg = tf.where(h2, c, gg)
    bb = tf.where(h2, x, bb)
    h3 = tf.equal(h_category, 3)
    gg = tf.where(h3, x, gg)
    bb = tf.where(h3, c, bb)
    h4 = tf.equal(h_category, 4)
    rr = tf.where(h4, x, rr)
    bb = tf.where(h4, c, bb)
    h5 = tf.equal(h_category, 5)
    rr = tf.where(h5, c, rr)
    bb = tf.where(h5, x, bb)
    r = rr + m
    g = gg + m
    b = bb + m
    return r,g,b

# Util function to match histograms
def match_histograms(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image (source to template)

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    # plt.figure()
    # plt.plot(range(len(s_quantiles)), s_quantiles, range(len(t_quantiles)), t_quantiles)
    # plt.show()

    return interp_t_values[bin_idx].reshape(oldshape)

def str2bool(v):
    return v.lower() in ('true', '1')

def prepare_dirs_and_logger(config):
    os.chdir(os.path.dirname("C:/Users/Ozeuth/Houdini-Plugin-for-Tensorflow-Smoke-Stylization"))

    model_name = "{}_{}".format(get_time(), config.tag)
    config.log_dir = os.path.join(config.log_dir, model_name)
    
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    save_config(config)

def save_config(config):
    param_path = os.path.join(config.log_dir, "params.json")

    print("[*] MODEL dir: %s" % config.log_dir, flush=True)
    print("[*] PARAM path: %s" % param_path, flush=True)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_video(imgdir, filename, ext='png', fps=24, delete_imgdir=False):
    filename = os.path.join(imgdir, '..', filename+'.mp4')
    try:
        writer = imageio.get_writer(filename, fps=fps)
    except Exception:
        imageio.plugins.ffmpeg.download()
        writer = imageio.get_writer(filename, fps=fps)

    imgs = glob("{}/*.{}".format(imgdir, ext))
    imgs = sorted(imgs, key=lambda x: int(os.path.basename(x).split('.')[0]))

    # print(imgs)
    for img in imgs:
        im = imageio.imread(img)
        writer.append_data(im)
    
    writer.close()
    
    if delete_imgdir: shutil.rmtree(imgdir)

def v2rgb(v):
    # lazyfluid colormap    
    theta = np.arctan2(-v[...,0], -v[...,1])
    theta = (theta + np.pi) / (2*np.pi)
    r = np.sqrt(v[...,0]**2+v[...,1]**2)
    r_max = r.max()
    r /= r_max
    o = np.ones_like(r)
    hsv = np.stack((theta,r,o), axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    rgb = (rgb*255).astype(np.uint8)
    return rgb

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False, gray=True):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    if padding == 0:
        if gray:
            grid = np.zeros([height * ymaps, width * xmaps], dtype=np.uint8)
        else:
            grid = np.zeros([height * ymaps, width * xmaps, 3], dtype=np.uint8)
    else:
        if gray:
            grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2], dtype=np.uint8)
        else:
            grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            if padding == 0:
                h, h_width = y * height, height
                w, w_width = x * width, width
            else:
                h, h_width = y * height + 1 + padding // 2, height - padding
                w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False, single=False, gray=True):
    if not single:
        ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                          normalize=normalize, scale_each=scale_each, gray=gray)
    else:
        # h, w = tensor.shape[0], tensor.shape[1]
        # if gray:
        #     ndarr = np.zeros([h,w], dtype=np.uint8)
        # else:
        #     ndarr = np.zeros([h,w,3], dtype=np.uint8)
        ndarr = tensor
        
    im = Image.fromarray(ndarr)
    im.save(filename)
# ------------------------------ poisson.py --------------------------------------
# https://github.com/scipython/scipython_maths/blob/master/poisson_disc_sampled_noise/poisson.py
# For mathematical details of this algorithm, please see the blog
# article at https://scipython.com/blog/poisson-disc-sampling-in-python/
# Christian Hill, March 2017.

class PoissonDisc(object):
    """A class for generating two-dimensional Possion (blue) noise)."""

    def __init__(self, rng, width=50, height=50, r=1, k=30):
        self.rng = rng
        self.width, self.height = width, height
        self.r = r
        self.k = k

        # Cell side length
        self.a = r/np.sqrt(2)
        # Number of cells in the x- and y-directions of the grid
        self.nx, self.ny = int(width / self.a) + 1, int(height / self.a) + 1

        self.reset()

    def reset(self):
        """Reset the cells dictionary."""

        # A list of coordinates in the grid of cells
        coords_list = [(ix, iy) for ix in range(self.nx)
                                for iy in range(self.ny)]
        # Initilalize the dictionary of cells: each key is a cell's coordinates
        # the corresponding value is the index of that cell's point's
        # coordinates in the samples list (or None if the cell is empty).
        self.cells = {coords: None for coords in coords_list}

    def get_cell_coords(self, pt):
        """Get the coordinates of the cell that pt = (x,y) falls in."""

        return int(pt[0] // self.a), int(pt[1] // self.a)

    def get_neighbours(self, coords):
        """Return the indexes of points in cells neighbouring cell at coords.
        For the cell at coords = (x,y), return the indexes of points in the
        cells with neighbouring coordinates illustrated below: ie those cells
        that could contain points closer than r.
                                     ooo
                                    ooooo
                                    ooXoo
                                    ooooo
                                     ooo
        """
        
        dxdy = [(-1,-2),(0,-2),(1,-2),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),
                (-2,0),(-1,0),(1,0),(2,0),(-2,1),(-1,1),(0,1),(1,1),(2,1),
                (-1,2),(0,2),(1,2),(0,0)]
        neighbours = []
        for dx, dy in dxdy:
            neighbour_coords = coords[0] + dx, coords[1] + dy
            if not (0 <= neighbour_coords[0] < self.nx and
                    0 <= neighbour_coords[1] < self.ny):
                # We're off the grid: no neighbours here.
                continue
            neighbour_cell = self.cells[neighbour_coords]
            if neighbour_cell is not None:
                # This cell is occupied: store the index of the contained point
                neighbours.append(neighbour_cell)
        return neighbours

    def point_valid(self, pt):
        """Is pt a valid point to emit as a sample?
        It must be no closer than r from any other point: check the cells in
        its immediate neighbourhood.
        """

        cell_coords = self.get_cell_coords(pt)
        for idx in self.get_neighbours(cell_coords):
            nearby_pt = self.samples[idx]
            # Squared distance between candidate point, pt, and this nearby_pt.
            distance2 = (nearby_pt[0]-pt[0])**2 + (nearby_pt[1]-pt[1])**2
            if distance2 < self.r**2:
                # The points are too close, so pt is not a candidate.
                return False
        # All points tested: if we're here, pt is valid
        return True
    
    def get_point(self, refpt):
        """Try to find a candidate point near refpt to emit in the sample.
        We draw up to k points from the annulus of inner radius r, outer radius
        2r around the reference point, refpt. If none of them are suitable
        (because they're too close to existing points in the sample), return
        False. Otherwise, return the pt.
        """

        i = 0
        while i < self.k:
            rho, theta = (self.rng.uniform(self.r, 2*self.r),
                          self.rng.uniform(0, 2*np.pi))
            pt = refpt[0] + rho*np.cos(theta), refpt[1] + rho*np.sin(theta)
            if not (0 < pt[0] < self.width and 0 < pt[1] < self.height):
                # This point falls outside the domain, so try again.
                continue
            if self.point_valid(pt):
                return pt
            i += 1
        # We failed to find a suitable point in the vicinity of refpt.
        return False

    def sample(self):
        """Poisson disc random sampling in 2D.
        Draw random samples on the domain width x height such that no two
        samples are closer than r apart. The parameter k determines the
        maximum number of candidate points to be chosen around each reference
        point before removing it from the "active" list.
        """

        # Pick a random point to start with.
        pt = (self.rng.uniform(0, self.width),
              self.rng.uniform(0, self.height))
        self.samples = [pt]
        # Our first sample is indexed at 0 in the samples list...
        self.cells[self.get_cell_coords(pt)] = 0
        # and it is active, in the sense that we're going to look for more
        # points in its neighbourhood.
        active = [0]

        # As long as there are points in the active list, keep looking for
        # samples.
        while active:
            # choose a random "reference" point from the active list.
            idx = self.rng.choice(active)
            refpt = self.samples[idx]
            # Try to pick a new point relative to the reference point.
            pt = self.get_point(refpt)
            if pt:
                # Point pt is valid: add it to samples list and mark as active
                self.samples.append(pt)
                nsamples = len(self.samples) - 1
                active.append(nsamples)
                self.cells[self.get_cell_coords(pt)] = nsamples
            else:
                # We had to give up looking for valid points near refpt, so
                # remove it from the list of "active" points.
                active.remove(idx)

        return self.samples
# ----------------------------- transform.py -------------------------------------
# https://github.com/Ryo-Ito/spatial_transformer_network
def mgrid(*args, **kwargs):
    """
    create orthogonal grid
    similar to np.mgrid

    Parameters
    ----------
    args : int
        number of points on each axis
    low : float
        minimum coordinate value
    high : float
        maximum coordinate value

    Returns
    -------
    grid : tf.Tensor [len(args), args[0], ...]
        orthogonal grid
    """
    low = kwargs.pop("low", -1)
    high = kwargs.pop("high", 1)
    low = tf.compat.v1.to_float(low)
    high = tf.compat.v1.to_float(high)
    coords = (tf.linspace(low, high, arg) for arg in args)
    grid = tf.stack(tf.meshgrid(*coords, indexing='ij'))
    return grid


def batch_mgrid(n_batch, *args, **kwargs):
    """
    create batch of orthogonal grids
    similar to np.mgrid

    Parameters
    ----------
    n_batch : int
        number of grids to create
    args : int
        number of points on each axis
    low : float
        minimum coordinate value
    high : float
        maximum coordinate value

    Returns
    -------
    grids : tf.Tensor [n_batch, len(args), args[0], ...]
        batch of orthogonal grids
    """
    grid = mgrid(*args, **kwargs)
    grid = tf.expand_dims(grid, 0)
    grids = tf.tile(grid, [n_batch] + [1 for _ in range(len(args) + 1)])
    return grids

def batch_warp2d(imgs, mappings, sample_shape):
    """
    warp image using mapping function
    I(x) -> I(phi(x))
    phi: mapping function

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    mapping : tf.Tensor
        grids representing mapping function
        [n_batch, xlen, ylen, 2]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, n_channel]
    """
    # n_batch = tf.shape(imgs)[0]
    n_batch = sample_shape[0]
    coords = tf.reshape(mappings, [n_batch, 2, -1])
    x_coords = tf.slice(coords, [0, 0, 0], [-1, 1, -1])
    y_coords = tf.slice(coords, [0, 1, 0], [-1, 1, -1])
    x_coords_flat = tf.reshape(x_coords, [-1])
    y_coords_flat = tf.reshape(y_coords, [-1])

    output = _interpolate2d(imgs, x_coords_flat, y_coords_flat, sample_shape)
    return output

def batch_warp3d(imgs, mappings, sample_shape):
    """
    warp image using mapping function
    I(x) -> I(phi(x))
    phi: mapping function

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    mapping : tf.Tensor
        grids representing mapping function
        [n_batch, 3, xlen, ylen, zlen]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, zlen, n_channel]
    """
    n_batch = sample_shape[0]
    coords = tf.reshape(mappings, [n_batch, 3, -1])
    x_coords = tf.slice(coords, [0, 0, 0], [-1, 1, -1])
    y_coords = tf.slice(coords, [0, 1, 0], [-1, 1, -1])
    z_coords = tf.slice(coords, [0, 2, 0], [-1, 1, -1])
    x_coords_flat = tf.reshape(x_coords, [-1])
    y_coords_flat = tf.reshape(y_coords, [-1])
    z_coords_flat = tf.reshape(z_coords, [-1])

    output = _interpolate3d(imgs, x_coords_flat, y_coords_flat, z_coords_flat, sample_shape)
    return output

def _repeat(base_indices, n_repeats):
    base_indices = tf.matmul(
        tf.reshape(base_indices, [-1, 1]),
        tf.ones([1, n_repeats], dtype='int32'))
    #     tf.reshape(tf.to_float(base_indices), [-1, 1]),
    #     tf.ones([1, n_repeats], dtype=tf.float32))
    # base_indices = tf.to_int32(base_indices)
    return tf.reshape(base_indices, [-1])

def _interpolate2d(imgs, x, y, sample_shape):
    # n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    n_channel = tf.shape(imgs)[3]

    n_batch = sample_shape[0]
    xlen_ = sample_shape[1]
    ylen_ = sample_shape[2]
    
    x = tf.compat.v1.to_float(x)
    y = tf.compat.v1.to_float(y)
    xlen_f = tf.compat.v1.to_float(xlen)
    ylen_f = tf.compat.v1.to_float(ylen)
    zero = tf.zeros([], dtype='int32')
    max_x = tf.cast(xlen - 1, 'int32')
    max_y = tf.cast(ylen - 1, 'int32')

    # scale indices from [-1, 1] to [0, xlen/ylen]
    x = (x + 1.) * (xlen_f - 1.) * 0.5
    y = (y + 1.) * (ylen_f - 1.) * 0.5

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    base = _repeat(tf.range(n_batch) * xlen_ * ylen_, ylen_ * xlen_)
    base_x0 = base + x0 * ylen
    base_x1 = base + x1 * ylen
    index00 = base_x0 + y0
    index01 = base_x0 + y1
    index10 = base_x1 + y0
    index11 = base_x1 + y1

    # use indices to lookup pixels in the flat image and restore
    # n_channel dim
    imgs_flat = tf.reshape(imgs, [-1, n_channel])
    imgs_flat = tf.compat.v1.to_float(imgs_flat)
    I00 = tf.gather(imgs_flat, index00)
    I01 = tf.gather(imgs_flat, index01)
    I10 = tf.gather(imgs_flat, index10)
    I11 = tf.gather(imgs_flat, index11)

    # and finally calculate interpolated values
    dx = x - tf.compat.v1.to_float(x0)
    dy = y - tf.compat.v1.to_float(y0)
    w00 = tf.expand_dims((1. - dx) * (1. - dy), 1)
    w01 = tf.expand_dims((1. - dx) * dy, 1)
    w10 = tf.expand_dims(dx * (1. - dy), 1)
    w11 = tf.expand_dims(dx * dy, 1)
    output = tf.add_n([w00*I00, w01*I01, w10*I10, w11*I11])

    # reshape
    output = tf.reshape(output, [n_batch, xlen_, ylen_, n_channel])

    return output

def _interpolate3d(imgs, x, y, z, sample_shape):
    # n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    zlen = tf.shape(imgs)[3]
    n_channel = tf.shape(imgs)[4]

    n_batch = sample_shape[0]
    xlen_ = sample_shape[1]
    ylen_ = sample_shape[2]
    zlen_ = sample_shape[3]

    x = tf.compat.v1.to_float(x)
    y = tf.compat.v1.to_float(y)
    z = tf.compat.v1.to_float(z)
    xlen_f = tf.compat.v1.to_float(xlen)
    ylen_f = tf.compat.v1.to_float(ylen)
    zlen_f = tf.compat.v1.to_float(zlen)
    zero = tf.zeros([], dtype='int32')
    max_x = tf.cast(xlen - 1, 'int32')
    max_y = tf.cast(ylen - 1, 'int32')
    max_z = tf.cast(zlen - 1, 'int32')

    # scale indices from [-1, 1] to [0, xlen/ylen]
    x = (x + 1.) * (xlen_f - 1.) * 0.5
    y = (y + 1.) * (ylen_f - 1.) * 0.5
    z = (z + 1.) * (zlen_f - 1.) * 0.5

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    z0 = tf.cast(tf.floor(z), 'int32')
    z1 = z0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)
    base = _repeat(tf.range(n_batch) * xlen_ * ylen_ * zlen_,
                   xlen_ * ylen_ * zlen_)
    base_x0 = base + x0 * ylen * zlen
    base_x1 = base + x1 * ylen * zlen
    base00 = base_x0 + y0 * zlen
    base01 = base_x0 + y1 * zlen
    base10 = base_x1 + y0 * zlen
    base11 = base_x1 + y1 * zlen
    index000 = base00 + z0
    index001 = base00 + z1
    index010 = base01 + z0
    index011 = base01 + z1
    index100 = base10 + z0
    index101 = base10 + z1
    index110 = base11 + z0
    index111 = base11 + z1

    # use indices to lookup pixels in the flat image and restore
    # n_channel dim
    imgs_flat = tf.reshape(imgs, [-1, n_channel])
    imgs_flat = tf.compat.v1.to_float(imgs_flat)
    I000 = tf.gather(imgs_flat, index000)
    I001 = tf.gather(imgs_flat, index001)
    I010 = tf.gather(imgs_flat, index010)
    I011 = tf.gather(imgs_flat, index011)
    I100 = tf.gather(imgs_flat, index100)
    I101 = tf.gather(imgs_flat, index101)
    I110 = tf.gather(imgs_flat, index110)
    I111 = tf.gather(imgs_flat, index111)

    # and finally calculate interpolated values
    dx = x - tf.compat.v1.to_float(x0)
    dy = y - tf.compat.v1.to_float(y0)
    dz = z - tf.compat.v1.to_float(z0)
    w000 = tf.expand_dims((1. - dx) * (1. - dy) * (1. - dz), 1)
    w001 = tf.expand_dims((1. - dx) * (1. - dy) * dz, 1)
    w010 = tf.expand_dims((1. - dx) * dy * (1. - dz), 1)
    w011 = tf.expand_dims((1. - dx) * dy * dz, 1)
    w100 = tf.expand_dims(dx * (1. - dy) * (1. - dz), 1)
    w101 = tf.expand_dims(dx * (1. - dy) * dz, 1)
    w110 = tf.expand_dims(dx * dy * (1. - dz), 1)
    w111 = tf.expand_dims(dx * dy * dz, 1)
    output = tf.add_n([w000 * I000, w001 * I001, w010 * I010, w011 * I011,
                       w100 * I100, w101 * I101, w110 * I110, w111 * I111])

    # reshape
    output = tf.reshape(output, [n_batch, xlen_, ylen_, zlen_, n_channel])
    # output = tf.concat([output]*n, axis=0)
    return output

def batch_affine_warp2d(imgs, theta):
    """
    affine transforms 2d images

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    theta : tf.Tensor
        parameters of affine transformation
        [n_batch, 6]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    theta = tf.reshape(theta, [-1, 2, 3])
    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 2])
    t = tf.slice(theta, [0, 0, 2], [-1, -1, -1])

    grids = batch_mgrid(n_batch, xlen, ylen)
    coords = tf.reshape(grids, [n_batch, 2, -1])

    T_g = tf.matmul(matrix, coords) + t
    T_g = tf.reshape(T_g, [n_batch, 2, xlen, ylen])
    with tf.Session() as sess:
        print(sess.run(T_g), flush=True)

    output = batch_warp2d(imgs, T_g)
    return output


def batch_affine_warp3d(imgs, theta):
    """
    affine transforms 3d images

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, zlen, n_channel]
    theta : tf.Tensor
        parameters of affine transformation
        [n_batch, 12]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, zlen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    zlen = tf.shape(imgs)[3]
    theta = tf.reshape(theta, [-1, 3, 4])
    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 3])
    t = tf.slice(theta, [0, 0, 3], [-1, -1, -1])

    grids = batch_mgrid(n_batch, xlen, ylen, zlen)
    grids = tf.reshape(grids, [n_batch, 3, -1])

    T_g = tf.matmul(matrix, grids) + t
    T_g = tf.reshape(T_g, [n_batch, 3, xlen, ylen, zlen])
    output = batch_warp3d(imgs, T_g)
    return output

def grad(p):
    dx = p[:,:,:,1:] - p[:,:,:,:-1]
    dy = p[:,:,1:,:] - p[:,:,:-1,:]
    dz = p[:,1:,:,:] - p[:,:-1,:,:]
    dx = tf.concat((dx, tf.expand_dims(dx[:,:,:,-1], axis=3)), axis=3)
    dy = tf.concat((dy, tf.expand_dims(dy[:,:,-1,:], axis=2)), axis=2)
    dz = tf.concat((dz, tf.expand_dims(dz[:,-1,:,:], axis=1)), axis=1)
    return tf.concat([dx,dy,dz], axis=-1)    

def curl(s):
    # s: [B,D H,W,3]
    # dudx = s[:,:,:,1:,0] - s[:,:,:,:-1,0]
    dvdx = s[:,:,:,1:,1] - s[:,:,:,:-1,1]
    dwdx = s[:,:,:,1:,2] - s[:,:,:,:-1,2]
    
    dudy = s[:,:,1:,:,0] - s[:,:,:-1,:,0]
    # dvdy = s[:,:,1:,:,1] - s[:,:,:-1,:,1]
    dwdy = s[:,:,1:,:,2] - s[:,:,:-1,:,2]
    
    dudz = s[:,1:,:,:,0] - s[:,:-1,:,:,0]
    dvdz = s[:,1:,:,:,1] - s[:,:-1,:,:,1]
    # dwdz = s[:,1:,:,:,2] - s[:,:-1,:,:,2]

    # dudx = tf.concat((dudx, tf.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dvdx = tf.concat((dvdx, tf.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
    dwdx = tf.concat((dwdx, tf.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

    dudy = tf.concat((dudy, tf.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
    # dvdy = tf.concat((dvdy, tf.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
    dwdy = tf.concat((dwdy, tf.expand_dims(dwdy[:,:,-1,:], axis=2)), axis=2)

    dudz = tf.concat((dudz, tf.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
    dvdz = tf.concat((dvdz, tf.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
    # dwdz = tf.concat((dwdz, tf.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy

    return tf.stack([u,v,w], axis=-1)

def advect(d, vel, order=1, is_3d=False):
    n_batch = 1 # assert(tf.shape(d)[0] == 1)
    xlen = tf.shape(d)[1]
    ylen = tf.shape(d)[2]
    
    if is_3d:
        zlen = tf.shape(d)[3]
        grids = batch_mgrid(n_batch, xlen, ylen, zlen) # [b,3,u,v,w]
        vel = tf.transpose(vel, [0,4,1,2,3]) # [b,u,v,w,3] -> [b,3,u,v,w]
        grids -= vel # p' = p - v*dt, dt = 1

        if order == 1: # semi-lagrangian
            d_adv = batch_warp3d(d, grids, [n_batch, xlen, ylen, zlen])
        else: # maccormack
            d_fwd = batch_warp3d(d, grids, [n_batch, xlen, ylen, zlen])
            grids_ = batch_mgrid(n_batch, xlen, ylen, zlen) + vel
            d_bwd = batch_warp3d(d_fwd, grids_, [n_batch, xlen, ylen, zlen])
            d_adv = d_fwd + (d-d_bwd)*0.5
            d_max = tf.nn.max_pool3d(d, ksize=(1,2,2,2,1), strides=(1,1,1,1,1), padding='SAME')
            d_min = -tf.nn.max_pool3d(-d, ksize=(1,2,2,2,1), strides=(1,1,1,1,1), padding='SAME')
            grids = tf.compat.v1.to_int32(grids)
            d_max = batch_warp3d(d_max, grids, [n_batch, xlen, ylen, zlen])
            d_min = batch_warp3d(d_min, grids, [n_batch, xlen, ylen, zlen])
            d_max = tf.greater(d_adv, d_max)
            d_min = tf.greater(d_min, d_adv)
            d_adv = tf.where(tf.logical_or(d_min,d_max), d_fwd, d_adv)
    else:
        grids = batch_mgrid(n_batch, xlen, ylen) # [b,2,u,v]
        vel = tf.transpose(vel, [0,3,1,2]) # [b,u,v,2] -> [b,2,u,v]
        grids -= vel # p' = p - v*dt, dt = 1

        if order == 1:
            d_adv = batch_warp2d(d, grids, [n_batch, xlen, ylen])
        else:
            d_fwd = batch_warp2d(d, grids, [n_batch, xlen, ylen])
            grids_ = batch_mgrid(n_batch, xlen, ylen) + vel
            d_bwd = batch_warp2d(d_fwd, grids_, [n_batch, xlen, ylen])
            # flags = tf.clip_by_value(tf.math.ceil(d_fwd), 0, 1)
            d_adv = d_fwd + (d-d_bwd)*0.5
            d_max = tf.nn.max_pool(d, ksize=(1,2,2,1), strides=(1,1,1,1), padding='SAME')
            d_min = -tf.nn.max_pool(-d, ksize=(1,2,2,1), strides=(1,1,1,1), padding='SAME')
            grids = tf.compat.v1.to_int32(grids)
            # d_max = batch_warp2d(d_max, grids, [n_batch, xlen, ylen])
            # d_min = batch_warp2d(d_min, grids, [n_batch, xlen, ylen])
            d_max, d_min = d_max[grids], d_max[grids]
            # # hard clamp
            # d_adv = tf.clip_by_value(d_adv, d_min, d_max)
            # soft clamp
            d_max = tf.greater(d_adv, d_max) # find values larger than max (true if x > y)
            d_min = tf.greater(d_min, d_adv) # find values smaller than min (true if x > y)
            d_adv = tf.where(tf.logical_or(d_min,d_max), d_fwd, d_adv) # *flags
        
    return d_adv

def rotate(d):
    b = tf.shape(d)[0]
    xlen = tf.shape(d)[1]
    ylen = tf.shape(d)[2]
    zlen = tf.shape(d)[3]
    
    rot_mat = tf.compat.v1.placeholder(shape=[None,3,3], dtype=tf.float32)
    n_rot = tf.shape(rot_mat)[0]
    n_batch = b*n_rot

    d = tf.tile(d, [n_rot,1,1,1,1])
    r = tf.tile(rot_mat, [b,1,1])
    grids = batch_mgrid(n_batch, xlen, ylen, zlen) # [b,3,u,v,w]
    grids = tf.reshape(grids, [n_batch, 3, -1])
    grids = tf.matmul(r, grids)
    grids = tf.reshape(grids, [n_batch, 3, xlen, ylen, zlen])
    d_rot = batch_warp3d(d, grids, [n_batch, xlen, ylen, zlen])
    return d_rot, rot_mat

def subsample(d, scale):
    n_batch = tf.shape(d)[0]
    xlen = tf.compat.v1.to_int32(
        tf.multiply(tf.compat.v1.to_float(tf.shape(d)[1]),scale))
    ylen = tf.compat.v1.to_int32(
        tf.multiply(tf.compat.v1.to_float(tf.shape(d)[2]),scale))
    grids = batch_mgrid(n_batch, xlen, ylen) # [b,2,u,v]
    d_sample = batch_warp2d(d, grids, [n_batch, xlen, ylen])
    return d_sample

def rot_z_3d(deg):
    rad = deg/180.0*np.pi
    c = np.cos(rad)
    s = np.sin(rad)
    rot_mat = np.array([
        [c,-s,0],
        [s,c,0],
        [0,0,1]])
    return rot_mat

def rot_y_3d(deg):
    rad = deg/180.0*np.pi
    c = np.cos(rad)
    s = np.sin(rad)
    rot_mat = np.array([
        [c,0,-s],
        [0,1,0],
        [s,0,c]])
    return rot_mat

def rot_x_3d(deg):
    rad = deg/180.0*np.pi
    c = np.cos(rad)
    s = np.sin(rad)
    rot_mat = np.array([
        [1,0,0],
        [0,c,-s],
        [0,s,c]])
    return rot_mat

def scale(s):
    s_mat = np.array([
        [s,0,0],
        [0,s,0],
        [0,0,s]])
    return s_mat

    
def rot_mat_turb(theta_unit, poisson_sample=False, rng=None):
    views = [{'theta':0}, {'theta':90}, {'theta':180}]
    if poisson_sample:
        views += rot_mat_poisson(0,0,0, 0, 180, theta_unit, rng)

    mat = []
    for view in views:
        theta = view['theta']
        mat.append(rot_y_3d(theta))
    return mat, views

def rot_mat(phi0, phi1, phi_unit, theta0, theta1, theta_unit, 
            sample_type='uniform', rng=None, nv=None):

    if 'uniform' in sample_type:
        views = rot_mat_uniform(phi0, phi1, phi_unit, theta0, theta1, theta_unit)
    elif 'poisson' in sample_type:
        views = rot_mat_poisson(phi0, phi1, phi_unit, theta0, theta1, theta_unit, rng)
        views += rot_mat_uniform(phi0, phi1, 0, theta0, theta1, 0) # [midpoint]
        if nv is not None:
            if len(views) > nv:
                views = views[len(views)-nv:]
            elif len(views) < nv:
                views_ = rot_mat_poisson(phi0, phi1, phi_unit, theta0, theta1, theta_unit, rng)
                views += views_[:nv-len(views)]
    else: # both
        views = rot_mat_uniform(phi0, phi1, phi_unit, theta0, theta1, theta_unit)
        views += rot_mat_poisson(phi0, phi1, phi_unit*2, theta0, theta1, theta_unit*2, rng)
        if nv is not None:
            if len(views) > nv:
                views = views[len(views)-nv:]
            elif len(views) < nv:
                views_ = rot_mat_poisson(phi0, phi1, phi_unit*2, theta0, theta1, theta_unit*2, rng)
                views += views_[:nv-len(views)]

    mat = []
    for view in views:
        phi, theta = view['phi'], view['theta']
        rz = rot_z_3d(phi)
        ry = rot_y_3d(theta)
        rot_mat = np.matmul(ry,rz)
        # s = scale(3)
        # rot_mat = np.matmul(s, rot_mat)
        mat.append(rot_mat)
    return mat, views

def rot_mat_poisson(phi0, phi1, phi_unit, theta0, theta1, theta_unit, rng):
    if phi_unit == 0:
        h = 1
        phi0 = -0.5
    else:
        h = phi1 - phi0

    if theta_unit == 0:
        w = 1
        theta0 = -0.5
    else:
        w = theta1 - theta0

    r = max(phi_unit, theta_unit)/2

    p = PoissonDisc(rng, height=h, width=w, r=r)
    s = p.sample()

    views = []
    for s_ in s:
        phi_ = s_[1]+phi0
        theta_ = s_[0]+theta0
        views.append({'phi':phi_, 'theta':theta_})

    return views

def rot_mat_uniform(phi0, phi1, phi_unit, theta0, theta1, theta_unit):
    if phi_unit == 0:
        phi = [(phi1-phi0)/2]
    else:
        n_phi = np.abs(phi1-phi0) / float(phi_unit) + 1
        phi = np.linspace(phi0, phi1, n_phi, endpoint=True)

    if theta_unit == 0:
        theta = [(theta1-theta0)/2]
    else:
        n_theta = np.abs(theta1-theta0) / float(theta_unit) + 1
        theta = np.linspace(theta0, theta1, n_theta, endpoint=True)    

    views = []
    for phi_ in phi:
        for theta_ in theta:
            views.append({'phi':phi_, 'theta':theta_})

    return views    

# ----------------------------- modified styler.py -------------------------------
class Styler(object):
    def __init__(self, self_dict):
        # get arguments
        for arg in self_dict: setattr(self, arg, self_dict[arg])
        self.rng = np.random.RandomState(self.seed)
        tf.random.set_seed(self.seed)

        # network setting
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.InteractiveSession(graph=self.graph)
        self.model_path = self.style_path + "/" + self.model_path
        with tf.compat.v1.gfile.GFile(self.model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        # fix checkerboard artifacts: ksize should be divisible by the stride size
        # but it changes scale
        if self.pool1:
            for n in graph_def.node:
                if 'conv2d0_pre_relu/conv' in n.name:
                    n.attr['strides'].list.i[1:3] = [1,1]


        # density input
        # shape: [D,H,W]
        d_shp = [None,None,None]
        self.d = tf.compat.v1.placeholder(dtype=tf.float32, shape=d_shp, name='density')

        # add batch dim / channel dim
        # shape: [1,D,H,W,1]
        d = tf.expand_dims(tf.expand_dims(self.d, axis=0), axis=-1)
        
        ######
        # sequence stylization
        self.d_opt = tf.compat.v1.placeholder(dtype=tf.float32, name='opt')

        if 'field' in self.field_type:
            if self.w_field == 1:
                self.c = 1
            elif self.w_field == 0:
                self.c = 3
            else:
                self.c = 4 # scalar (1) + vector field (3)
        elif 'density' in self.field_type:
            self.c = 1 # scalar field
        else:
            self.c = 3 # vector field

        if 'field' in self.field_type:
            d_opt = self.d_opt[:,:,::-1] * tf.compat.v1.to_float(tf.shape(self.d_opt)[2])
            if self.w_field == 1:
                self.v_ = grad(d_opt)
            elif self.w_field == 0:
                self.v_ = curl(d_opt)
            else:
                pot = d_opt[...,0,None]
                strf = d_opt[...,1:]
                self.v_p = grad(pot)
                self.v_s = curl(strf)
                self.v_ = self.v_p*self.w_field + self.v_s*(1-self.w_field)

            v = self.v_[:,:,::-1]
            vx = v[...,0] / tf.compat.v1.to_float(tf.shape(v)[3]) 
            vy = -v[...,1] / tf.compat.v1.to_float(tf.shape(v)[2])
            vz = v[...,2] / tf.compat.v1.to_float(tf.shape(v)[1])
            v = tf.stack([vz,vy,vx], axis=-1)
            d = advect(d, v, order=self.adv_order, is_3d=True)
        elif 'velocity' in self.field_type:
            v = self.d_opt # [1,D,H,W,3]
            d = advect(d, v, order=self.adv_order, is_3d=True)
        else:
            # stylize by addition
            d += self.d_opt # [1,D,H,W,1]

        self.b_num = self.v_batch
        ######

        ######
        # velocity fields to advect gradients [B,D,H,W,3]
        if self.window_size > 1:
            self.v = tf.compat.v1.placeholder(dtype=tf.float32, name='velocity')
            self.g = tf.compat.v1.placeholder(dtype=tf.float32, name='gradient')
            self.adv = advect(self.g, self.v, order=self.adv_order, is_3d=True)
        ######

        # value clipping (d >= 0)
        d = tf.maximum(d, 0)

        # stylized 3d result
        self.d_out = d

        if self.rotate:
            d, self.rot_mat = rotate(d) # [b,D,H,W,1]

            # compute rotation matrices
            self.rot_mat_, self.views = rot_mat(self.phi0, self.phi1, self.phi_unit, 
                self.theta0, self.theta1, self.theta_unit, 
                sample_type=self.sample_type, rng=self.rng,
                nv=self.n_views)
            
            if self.n_views is None:
                self.n_views = len(self.views)
            print('# vps:', self.n_views, flush=True)
            assert(self.n_views % self.v_batch == 0)

        # render 3d volume
        transmit = tf.exp(-tf.cumsum(d[:,::-1], axis=1)*self.transmit)
        d = tf.reduce_sum(d[:,::-1]*transmit, axis=1)
        d /= tf.reduce_max(d) # [0,1]

        # resize if needed 
        if abs(self.resize_scale - 1) > 1e-7:
            h = tf.compat.v1.to_int32(tf.multiply(float(self.resize_scale), tf.compat.v1.to_float(tf.shape(d)[1])))
            w = tf.compat.v1.to_int32(tf.multiply(float(self.resize_scale), tf.compat.v1.to_float(tf.shape(d)[2])))
            d = tf.image.resize_images(d, size=[h, w])

        # change the range of image to [0-255]
        self.d_img = tf.concat([d*255]*3, axis=-1) # [B,H,W,3]

        # plug-in to the pre-trained network
        imagenet_mean = 117.0
        d_preprocessed = self.d_img - imagenet_mean
        tf.import_graph_def(graph_def, {'input': d_preprocessed})
        self.layers = [op.name for op in self.graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
        #print(self.layers)

    def _layer(self, layer):
        if 'input' in layer:
            return self.d_img

        if 'vgg' in self.model_path:
            return self.layers[layer]
        else:
            return self.graph.get_tensor_by_name("import/%s:0" % layer)

    def _gram_matrix(self, x):
        g_ = []
        for i in range(self.b_num):
            F = tf.reshape(x[i], (-1, x.shape[-1]))
            g = tf.matmul(tf.transpose(F), F)
            g_.append(g)
        return tf.stack(g_, axis=0)

    def _loss(self, params):
        self.content_loss = 0
        self.style_loss = 0
        self.total_loss = 0

        if self.w_content:
            feature = self._layer(self.content_layer) # assert only one layer
            if 'content_target' in params:
                self.content_feature = tf.compat.v1.placeholder(tf.float32)
                # self.content_loss -= tf.reduce_mean(feature*self.content_feature) # dot
                self.content_loss += tf.reduce_mean(tf.squared_difference(feature, 
                                               self.content_feature*self.w_content_amp))
            else:
                if self.content_channel:
                    self.content_loss -= tf.reduce_mean(feature[...,self.content_channel])
                    self.content_loss += tf.reduce_mean(tf.abs(feature[...,:self.content_channel]))
                    self.content_loss += tf.reduce_mean(tf.abs(feature[...,self.content_channel+1:]))
                else:
                    self.content_loss -= tf.reduce_mean(feature)

            self.total_loss += self.content_loss*self.w_content

        if self.w_style and 'style_target' in params:
            self.style_features = []
            self.style_denoms = []
            style_layers = self.style_layer.split(',')
            for style_layer in style_layers:
                feature = self._layer(style_layer)
                gram = self._gram_matrix(feature)
                f_shp = feature.shape
                style_feature = tf.compat.v1.placeholder(tf.float32, shape=f_shp)
                style_gram = self._gram_matrix(style_feature)

                style_denom = tf.compat.v1.placeholder(tf.float32, shape=1)
                self.style_loss += tf.reduce_sum(tf.squared_difference(gram, style_gram)) / style_denom
                self.style_features.append(style_feature)
                self.style_denoms.append(style_denom)

            self.total_loss += self.style_loss*self.w_style

    def _content_feature(self, content_target, content_shp):
        if abs(self.resize_scale - 1) > 1e-7:
            content_shp = [int(s*self.resize_scale) for s in content_shp]
        content_target_ = resize(content_target, content_shp)
        feature = self._layer(self.content_layer)
        feature_ = self.sess.run(feature, {self.d_img: [content_target_]*self.b_num})

        if self.top_k > 0:
            assert('softmax2_pre_activation' in self.content_layer)
            feature_k_ = self.sess.run(tf.nn.top_k(np.abs(feature_), k=self.top_k))
            for i in range(len(feature_)):
                exclude_idx = np.setdiff1d(np.arange(feature_.shape[1]), feature_k_.indices[i])
                feature_[i,exclude_idx] = 0
        
        return feature_
    def _style_feature(self, style_target, style_shp):
        style_mask = None
        if style_target.shape[-1] == 4:
            style_mask = style_target[...,-1] / 255
            style_target = style_target[...,:-1]


        if abs(self.resize_scale - 1) > 1e-7:
            style_shp = [int(s*self.resize_scale) for s in style_shp]
        style_target_ = resize(style_target, style_shp)
        style_layers = self.style_layer.split(',')
        w_style_layers = self.w_style_layer.split(',')
        style_features = []
        style_denoms = []
        for style_layer, w_style_layer in zip(style_layers, w_style_layers):
            style_feature = self._layer(style_layer)
            style_feature_ = self.sess.run(style_feature, {self.d_img: [style_target_]*self.b_num})

            f_shp = style_feature_.shape
            area = f_shp[1]*f_shp[2]
            nc = f_shp[3]
            denom = [4.0 * area**2 * nc**2 * 1e6 / float(w_style_layer)]
            if style_mask is not None:
                feature_mask = resize(style_mask, style_feature_.shape[1:-1])
                feature_mask = np.stack([feature_mask]*style_feature_.shape[-1], axis=-1)
                for i in range(self.b_num):
                    style_feature_[i] *= feature_mask

            style_features.append(style_feature_)
            style_denoms.append(denom)

        return style_features, style_denoms

    def _transport(self, g, v, a, b):
        if a < b:
            for i in range(a,b):
                g = self.sess.run(self.adv, {self.g: g, self.v: v[i,None]})
        elif a > b:
            for i in reversed(range(b,a)):
                g = self.sess.run(self.adv, {self.g: g, self.v: -v[i,None]})
        return g

    def run(self, params):
        # loss
        self._loss(params)

        # gradient
        g = tf.gradients(-self.total_loss, self.d_opt)[0]

        # laplacian gradient normalizer
        grad_norm = tffunc(np.float32)(partial(lap_normalize, 
            scale_n=self.lap_n, c=self.c, is_3d=True))

        d = params['d']
        if 'mask' in params:
            mask = params['mask']
            mask = np.stack([mask]*self.c, axis=-1)

        if 'v' in params:
            v = params['v']

        # settings for octave process
        oct_size = []
        hw = np.int32(d.shape)[1:]
        for _ in range(self.octave_n):
            oct_size.append(hw.copy())
            hw = np.int32(np.float32(hw)/self.octave_scale)
        print('input size for each octave', oct_size, flush=True)

        d_shp = [self.num_frames] + [s for s in oct_size[-1]] + [self.c]
        d_opt_ = np.zeros(shape=d_shp, dtype=np.float32)

        # optimize
        loss_history = []
        for octave in trange(self.octave_n):
            # octave process: scale-down for input
            if octave < self.octave_n-1:
                d_ = []
                for i in range(self.num_frames):
                    d_.append(resize(d[i], oct_size[-octave-1]))
                d_ = np.array(d_)

                if 'mask' in params:
                    mask_ = []
                    for i in range(self.num_frames):
                        m = resize(mask[i], oct_size[-octave-1])
                        mask_.append(m)

                if 'v' in params:
                    v_ = []
                    for i in range(self.num_frames-1):
                        v_.append(resize(v[i], oct_size[-octave-1]))
                    v_ = np.array(v_)
            else:
                d_ = d
                if 'mask' in params: mask_ = mask
                if 'v' in params: v_ = v

            if octave > 0:
                d_opt__ = []
                for i in range(self.num_frames):
                    d_opt__.append(resize(d_opt_[i], oct_size[-octave-1]))
                del d_opt_
                d_opt_ = np.array(d_opt__)
            
            feed = {}
            
            if 'content_target' in params:
                feed[self.content_feature] = self._content_feature(
                    params['content_target'], oct_size[-octave-1][1:])

            if 'style_target' in params:
                style_features, style_denoms = self._style_feature(
                    params['style_target'], oct_size[-octave-1][1:]
                )

                for i in range(len(self.style_features)):
                    feed[self.style_features[i]] = style_features[i]
                    feed[self.style_denoms[i]] = style_denoms[i]
            if (octave == self.octave_n - 1):
                d_opt_iter = []
            for step in trange(self.iter):
                g__ = []
                for t in trange(self.num_frames):
                    feed[self.d] = d_[t]
                    feed[self.d_opt] = d_opt_[t,None]

                    if self.rotate:
                        g_ = None
                        l_ = 0
                        for i in range(0, self.n_views, self.v_batch):
                            feed[self.rot_mat] = self.rot_mat_[i:i+self.v_batch]
                            g_vp, l_vp = self.sess.run([g, self.total_loss], feed)
                            if g_ is None:
                                g_ = g_vp
                            else:
                                g_ += g_vp
                            l_ += l_vp
                        l_ /= np.ceil(self.n_views/self.v_batch)

                        if not 'uniform' in self.sample_type:
                            self.rot_mat_, self.views = rot_mat(
                                self.phi0, self.phi1, self.phi_unit, 
                                self.theta0, self.theta1, self.theta_unit, 
                                sample_type=self.sample_type, rng=self.rng,
                                nv=self.n_views)
                    else:
                        g_, l_ = self.sess.run([g, self.total_loss], feed)
                        loss_history.append(l_)

                    g_ = denoise(g_, sigma=self.g_sigma)

                    if 'lr' in params:
                        lr = params['lr'][min(t, len(params['lr'])-1)]
                        g_[0] = grad_norm(g_[0]) * lr
                    else:
                        g_[0] = grad_norm(g_[0]) * self.lr
                    if 'mask' in params: g_[0] *= mask_[t]

                    g__.append(g_)

                if self.window_size > 1:
                    n = (self.window_size-1) // 2
                    for t in range(self.num_frames):
                        t0 = np.maximum(t - n, 0)
                        t1 = np.minimum(t + n, self.num_frames-1)
                        # print(t, t0, t1)
                        w = [1/(t1-t0+1)]*self.num_frames

                        g_ = g__[t].copy() * w[t]
                        for s in range(t0,t1+1):
                            if s == t: continue
                            g_ += self._transport(g__[s].copy(), v_, s, t) * w[s] # move s to t

                        d_opt_[t] += g_[0]
                        g__[t] = g_
                else:
                    for t in range(self.num_frames):
                        d_opt_[t] += g__[t][0]

                # to avoid resizing numerical error
                if 'mask' in params:
                    for t in range(self.num_frames):
                        d_opt_[t] *= np.ceil(mask_[t])

                if self.iter_seg > 0 and octave == self.octave_n - 1:
                    if (((step / float(self.iter_seg)) - int(step / self.iter_seg)) < 0.00001) and (step != self.iter - 1) and (step != 0):
                        d_opt_iter.append(np.array(d_opt_, copy=True))
        # gather outputs
        result = {'l': loss_history}


        d_opt_iter = np.array(d_opt_iter)
        d_iter = []
        for i in range(d_opt_iter.shape[0]):
            d__ = []
            d_out_ = tf.identity(self.d_out)
            #feed_ = tf.identity(feed)
            for t in range(self.num_frames):
                feed[self.d_opt] = d_opt_iter[i, t, None]
                feed[self.d] = d[t]
                d__.append(self.sess.run(d_out_, feed)[0,...,0])
            d__ = np.array(d__)
            d_iter.append(d__)
        d_iter = np.array(d_iter)
        result['d_iter'] = d_iter
       
        d_ = []
        for t in range(self.num_frames):
            feed[self.d_opt] = d_opt_[t,None]
            feed[self.d] = d[t]
            d_.append(self.sess.run(self.d_out, feed)[0,...,0])
        d_ = np.array(d_)
        result['d'] = d_
        return result

def stylize(args, v, d):
    # create a styler
    styler = Styler(vars(args))

    prepare_dirs_and_logger(args)

    # set file path format
    d_path_format = os.path.join(args.data_dir, 'd', '%03d.npz')
    v_path_format = os.path.join(args.data_dir, 'v', '%03d.npz')

    # directories for some visual results
    d_dir = os.path.join(args.log_dir, 'd') # original
    v_dir = os.path.join(args.log_dir, 'v') # velocity-mid slice
    r_dir = os.path.join(args.log_dir, 'r') # result
    for img_dir in [d_dir, v_dir, r_dir]:
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

    d_path = d_path_format % args.target_frame

    print('load density fields', flush=True)
    # First Step: load all data from stfin
    transmit = np.exp(-np.cumsum(d[::-1], axis=0)*args.transmit)
    d_img = np.sum(d[::-1]*transmit, axis=0)
    d_img /= d_img.max()
    d_img_amount = [np.sum(d_img)]

    v = np.array([v])
    d = np.array([d])
    # Second Step: Normalize the results
    d_max = d.max()
    d /= d_max
    v_max = v.max()
    v /= v_max

    params = {'d': d} 
    # Third Step: set learning rate depending on the amount of density
    d_amount = np.sum(d, axis=(1,2,3))
    d_img_amount = np.array(d_img_amount)
    d_img_amount /= d_img_amount.max()
    params['lr'] = d_img_amount*args.lr
    d_shp = d.shape[1:] # zyx -> dhw
    params['v'] = v

    # Fourth Step (All): mask
    if args.mask:
        params['mask'] = denoise(d, args.g_sigma)

    # Fifth Step (All): load a content target image
    if args.content_target:
        content_target = np.float32(Image.open(args.content_target))
        # remove alpha channel
        if content_target.shape[-1] == 4:
            content_target = content_target[...,:-1]
        
        # crop
        ratio = d_shp[2] / float(d_shp[1]) # x/y
        content_target = crop_ratio(content_target, ratio)

        # range is still [0-255]
        params['content_target'] = content_target
        # plt.figure()
        # plt.imshow(content_target/255)
        # plt.show()
    # Sixth Step (All): load a style target
    if args.style_target:
        style_target = np.float32(Image.open(args.style_target))
        # crop
        ratio = d_shp[2] / float(d_shp[1])
        style_target = crop_ratio(style_target, ratio)
        # range is still [0-255]
        params['style_target'] = style_target
        # plt.figure()
        # plt.imshow(style_target/255)
        # plt.show()

    #########
    # Seventh Step (All): stylize
    result = styler.run(params)
    d_sty, loss, d_iter = result['d'], result['l'], result['d_iter']

    # Denormalize Results
    d_sty *= d_max

    rs = []
    for i, d_sty_ in enumerate(d_sty):
        r = np.array(d_sty_, copy=True, dtype = np.float32)
        r = r[:,::-1]
        rs.append(r)
    
    styler.sess.close()
    print("complete", flush=True)
    return rs
    
args = parser.parse_args()
frames_r = stylize(args, frame_v, frame_d)
resolution = density.resolution()

for z in range(resolution[2]):
  for y in range(resolution[1]): 
    for x in range(resolution[0]-1):
      density.setVoxel((x, y, z), float(frames_r[0][z][y][x]))

