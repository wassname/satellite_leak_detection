import logging
import numpy as np
import ee
import zipfile
import tempfile
from tqdm import tqdm
from scipy.misc import imread
import urllib
from path import Path

ee.Initialize()  # should give no errors, if so follow instructions
logger = logging.getLogger('leaks_helpers')
temp_dir = Path(tempfile.mkdtemp())

bands_NAIP = ['R', 'G', 'B', 'N']

# params https://explorer.earthengine.google.com/#detail/COPERNICUS%2FS2
# 10-30m, 10 day repeat
bands_s2 = [
    'B1',  # Aerosols
    'B2',  # B
    'B3',  # G
    'B4',  # R
    'B5',  # Red Edge 1 705nm
    'B6',  # Red Edge 2
    'B7',  # Red Edge 3
    'B8',  # NIR
    'B8A',
    'B9',
    'B10',
    'B11',
    'B12',  # 2190 nm
    #     'QA10', # empty
    #     'QA20', # empty
    'QA60',  # cloud
]

# Repeats 6 days?
bands_s1 = [
    'VV',
    'HH',
    'VH',
    'HV',
    'angle'
]

# https://explorer.earthengine.google.com/#detail/LANDSAT%2FLE7_L1T
# 16-60m resolution 16 day repeat
bands_l7 = [
    'B1',
    'B2',
    'B3',
    'B4',
    'B5',
    'B6_VCID_1',
    'B6_VCID_2',
    'B7',
    'B8'
]

# https://explorer.earthengine.google.com/#detail/LANDSAT%2FLC8_L1T
# 15-100m, 16 day repeat
bands_l8 = [
    'B1',
    'B2',
    'B3',
    'B4',
    'B5',
    'B6',
    'B7',
    'B8',
    'B9',
    'B10',
    'B11',
    'BQA'
]


def display_ee(geom):
    """show earth-engine object in folium map"""
    import folium
# example https://github.com/python-visualization/folium/blob/master/examples/Geopandas.ipynb
    geojson = geom.getInfo()
    center = geom.centroid(maxError=1).coordinates().getInfo()

#     folium.initialize_notebook()

    map_osm = folium.Map(location=[center[1], center[0]], tiles='Stamen Terrain')
    folium.GeoJson(geojson).add_to(map_osm)
    return map_osm

# def eethumb(image):
#     """Show ee image thumbnail in jupyter notebook"""
#     from IPython.display import HTML
#     return HTML('<img src="'+image.getThumbUrl()+'"/>')


# A tqdm progress bar for urlretrieve see https://github.com/tqdm/tqdm#hooks-and-callbacks
def my_hook(t):
    """
  Wraps tqdm instance. Don't forget to close() or __exit__()
  the tqdm instance once you're done with it (easiest using `with` syntax).

  Example
  -------

  >>> with tqdm(...) as t:
  ...     reporthook = my_hook(t)
  ...     urllib.urlretrieve(..., reporthook=reporthook)

  """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
    b  : int, optional
        Number of blocks just transferred [default: 1].
    bsize  : int, optional
        Size of each block (in tqdm units) [default: 1].
    tsize  : int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
    """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


# https://github.com/google/earthengine-api/blob/master/python/examples/Image/download.py
def download_image(clipped_image, scale=10, crs=4326, name=None, cache_dir=temp_dir, progress_bar=False, report=False):
    """Download image from google earth engine"""

    # TODO add progress bar like https://github.com/fchollet/keras/blob/master/keras/utils/data_utils.py#L103
    path = clipped_image.getDownloadURL({
        'scale': scale,
        'crs': 'EPSG:%s' % crs
    })
    if name is None:
        name = clipped_image.getMapId()['mapid']

    # TODO check that it's not too big, but we don't get given clipped size

    filename = '{name:}_{crs:}_{scale:}'.format(
        name=name, scale=scale, crs=crs)
    zip_dwn_file = temp_dir.joinpath(filename + '.zip')
    if report:
        with tqdm(
                unit='B', unit_scale=True, miniters=1, mininterval=1,
                desc=path.split('/')[-1][:40]) as t:  # all optional kwargs
            zip_dwn_file, r = urllib.request.urlretrieve(
                path, zip_dwn_file, reporthook=my_hook(t))
    else:
        zip_dwn_file, r = urllib.request.urlretrieve(path, zip_dwn_file)
    zip_dwn_file

    # extract
    zfile = zipfile.ZipFile(zip_dwn_file)
    extract_dir = cache_dir.joinpath(filename)
    zfile.extractall(extract_dir)
    #     logger.debug(extract_dir, zip_dwn_file)

    files = [str(f.relpath(extract_dir)) for f in extract_dir.listdir()]
    logger.debug('Extracted files %s to %s', files, str(extract_dir))
    return extract_dir, files

import pyproj
import numpy as np


def get_boundary(leak, distance=100, maxError=None):
    """get rectangular around geopandas point"""
    # coords = np.array(leak.geometry.values[0].xy)[:, 0].tolist()
    # geom = ee.Geometry.Point(coords)
    # boundary = ee.Geometry.buffer(geometry=geom, distance=distance, maxError=maxError)
    # rect = boundary.bounds()

    # Here we make a boundary in wgs84 that will make an exact rectangle in
    # epsg 3857 (aux sphere) so we will get a rectangle when we clip the image
    point_aux = np.array(leak.geometry.to_crs(epsg=3857).values[0].xy)[:, 0]
    xMin, yMin, xMax, yMax = point_aux[0] - distance, point_aux[1] - distance, point_aux[0] + distance, point_aux[1] + distance

    # convert to wgs8
    p0 = pyproj.Proj(init='epsg:%s' % 4326)
    p1 = pyproj.Proj(init='epsg:%s' % 3857)
    bound_grid2 = pyproj.transform(p1, p0, [xMin, xMax], [yMin, yMax])
    bound_grid2 = np.array(bound_grid2).T
    [xMin, yMin], [xMax, yMax] = bound_grid2.min(0), bound_grid2.max(0)

    # make into earth engine rec
    rect = ee.Geometry.Rectangle([xMin, yMin, xMax, yMax])
    return rect

# boundary = get_boundary(leak.geometry)
# # TEST TODO
# b = boundary.getInfo()
# import pyproj
# p0=pyproj.Proj(init='epsg:%s'%4326)
# p1=pyproj.Proj(init='epsg:%s'%3857)
# bb=np.array(b['coordinates'][0])
# bound_grid = pyproj.transform(p0,p1,bb[:,0],bb[:,1])
# # get distance in aux grid
# bg2=np.array(bound_grid).T
# assert bg2.max(0)-bg2.min(0)==[distance, distance]
# assert (bg2.max(0)-bg2.min(0))/min_resolution==pixel_length
#
# def image2array(image, point, crs=4326, resolution_min=resolution_min, pixel_length=pixel_length, name=None):
#
#     scale = resolution_min
#
#     # get image
#     path,files=download_image(image, scale=scale, crs=crs, name=name)
#
#     data = tifs2np(path,files,bands=bands)
#
#     # now if the size is wrong let's interp it
#     if data.shape[-2]!=pixel_length or data.shape[-1]!=pixel_length:
#         data = np.array([sp.misc.imresize(x,size=(pixel_length,pixel_length),interp='cubic', mode='F') for x in data])
#     return data
# test
# data = tifs2np(path,files)

import scipy as sp


def tifs2np(path, files, pixel_length=None, bands=bands_s2):
    """Convert tifs to numpy array"""
    tifs = [f for f in files if f.endswith('.tif')]

    if pixel_length:
        pixel_length = int(pixel_length)

    channels = {}
    for tif in tifs:
        band = tif.split('.')[-2]
        # read tif as float32
        x = imread(path.joinpath(tif), mode='F')
        if pixel_length:
            if x.shape[-2] != pixel_length or x.shape[-1] != pixel_length:
                logger.warn('warning had to reshape band %s from %s to %s' % (band, x.shape, pixel_length))
                x = sp.misc.imresize(x, size=(pixel_length, pixel_length), interp='cubic', mode='F')
        channels[band] = x

    if not pixel_length:
        pixel_length = x.shape[1]

    logger.debug('keys %s', (channels.keys()))
    data = []
    for band in bands:
        if band not in channels:
            channels[band] = np.zeros((pixel_length, pixel_length))
        data.append(channels[band])
    return np.array(data)
