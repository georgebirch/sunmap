
import rasterio as rio
from rasterio.plot import show
from rasterio import merge
import pandas as pd
import numpy as np

import urllib.request

import geopandas as gpd
from shapely.geometry import LineString, LinearRing, Polygon

from pyproj import Transformer

import geopandas as gpd
from shapely.geometry import LineString, LinearRing, Polygon

def get_masked_data(gps_coords, radius, grid_size, path):

    # path = 'DTM Switzerland 10m v2 by Sonny.tif'
    # path = '/Users/george-birchenough/sunmap_rasters/Himalaya_DEM_20m.tif'
    src = rio.open(path)

    north_buffer = 0.2
    mask_xy = [ [x, (1 - x**2)**0.5] for x in [ np.cos(theta) for theta in np.linspace(0, np.pi, 10)  ]   ]
    mask_xy.insert(0, [1, -north_buffer] )
    mask_xy.append( [-1, -north_buffer] )

    mask_df = pd.DataFrame(mask_xy, columns = ['x', 'y'])
    mask_df['y'] = -1 * mask_df.y
    mask_df = mask_df.append(mask_df.loc[0], ignore_index=True)
    mask_df['order'] = mask_df.index

    gps_lat, gps_lon = gps_coords
    transformer = Transformer.from_crs( 'epsg:4326', src.crs )
    lon, lat = transformer.transform( gps_lat, gps_lon)
    mask_df['lon'] = mask_df.x * radius  + lon
    mask_df['lat'] = mask_df.y * radius  + lat

    gdf = gpd.GeoDataFrame(
        mask_df, geometry=gpd.points_from_xy(mask_df['lon'], mask_df['lat']))
    lineStringObj = Polygon( [[a.x, a.y] for a in gdf.geometry.values] )
    line_dict = {'line':['line'], 'geometry':[lineStringObj]}
    shpgdf = gpd.GeoDataFrame(line_dict, crs = src.crs)
    array, transform = rio.mask.mask(src, shpgdf.geometry, crop=True, all_touched = True)
    out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": array.shape[1],
                     "width": array.shape[2],
                     "transform": transform})

    with rio.open("mask.tif", "w", **out_meta) as dest:
        dest.write(array)

    observer_row = int(north_buffer * radius / grid_size)
    observer_col = int(array.shape[2] / 2)

    observer_pixel = [ observer_row, observer_col] 

    observer_height =  array[0, observer_pixel[0] , observer_pixel[1]] + 2
    array = array[0,:,:]

    return array, observer_pixel, observer_height


def get_square_masked_data(gps_coords, radius, grid_size, path):

    # path = 'DTM Switzerland 10m v2 by Sonny.tif'
    # path = '/Users/george-birchenough/sunmap_rasters/Himalaya_DEM_20m.tif'
    src = rio.open(path)

    x = [-1, -1, 1, 1, -1]
    y = [1, -1, -1, 1, 1]

    mask_df = pd.DataFrame({'x':x, 'y': y}, columns = ['x', 'y'] )
    mask_df
    mask_df['order'] = mask_df.index

    gps_lat, gps_lon = gps_coords
    transformer = Transformer.from_crs( 'epsg:4326', src.crs )
    lon, lat = transformer.transform( gps_lat, gps_lon)
    mask_df['lon'] = mask_df.x * radius  + lon
    mask_df['lat'] = mask_df.y * radius  + lat

    gdf = gpd.GeoDataFrame(
        mask_df, geometry=gpd.points_from_xy(mask_df['lon'], mask_df['lat']))
    lineStringObj = Polygon( [[a.x, a.y] for a in gdf.geometry.values] )
    line_dict = {'line':['line'], 'geometry':[lineStringObj]}
    shpgdf = gpd.GeoDataFrame(line_dict, crs = src.crs)
    array, transform = rio.mask.mask(src, shpgdf.geometry, crop=True, all_touched = True)
    out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": array.shape[1],
                     "width": array.shape[2],
                     "transform": transform})

    with rio.open("mask.tif", "w", **out_meta) as dest:
        dest.write(array)

    observer_row = int(array.shape[1] / 2)

    observer_col = int(array.shape[2] / 2)

    observer_pixel = [ observer_row, observer_col] 

    observer_height =  array[0, observer_pixel[0] , observer_pixel[1]] + 2
    array = array[0,:,:]

    return array, observer_pixel, observer_height
    