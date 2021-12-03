import csv, sys
import requests
import urllib.request
import os
import pandas as pd
import numpy as np

import rasterio as rio
from rasterio.plot import show
from rasterio import merge, mask

from pyproj import Transformer

import plotly.graph_objects as go
import plotly.io as pio
import plotly as py

from scipy.interpolate import interpn

import datetime as datetime
from astropy.coordinates import get_sun, AltAz, EarthLocation
from astropy.time import Time , TimeDelta, TimezoneInfo
import astropy.units as u

from utils.dataset import *

def get_suntimes (peaks_df, sun_df, date = 'Today', print_times = False):
    peaks_df = peaks_df.copy()
    sun_df = sun_df.copy()
    peaks_df.set_index('bearing_deg', inplace = True)
    peaks_df.drop(columns='bearing', inplace = True)
    peaks_df.index = peaks_df.index.rename('bearing')

    sun_df.index = sun_df.azimuth.copy().rename('bearing')

    mdf = pd.merge(peaks_df, sun_df, how = 'outer', left_index = True , right_index = True)
    mdf.peak_angle = mdf.peak_angle.interpolate('linear')
    mdf.dropna(subset = ['azimuth'], inplace = True)
    mdf.horizon = 0
    mdf.reset_index(inplace = True)

    # mtn_sunrise = Time( mdf.loc[mdf.peak_angle < mdf.elevation].time.min() ).to_value( format = 'ymdhms' )[[ 'hour', 'minute', 'second'] ] 
    # mtn_sunset = Time( mdf.loc[mdf.peak_angle < mdf.elevation].time.max() ).to_value( format = 'ymdhms' )[[ 'hour', 'minute', 'second'] ] 

    mtn_sunrise = [ mdf.loc[mdf.peak_angle < mdf.elevation].time.min() ][0]
    mtn_sunset = [ mdf.loc[mdf.peak_angle < mdf.elevation].time.max() ][0]

    if print_times:
        print(date)
        print('Sunrise at ', mtn_sunrise.hour, ':', mtn_sunrise.minute)
        print('Sunset at ', mtn_sunset.hour, ':', mtn_sunset.minute)
    
    mdf['daylight'] = mdf.loc[mdf.sunlight == 'day'].sunlight
    night = mdf.loc[mdf.sunlight != 'day'].sunlight
    mdf.loc[night.index, 'daylight'] = 'night'    

    mdf['angle_delta'] =  mdf.elevation - mdf.peak_angle

    posneg = 1
    epoch = 0
    for i in mdf.index:
        if mdf.loc[i, 'angle_delta'] * posneg < 0:
            mdf.loc[i, 'epoch'] = epoch
        else:
            epoch+=1
            posneg*=-1
            mdf.loc[i, 'epoch'] = epoch

    df = mdf.loc[mdf.epoch%2 == 1].elevation - mdf.loc[mdf.epoch%2 == 1].peak_angle
    df = df.rename('el_diff')
    mdf = pd.concat([mdf, df], axis = 1).fillna(0)

    return mdf

def get_sun_path (gps_coords, height, date = None ): 
    resolution = 500
    CET = TimezoneInfo(utc_offset = 1*u.hour)

    if date == None:
        dt = Time.now().to_value( format = 'ymdhms' )
        year = dt['year']
        month = dt['month']
        day = dt['day']
    else:
        year = date.year
        month = date.month
        day = date.day

    midnight_this_morning = datetime.datetime(year,month,day, 0,0,0 , tzinfo = CET)

    time_since_midnight = np.linspace(0, 23*60 + 59, resolution) * u.min
    time = ( Time(midnight_this_morning) + time_since_midnight )
    time = time.to_datetime(timezone=CET)

    lat,lon = gps_coords
    loc = EarthLocation.from_geodetic(lon, lat, height = height, ellipsoid = 'WGS84')
    altaz = AltAz(obstime=time, location=loc )
    zen_ang = get_sun(Time(time)).transform_to(altaz)

    elevation = np.array ( zen_ang.alt )
    azimuth= np.array( zen_ang.az )

    dict = {'time' : time, 'time_since_midnight' : time_since_midnight, 'azimuth' : azimuth, 'elevation' : elevation}
    df = pd.DataFrame.from_dict(dict)

    night = df.loc[ df.elevation < 0 ] 
    df.loc[night.index, 'sunlight' ] = 'night'

    day = df.loc[ df.elevation > 0 ]
    df.loc[day.index, 'sunlight' ] = 'day'

    morning_twighlight = df.loc[ (df.elevation < 0) & (df.elevation > -18) & (df.azimuth < 180) ]
    df.loc[morning_twighlight.index, 'sunlight' ] = 'morning_twighlight'

    evening_twighlight = df.loc[ (df.elevation < 0) & (df.elevation > -18) & (df.azimuth > 180) ]
    df.loc[evening_twighlight.index, 'sunlight' ] = 'evening_twighlight'
    # dawn = daytime.head(1).time - TimeDelta( 30 * u.min ).to_datetime()
    # dusk = daytime.tail(1).time + TimeDelta( 30 * u.min ).to_datetime()
    return df

def get_peaks( array, observer_pixel, observer_height, radius, grid_size ):
    
    nrows,ncols = array.shape[1:]
    rows = np.arange(nrows)
    cols = np.arange(ncols)
    points = ( cols, rows )
    array_for_interp = array.T

    angular_resolution = 1000 # / 360 deg
    pixels = radius / grid_size 
    peak = []
    bearing = np.linspace(0, np.pi * 2 , angular_resolution)

    for i, bearing_ in enumerate( bearing ):
        step = np.arange(pixels)
        x_sample =  observer_pixel[1] + np.array( step * np.sin(bearing_) )
        y_sample =  observer_pixel[0] - np.array( step * np.cos(bearing_) )

        inter_points = np.array([ x_sample, y_sample ]).T

        heights = interpn(points, array_for_interp, inter_points, \
                    method = 'linear', bounds_error = False, fill_value = observer_height )[:,0]   \
                     - observer_height
        distances = step * grid_size + 1 # in metres
        peak_angle = max (  heights / ( distances) )
        peak.append(peak_angle)

    df = pd.DataFrame()
    df['bearing'] = bearing
    df['bearing_deg'] = bearing * 180/np.pi
    df['peak_angle'] = np.array(peak) * 180/np.pi
    df.loc[df.peak_angle < 0, 'peak_angle'] = 0

    df['horizon'] = 0
    
    return df

def get_data(gps_coords, observer_height, peaks_df, start_date, final_date = None, td = None):
    hour = np.arange(4, 22)
    tdf_ = pd.DataFrame(columns = ['date', 'time', 'azimuth', 'elevation'])
    tdf = tdf_.copy()
    date = start_date
    while date <= (start_date if final_date == None else final_date):
        sun_df = get_sun_path(gps_coords, observer_height, date)
        mdf = get_suntimes (peaks_df, sun_df)
        mdf['date'] = date
        # time_df.index = ['azimuth', 'elevation']
        amdf = pd.DataFrame()
        for i in hour:
            ind = mdf.time_since_midnight.sub( i*60 ).abs().idxmin()
            tdf_.loc[i, 'azimuth'] = mdf.loc[ind, 'azimuth']
            tdf_.loc[i, 'elevation'] = mdf.loc[ind, 'elevation'] 
        tdf_['grad'] = np.gradient(tdf_.elevation, tdf_.azimuth)
        tdf_['date'] = date
        tdf_['time'] = hour
        tdf_['midday_elev'] = max(mdf.elevation)
        tdf = pd.concat([tdf, tdf_])
        tdf.reset_index(drop = True, inplace = True)
        amdf = pd.concat([amdf, mdf]) # Group all data into one 
        date = date + ( datetime.timedelta(days = 1) if td == None else td) 

    epoch_df = pd.DataFrame()
    for date, gdf in amdf.groupby('date'):
        gdf.sort_values('time_since_midnight', inplace =True)
        epoch_df_ = pd.DataFrame(columns = ['date', 'epoch', 'start_time', 'end_time', 'daylight'])
        for i, epoch in enumerate(gdf.epoch.unique()):
            start = gdf.loc[ gdf.epoch == epoch , 'time_since_midnight' ].min()
            end = gdf.loc[ gdf.epoch == epoch , 'time_since_midnight' ].max()
            daylight = str( gdf.loc[ gdf.epoch == epoch , 'daylight' ].unique() )
            epoch_df_.loc[i, :] = [ date, epoch, start, end, daylight ]
        epoch_df = pd.concat([epoch_df, epoch_df_])

    return mdf, tdf, epoch_df

def make_line_dict(x, y, color = 'black', width = .5):
    return {'line': {
                'color': color,
                'dash': 'solid',
                'width': width},
            'mode': 'none',
            'name': '_line0',
            'showlegend' : False,
            'x': x,
            'xaxis': 'x',
            'y': y,
            'yaxis': 'y',
            'type': 'scatter',
            'fill':'tonexty'
            }

def make_annotation_dict(x, y, text, xshift):
    return  {'text':text, 'x':float(x), 'y':float(y), 
            'xanchor':'center', 
            'yanchor':'middle', 
            # 'height':30,
            'xshift':10*xshift,
            'yshift':-20,
            'showarrow':False,
            'textcolor':'white'
            } 

def get_sun_lines(mdf):
    lines = []
    x = mdf.loc[ mdf.daylight == 'day' ].bearing
    y = mdf.loc[ mdf.daylight == 'day' ].elevation
    lines.append(dict(
                type = 'scatter',
                x = mdf.loc[ mdf.daylight == 'day' ].bearing,
                y = mdf.loc[ mdf.daylight == 'day' ].elevation,
                line = dict(
                    color = 'white',
                    width = 0.5),
                # mode = 'none',
            ))

    # x = mdf.loc[ mdf.daylight == 'day' ].bearing
    # y = mdf.loc[ mdf.daylight == 'day' ].elevation
    # f = sns.lineplot(x = x, y = y , color = 'gold', linewidth=4)

    # x = mdf.loc[ mdf.sunlight == 'morning_twighlight' ].azimuth
    # y = mdf.loc[ mdf.sunlight == 'morning_twighlight' ].elevation
    # # sns.lineplot(x = x, y = y , color = 'royalblue', linewidth=1)
    # lines.append(make_line_dict(x, y, color = 'royalblue', width = .5))


    # x = mdf.loc[ mdf.sunlight == 'evening_twighlight' ].azimuth
    # y = mdf.loc[ mdf.sunlight == 'evening_twighlight' ].elevation
    # # sns.lineplot(x = x, y = y , color = 'royalblue', linewidth=1)
    # lines.append(make_line_dict(x, y, color = 'royalblue', width = .5))

    for epoch in mdf.epoch.unique():
        if epoch%2 == 1:
            x = mdf.loc[ mdf.epoch == epoch ].azimuth
            y = mdf.loc[ mdf.epoch == epoch ].elevation
            # sns.lineplot(x = x, y = y , color = 'gold', linewidth=4)
            lines.append( dict(
                type = 'scatter',
                name = epoch,
                x = x,
                y = y,
                line = dict(color = 'gold'),
                mode = 'none',
                # stackgroup = 'sun',
                fill = 'tonext'
                )
            )

    return lines            

