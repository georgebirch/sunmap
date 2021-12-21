import pandas as pd
import numpy as np
from scipy.interpolate import interpn

def get_azimuth_vector(azimuth):
    # Calculate the cartesian vectors from the azimuth angles (clockwise from north)
    # Return the normalised vectors, and the quadrant (1 - 4)
    opp = abs( np.sin( (90 - azimuth) * np.pi / 180 ) )
    adj = (1 - opp**2)**0.5
    if azimuth < 90:
        return adj, opp
    elif azimuth < 180:
        return adj, -1*opp
    elif azimuth < 270:
        return -1*adj, -1*opp
    elif azimuth <= 360:
        return -1*adj, opp

def get_shadows_along_slice(heights, distances, inter_points, elevation):
    x = [inter_point[1] for inter_point in inter_points]
    y = [inter_point[0] for inter_point in inter_points]
    d = {'d':distances, 'z':heights, 'x':x, 'y':y, 'max_el':np.zeros([len(distances)])}
    df = pd.DataFrame.from_dict(d)
    el_vector = np.tan(elevation * np.pi / 180)
    for i, z in enumerate(df['z']):
        difs = df.loc[i::, 'z'] - df.loc[i,'z']
        dists = df.loc[i::, 'd'] - df.loc[i, 'd']
        max_el = (difs / dists).max()
        df.loc[i,'max_el'] = max_el
        df.loc[i,'shadow'] = [1 if max_el > el_vector else 0]
    return df

    # for index in df.index:
    #     difs = df.loc[index::, 'z'] - df.loc[index,'z']
    #     dists = df.index[index::] * grid_size
    #     if any(difs / dists > el_vector):
    #         df.loc[index::, 'shadow'] = 1
    #     df.fillna(0, inplace = True)

def get_raster_shadows(array, grid_size, elevation, azimuth ):
    inter_grid_size = grid_size
    nrows,ncols = array.shape
    x_points = np.arange(ncols)
    y_points = np.arange(nrows)
    xy_points = (np.arange(nrows), np.arange(ncols))
    # xy_points = ( np.arange(ncols), np.arange(nrows))

    array_cartesian = np.flip(array, axis=0)
    x_vector, y_vector = get_azimuth_vector(azimuth)
    x_vector*=inter_grid_size
    y_vector*=inter_grid_size

    df = pd.DataFrame()
    if x_vector != 0:
        y_starts =  np.arange(0, nrows, inter_grid_size**2 / abs(x_vector))
        for y_start in y_starts:
            # print('row ',y_start, ' of ', nrows)
            # We make interpolation points starting from first column of each row.
            x_start = 0 if x_vector > 0 else ncols
            x_sample = x_start
            y_sample = y_start
            inter_points = [[y_sample, x_sample]]

            while y_sample >= 0 and y_sample <= nrows  and x_sample >= 0 and x_sample <= ncols :

                y_sample = y_sample + y_vector
                x_sample = x_sample + x_vector
                inter_points_ = [[y_sample, x_sample]]
                inter_points = np.concatenate([ inter_points , inter_points_ ])

            heights = interpn(xy_points, array_cartesian, inter_points, \
                        method = 'linear', bounds_error = False, fill_value = 0 ) 
            distances =  np.arange(1, len(inter_points) + 1) * inter_grid_size  # in metres
            df = pd.concat([df, get_shadows_along_slice(heights, distances, inter_points, elevation)])

        # fig.add_trace( dict(
        #             type = 'scatter',
        #             x = [inter_point[1] for inter_point in inter_points],
        #             y = [inter_point[0] for inter_point in inter_points],
        #             mode = 'lines'
        #             )
        #     )
    if y_vector != 0:
        x_starts = np.arange(0, ncols, inter_grid_size**2 / abs(y_vector))
        for x_start in x_starts:
            # print('row ',y_start, ' of ', nrows)
            # We make interpolation points starting from first column of each row.
            y_start = 0 if y_vector > 0 else nrows
            x_sample = x_start
            y_sample = y_start
            inter_points = [[y_sample, x_sample]]

            while y_sample >= 0 and y_sample <= nrows and x_sample >= 0 and x_sample <= ncols :
                y_sample = y_sample + y_vector
                x_sample = x_sample + x_vector
                inter_points_ = [[y_sample, x_sample]]
                inter_points = np.concatenate([ inter_points , inter_points_ ])

            heights = interpn(xy_points, array_cartesian, inter_points, \
                        method = 'linear', bounds_error = False, fill_value = 0 ) 
            distances =  np.arange(len(inter_points)) * inter_grid_size  # in metres
            df = pd.concat([df, get_shadows_along_slice(heights, distances, inter_points, elevation)])

    df.fillna(0,inplace=True)
    df = df.loc[df.shadow == 1]

    z=array_cartesian
    x=x_points * grid_size
    y=y_points * grid_size
    return df
