import pandas as pd
import numpy as np

def get_azimuth_vector(azimuth):
    # Calculate the cartesian vectors from the azimuth angles (clockwise from north)
    # Return the normalised vectors, and the quadrant (1 - 4)
    opp = abs( np.sin( (90 - azimuth) * np.pi / 180 ) )
    adj = (1 - opp**2)**0.5
    if azimuth < 90:
        return adj, opp, 1 
    elif azimuth < 180:
        return adj, -1*opp, 2
    elif azimuth < 270:
        return -1*adj, -1*opp, 3
    elif azimuth < 360:
        return -1*adj, opp, 4

def get_shadows(heights, distances, inter_points, elevation):
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

