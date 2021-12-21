from flask.scaffold import find_package
from utils.paths import *
from utils.dataset import *

from pkg_resources import get_platform

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

gps_coords = 46.05167, 7.68749
# global radius, grid_size
radius = 1000
grid_size = 10
path = '/Users/george-birchenough/sunmap_rasters/Switzerland_DEM_10m.tif'

array, observer_pixel, observer_height  = get_square_masked_data(gps_coords, radius, grid_size, path)
year = 2021
day = 1
month = 6
date = {'year':year,'month':month,'day':day}
date = datetime.date(year = date['year'], month = date['month'], day = date['day'])
df = get_sun_path (gps_coords, observer_height, 4, 22, 19, date = date)
df = df.loc[df.sunlight == 'day'].reset_index(drop=True)
df_dict = {}
for row in df.index:
    df_dict[row] = get_raster_shadows(array, grid_size, df.loc[row, 'elevation'], df.loc[row, 'azimuth'])
    print ('done ', row, ' of ', df.index.max())

slider_marks_dict = { i:{'label':i } for i in df.time_since_midnight/60 }

def main():
    app = dash.Dash(__name__)
    server = app.server

    app.layout = html.Div(children=[

        html.Div(className=' container',  # Define the row element
            children=
                html.Div(html.Img(src=app.get_asset_url('sunmap.png') ),  # Define the left element
                )
            ),
   
        # html.Div(
        #     className='row',  # Define the row element
        #     children=[
        # html.Div(className = ' container ',
        #     children = [
        #         dcc.Input(id="input_lat", type="text", value=str(gps_coords[0]) ),
        #         dcc.Input(id="input_lon", type="text", value=str(gps_coords[1]) ),
        #         html.Button('Confirm', id='btn_state', n_clicks = 0),
        #                 ]
        #             # )
        #     # ]
        # ),  # Define the left elemen

        html.Div(className='row',  # Define the row element
            children=[
                html.Div(className='twelve columns div-for-chart center',
                    children = [
                        dcc.Graph(
                            id='plot1',
                        )
                    ]
                ),  # Define the left element
            ]),

        html.Div(className='row',
            children=[
                # html.Div(className='10 columns center'),
                html.Div(
                    className='div-for-slider center',
                    style = {'color': 'blue', 'fontSize': 14, 'fontFamily':'Arial'},
                    children = [
                        dcc.Slider(
                            id='time_slider',
                            min=0,
                            max=df.index.max(),
                            # marks = { i+1:{'label':month_names[i], 'style':{'color':'red', 'font':{'family':'Verdana','size':22} } } for i in range(12) },
                            # marks = slider_marks_dict,
                            value = 3,
                            # tooltip=dict(always_visible = True, placement = 'bottom')
                        ),
                    ]
                )
            ]
        ),

    ])

    @app.callback(
        Output('plot1', 'figure'),
        Input('time_slider', 'value'))    
    def update_figure(time_index):
        fig = make_figure(time_index)
        return fig

    app.run_server(port = 5050, debug = True)

def make_figure(i):
    df = df_dict[i]

    fig = go.Figure()

    z = array_cartesian = np.flip(array, axis=0)
    y = np.arange( array.shape[0] ) * grid_size
    x = np.arange( array.shape[1] ) * grid_size

    fig.add_trace( dict(
            type = 'contour',
            z=z,
            y = y,
            x = x,
            contours = dict(
                start = array.min() - array.min() % -100,
                end = array.max() - array.max() % -100,
                size = 50,
                coloring = 'none',
                showlabels = True,
            ),
            showlegend = False,
        ))

    fig.add_trace( dict(
            type = 'contour',
            z=z,
            x=x,
            y=y,
            contours = dict(
                start = array.min() - array.min() % -20,
                end = array.max() - array.max() % -20,
                size = 20,
                coloring = 'none',
                ),
            opacity = 0.5,
            showlegend = False,
        ))

    fig.add_trace( dict(
                type = 'scatter',
                x = df.x * grid_size,
                y = df.y * grid_size,
                mode = 'markers',
                marker = {'color':'black', 'size':5},
                opacity = 0.5
                )
        )

    fig.update_layout(
        autosize = False,
        height = 800,
        width = 800,
        yaxis = dict(
                scaleanchor = 'x',
                scaleratio = 1
            ),
        xaxis = dict(
            range = [0, 2 * radius]
            )
        )

    return fig

if __name__ == '__main__':
    main()