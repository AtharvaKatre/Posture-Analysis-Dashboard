import urllib.parse as urlparse
import imageio
import cv2
import time
import pafy
import random

import plotly.graph_objs as go
from dash import Dash, html, dcc, Output, Input, State, callback_context
from dash.long_callback import DiskcacheLongCallbackManager
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from engine.PoseEstimation import poseDetector


# -----------------------------------------APP DEFINITION---------------------------------------------------------------
CARD_STYLE = "https://fonts.googleapis.com/css?family=Saira+Semi+Condensed:300,400,700"
## Diskcache
import diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = Dash(
    __name__,
    external_stylesheets=[CARD_STYLE],
    long_callback_manager=long_callback_manager,
    suppress_callback_exceptions=True
)

app.title = 'Posture Analyzer'

# -----------------------------------------APP COMPONENTS---------------------------------------------------------------
def angle_graph(list1, list2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(0, len(list2))], y=list2, mode='lines', name='Shoulder'))
    fig.add_trace(go.Scatter(x=[i for i in range(0, len(list1))], y=list1, mode='lines', name='Hip'))
    fig.update_layout(title='Body Joints Angle Over Time', template='plotly_white', xaxis_title='Time', yaxis_title='Angle (degrees)', height=370, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white', family='changa'))
    return fig

video_dict = {
    "Kickboxing Workout": "https://www.youtube.com/embed/Hri2rYgOLKI?start=411",
    "Katelyn Ohashi - Perfect 10 Beam Gymnastics": "https://www.youtube.com/embed/LWHdVg_g2ng?start=010",
    "Yoga: Full Body Stretch": "https://www.youtube.com/embed/sTANio_2E0Q?start=512",
    "Football Curve Free Kick Tutorial": "https://www.youtube.com/embed/l3j5AegdZME?start=460",
}

yt_input = html.Div([
            dmc.Col(
                dmc.Text("Video Input", color='white', style={"fontSize": 20, "font-family":'changa'})
            ),
            dmc.Center([
                dmc.Col(
                    dmc.Select(
                        label="Select Video",
                        placeholder="Select one",
                        style={'color':'white', "font-family":'changa'},
                        id="video-dropdown",
                        value=list(video_dict.values())[0],
                        data=[
                            {"label": i, "value": k} for i, k in video_dict.items()
                        ],
                        # style={"width": 200, "marginBottom": 10},
                    ), span=10
                )
            ]),
            html.Br(),
            dmc.Text("OR", color='white', weight=700, align='center', style={"font-family":'changa'}),
            html.Br(),
            dmc.Center([
                dmc.Col([
                    dmc.TextInput(label='Paste a Youtube URL Below', id='url-input', style={'color':'white', "font-family":'changa'}),
                    dmc.Text(id="video-flag", style={"color": "red"}),
                ], span=10)
            ]),
            html.Br(),
            dmc.Center(
                dmc.Button("Fetch", id='fetch-button', variant="filled", style={"font-family":'changa'}),
            ),
            html.Br(),
],style={'border':'1px solid white'}
)

yt_video = html.Div([
    dmc.Center(
        html.Iframe(
                    id="youtube-iframe",
                    src=list(video_dict.values())[0],
                    width=850,
                    height=339,
                )
    )
])

model_input = html.Div([
                dmc.Col(
                    dmc.Text("Model Settings", color='white', style={"fontSize": 20, "font-family":'changa'})
                ),
                dmc.Center([
                    dmc.Col(
                        dmc.NumberInput(
                            id='starttime-input',
                            label="Start Time (s): ",
                            value=0,
                            min=0,
                            style={"color": 'white', "font-family":'changa'},
                        ), span=10
                    )
                ]),
                dmc.Center([
                    dmc.Col(
                        dmc.NumberInput(
                            id='duration-input',
                            label="Duration (s): ",
                            value=5,
                            min=1,
                            max=10,
                            style={"color": 'white', "font-family":'changa'},
                        ), span=10
                    )
                ]),
                dmc.Center([
                    dmc.Col([
                        dmc.Text('Detection Confidence: ', color='white', style={"font-family":'changa'}),
                        dmc.Slider(
                            id="detection-slider",
                            value=5,
                            min=0, max=10, step=0.5,
                            style={"width": 250},
                        )
                    ], span=10)
                ]),
                dmc.Center([
                    dmc.Col([
                        dmc.Text('Tracking Confidence: ', color='white', style={"font-family":'changa'}),
                        dmc.Slider(
                            id="tracking-slider",
                            value=5,
                            min=0, max=10, step=0.5,
                            style={"width": 250},
                        )
                    ], span=10)
                ]),
                dmc.Space(h=20),
], style={'border':'1px solid white'})

export_card = html.Div([
                dmc.Col(
                    dmc.Text("Export Output", color='white', style={"fontSize": 20, "font-family":'changa'})
                ),
                dmc.Center([
                    dmc.Col([
                        html.Br(),
                        html.Br(),
                        dmc.Center(DashIconify(icon="el:download", width=90, color='green')),
                        html.Br(),
                        html.Br(),
                        dmc.Center(dmc.Button('Dowload GIF', id='download-button', style={"font-family":'changa'})),
                        dcc.Download(id='download-image')
                    ], span=10)
                ]),
], style={'border':'1px solid white', 'height':'380px'})

output_body = html.Div([
                dmc.Grid([
                        dmc.Col([
                            export_card
                        ], span=3),
                        dmc.Col([
                            dmc.LoadingOverlay(dmc.Image(id='model-output', height=380, style={'max-width':'810'}), loaderProps={"variant": "bars", "size": "xl"})
                        ], span=6),
                        dmc.Col([
                            dcc.Graph(id='line-graph', config={'displaylogo':False}, style={'height':375})
                        ], span=3, style={'border':'1px solid white'}),
                    ])
], id='output-div', style={"visibility": "hidden"})


# ---------------------------------------------APP LAYOUT---------------------------------------------------------------
app.layout = dmc.Container([
    dmc.Center([
        dmc.Grid([
            dmc.Col([
                dmc.Text(
                    "Full Body Posture Analysis App",
                    color='white',
                    align="center",
                    weight=500,
                    style={"fontSize": 36, "font-family":'changa'}
                )
            ], span=12)
        ]),
    ]),
    html.Hr(),
    dmc.Space(h=30),
    dmc.Grid([
            dmc.Col([
                yt_input
            ], span=3),
            dmc.Col([
                yt_video
            ], span=6),
            dmc.Col(
                model_input,
                span=3
            ),
    ]),
    dmc.Space(h=30),
    dmc.Center(
        dmc.Button("Run Model", id="run-model-button", variant='filled', size='lg', style={"font-family":'changa'})
    ),
    dmc.Space(h=30),
    dcc.Store(id='output-path'),
    dmc.LoadingOverlay(output_body, loaderProps={"variant": "bars", "size": "xl"}),
    dmc.Space(h=30)
], fluid=True, style={'backgroundColor':'#111b2b','overflow-y':'hidden'})


# -----------------------------------------APP CALLBAKCS----------------------------------------------------------------
# callback to display youtube video
@app.callback(
    Output("youtube-iframe", "src"),
    Output("video-flag", "children"),
    Input("video-dropdown", "value"),
    Input("fetch-button", "n_clicks"),
    State("url-input", "value")
)
def update_video(url, n_clicks, url2):
    # using callback context to check which input was fired
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "fetch-button":
        if n_clicks:
            url_data = urlparse.urlparse(url2)
            query = urlparse.parse_qs(url_data.query)
            if url_data.path == "/watch":
                id = query["v"][0]
            elif url_data.path[:6] == "/embed":
                id = url_data.path[7:]
            elif not url_data.netloc:
                return "https://www.youtube.com/embed/zz", "Invalid URL"
            else:
                id = url_data.path[1:]
            video = "https://www.youtube.com/embed/{}".format(str(id))
            return video, ""

    if trigger_id == "video-input-dropdown":
        url_data = urlparse.urlparse(url)
        id = url_data.path[7:]
        return url, ""

    else:
        return url, ""

# Disable fetch button when input is empty
@app.callback(Output("fetch-button", "disabled"), Input("url-input", "value"))
def button_flag(value):
    if value:
        return False
    else:
        return True

# callback for updating timestamps of youtube videos
@app.callback(
    Output("starttime-input", "value"),
    Output("duration-input", "value"),
    Input("video-dropdown", "value"),
    Input("fetch-button", "n_clicks")
)
def update_time(url, n_clicks):
    # using callback context to check which input was fired
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "fetch-button":
        return 0, 5
    else:
        return int(url[-3:]), 2

# Download GIF
@app.callback(
    Output("download-image", "data"),
    Input("download-button", "n_clicks"),
    State("output-path", "data"),
    # prevent_initial_call=True
)
def func(n_clicks, path):
    if n_clicks:
        return dcc.send_file(
            path
        )

# running model
@app.long_callback(
    Output("model-output", "src"),
    Output("output-path", "data"),
    Output("line-graph", "figure"),
    # Output("output-body", "style"),
    Input("run-model-button", "n_clicks"),
    State("youtube-iframe", "src"),
    State("starttime-input", "value"),
    State("duration-input", "value"),
    State("detection-slider", "value"),
    State("tracking-slider", "value"),
    running=[
            (Output("run-model-button", "disabled"), True, False),
            (Output("run-model-button", "children"), "Running Model", "Run Model"),
            (Output("output-div", "style"), {"visibility": "hidden"}, {"visibility": "visible"})
        ],
    manager=long_callback_manager,
    prevent_initial_call = True
)
def show_output(nClicks, url, start, duration, detectionCon, trackingCon):
    print('Running')
    if nClicks:

        print(url)
        url_data = urlparse.urlparse(str(url))
        query = urlparse.parse_qs(url_data.query)
        if url_data.path == "/watch":
            id = query["v"][0]
        elif url_data.path[:6] == "/embed":
            id = url_data.path[7:]
        else:
            id = url_data.path[1:]
        video = "https://youtu.be/{}".format(str(id))
        urlPafy = pafy.new(video)
        videoplay = urlPafy.getbest(preftype="any")
        cap = cv2.VideoCapture(videoplay.url)
        # cap = cv2.VideoCapture(0)
        milliseconds = 1000
        end_time = start + duration
        cap.set(cv2.CAP_PROP_POS_MSEC, start * milliseconds)
        pTime = 0
        detector = poseDetector(detectionCon=detectionCon/10, trackCon=trackingCon/10)
        frames = []

        angle_list1 = []
        angle_list2 = []
        while True and cap.get(cv2.CAP_PROP_POS_MSEC) <= end_time * milliseconds:
            # while True:
            success, img = cap.read()
            img = detector.findPose(img)
            lmList = detector.findPosition(img, draw=False)
            if len(lmList) != 0:
                angle1 = detector.findAngle(img, 28, 24, 27)
                angle_list1.append(angle1)
                angle2 = detector.findAngle(img, 14, 12, 24)
                angle_list2.append(angle2)
                # detector.findAngle(img, 26, 28, 32)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # show fps count
            cv2.putText(
                img, 'FPS:'+str(int(fps)), (120, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2
            )
            frames.append(img)

        print("Saving GIF file")
        filename=str(random.random())
        print(filename)
        with imageio.get_writer("..//Posture Tracker//assets//model_runtime_output/output{}.gif".format(filename), mode="I") as writer:
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(rgb_frame)
        print("File saved")

        fig = angle_graph(angle_list1, angle_list2)

        path = "assets/model_runtime_output/output{}.gif".format(filename)

        return path, path, fig


if __name__ == "__main__":
    app.run_server()
