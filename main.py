import pickle
import plotly.express as px
from dash import dash_table

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from scipy.spatial.distance import squareform

import pandas as pd
import numpy as np
import scipy
import plotly.figure_factory as ff
import scipy.cluster.hierarchy as sch

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import cut_tree
from scipy.cluster.hierarchy import linkage

from sklearn.manifold import MDS


from collections import Counter


# scenarios journal lists
journals_scenario_3 = ["ACADEMY OF MANAGEMENT JOURNAL",
                       "ACADEMY OF MANAGEMENT REVIEW",
                       "ADMINISTRATIVE SCIENCE QUARTERLY",
                       "AMERICAN ECONOMIC REVIEW",
                       "ECONOMETRICA",
                       "JOURNAL OF CONSUMER RESEARCH",
                       "JOURNAL OF MARKETING",
                       "JOURNAL OF MARKETING RESEARCH",
                       "MANAGEMENT SCIENCE",
                       "MARKETING SCIENCE",
                       "SCIENCE",
                       "ACADEMY OF MANAGEMENT ANNALS",
                       "ENTREPRENEURSHIP THEORY AND PRACTICE",
                       "INTERNATIONAL JOURNAL OF RESEARCH IN MARKETING",
                       "JOURNAL OF APPLIED PSYCHOLOGY",
                       "JOURNAL OF BUSINESS VENTURING",
                       "JOURNAL OF CONSUMER PSYCHOLOGY",
                       "JOURNAL OF INTERNATIONAL BUSINESS STUDIES",
                       "JOURNAL OF MANAGEMENT",
                       "JOURNAL OF MANAGEMENT STUDIES",
                       "JOURNAL OF PRODUCT INNOVATION MANAGEMENT",
                       "JOURNAL OF THE ACADEMY OF MARKETING SCIENCE",
                       "ORGANIZATION STUDIES",
                       "RESEARCH POLICY",
                       "STRATEGIC ENTREPRENEURSHIP JOURNAL",
                       "STRATEGIC MANAGEMENT JOURNAL",
                       "ACADEMY OF MANAGEMENT PERSPECTIVES",
                       "BRITISH JOURNAL OF MANAGEMENT",
                       "CALIFORNIA MANAGEMENT REVIEW",
                       "EUROPEAN JOURNAL OF INTERNATIONAL MANAGEMENT",
                       "EUROPEAN MANAGEMENT JOURNAL",
                       "INDUSTRIAL MARKETING MANAGEMENT",
                       "INDUSTRY AND INNOVATION",
                       "INTERNATIONAL JOURNAL OF ENTREPRENEURIAL VENTURING",
                       "INTERNATIONAL JOURNAL OF INDUSTRIAL ORGANIZATION",
                       "INTERNATIONAL JOURNAL OF INNOVATION MANAGEMENT",
                       "INTERNATIONAL JOURNAL OF MANAGEMENT REVIEWS",
                       "JOURNAL OF BUSINESS RESEARCH",
                       "JOURNAL OF INTERNATIONAL MANAGEMENT",
                       "JOURNAL OF INTERNATIONAL MARKETING",
                       "JOURNAL OF PURCHASING AND SUPPLY MANAGEMENT",
                       "JOURNAL OF SMALL BUSINESS MANAGEMENT",
                       "LONG RANGE PLANNING",
                       "REVIEW OF MANAGERIAL SCIENCE",
                       "SCANDINAVIAN JOURNAL OF MANAGEMENT",
                       "SMALL BUSINESS ECONOMICS",
                       "STRATEGIC ORGANIZATION",
                       "JOURNAL OF TECHNOLOGY TRANSFER",
                       "AMS REVIEW",
                       "JOURNAL OF BEHAVIORAL DECISION MAKING"]

#--------------------------------Data

with open('data_dash_2.pickle', 'rb') as f:
    df = pickle.load(f)

# generate subsets for scenario 2 and 3

df["journal"] = df["journal"].str.upper()
df["year"] = [i.year for i in df["dt_year"]]

df_scenario_3 = df[df["journal"].isin(journals_scenario_3)]
df_scenario_2 = df


def hierachy_k(dist_M, k):
    '''
    input: dist_M n*n with 1-pearson values, k number of clusters
    output: cluster mapping
    '''

    condensed_diss = squareform(dist_M)
    linkage_M = linkage(condensed_diss, method="ward", metric="pearson")

    ct = cut_tree(linkage_M, k)

    return [i[0] for i in ct]

def dist_func_pearson(X):
    '''
    equals 1-pearson
    '''
    return scipy.spatial.distance.pdist(X, metric="correlation")

def dist_func_n2(df_cooc):
    '''
    pearson N-2 implemntation
    '''
    M_corr = np.empty(shape=np.shape(df_cooc), dtype=np.float32)
    # loop through coocurenceMatrix and calc n-2 pearson for M_corr
    # inefficent since input and output symmetric - think late about it
    for i,row in enumerate(np.array(df_cooc)): #row
        for j,col in enumerate(np.array(df_cooc).T): # col
            # extract corresponding n-2 vectors
            if i == j:
                M_corr[i,j] = 1
            else:
                x = np.delete(row,obj=[i,j])
                y = np.delete(col,obj=[i,j])

                M_corr[i,j] = np.corrcoef(x,y)[0,1]
    return squareform(1-M_corr)

def dist_func_cosine(X):
    return scipy.spatial.distance.pdist(X, metric="cosine")

def dist_func_euc(X):
    return scipy.spatial.distance.pdist(X, metric="euclidean")

dist_dict = {0: dist_func_pearson,1: dist_func_n2,2: dist_func_euc,3: dist_func_cosine}



#---------------------------------------------Make App
app = dash.Dash(__name__)
server = app.server

# get the options (# of clusters) for the slider in the app
cluster_options = [int(i) for i in range(2,11)]

app.layout = html.Div(className="row", children=[

    # Title
    html.H1("TIME - Biblografic Analysis Dashboard"),

    # Filter + descriptive
    html.H2("I. Filter choices and descriptive statistics"),

    # filter choices
    html.Div(children=[

        # scenario choice
        html.Div(
            html.Div([
                dcc.Dropdown(id="journal-scenario-dropdown",
                             clearable=False,
                             options=[{'label': 'Scenario 2 (restricted)', 'value': 2},
                                      {'label': 'Scenario 3 (only top journals)', 'value': 3}],
                             value=3),
                # to return info message
                html.Div(id='scenario-dd-container', children=[])
            ]), style={"margin-right": "20px"}
        ),

        # dropdown to choose year range
        html.Div([

            # start year
            html.Div([html.H3("From: "),
                      dcc.Dropdown(id='dd-year-start',
                                   clearable=False,
                                   style={"margin-left": "0.5em", "width": "130%"})],
                     style={"display": "flex"}),

            html.Div([html.H3("Until: "),
                      dcc.Dropdown(id='dd-year-end',
                                   clearable=False, style={"margin-left": "0.5em", "width": "130%"})],
                     style={"display": "flex"})

        ]),

    ], style={"width": "100%", "display": "flex", "align-items": "center", "justify-content": "center"}),

    # frequency visualizations
    html.Div([

        # frequ per year graph
        dcc.Graph(id='year-freq-plot'),

        # kw graph
        html.Div(id="keyword-checkboxes", children=[
            dcc.Graph(id='kw-distri-plot')
        ], style={"display": "inline-block"})
    ], style={"width": "100%", "display": "flex", "align-items": "center", "justify-content": "center"}),

    html.H2("II. Cluster and MDS Analysis"),
    html.Div(id="mds-cluster-info"),

    # distance choice
    html.Div(children=[
        html.H3("Choose distance measure here:"),
        dcc.Dropdown(id="distance-measure-dd",
                     clearable=False,
                     options=[{'label': 'Pearson simple', 'value': 0},
                              {'label': 'Pearson N-2', 'value': 1},
                              {'label': 'Eucledean simple', 'value': 2},
                              {'label': 'Cosine simple', 'value': 3}],
                     value=1,
                     style={"margin-left":"0.5em","width":"40%","margin-top":"0.5em"}),
    ],style={"display":"flex"}),
    # Cluster and MDS Visualisations
    html.Div(children=[

        # dendogram
        dcc.Graph(id='dendogram', style={"display": "inline-block"}),

        # slider number clusters
        html.Div([
            dcc.Slider(
                id='cluster-slider',
                min=min(cluster_options),
                max=max(cluster_options),
                value=max(cluster_options),
                marks={str(n): str(n) for n in cluster_options},
                step=None,
                vertical=True,
                verticalHeight=600,
            )
        ], style={"display": "inline-block"}),

        # 3d viz
        html.Div([
            dcc.Graph(id='cluster-3d-graph',
                      style={"width": '80vh', "height": "80vh",
                             "display": "inline-block"}),
            html.Div(id='3d-graph-container')
        ])

    ], style={"width": "100%", "display": "flex", "align-items": "center", "justify-content": "center"}),

    html.Br(),

    # tables

    html.Div([

        # cluster
        html.Div([
            html.H4("Current keywords within each cluster:"),
            html.Div([
                html.Div(id='cluster-table')
            ], style={"width": "100%", "display": "flex", "align-items": "center", "justify-content": "center"})
        ], style={"width": "100%", "display": "inline-block"}),

        # papers
        html.Div([
            html.H4("Papers contained in current filter-settings:"),
            dcc.RadioItems(
                [{'label': 'Read filter_query', 'value': 'read'}, {'label': 'Write to filter_query', 'value': 'write'}],
                'read',
                id='filter-query-read-write',
            ),

            html.Br(),

            dcc.Input(id='filter-query-input', placeholder='Enter filter query'),

            html.Div(id='filter-query-output'),

            html.Hr(),

            html.Div([

                dash_table.DataTable(id="docs-table",
                     sort_action="native",
                     filter_action="native",
                     export_columns="visible",
                     export_format="xlsx",
                     export_headers="display",
                     sort_mode="multi",
                     style_data={'whiteSpace': 'normal',
                                 'height': 'auto',
                                 'lineHeight': '10px',
                                 'overflowX': 'auto'},
                     style_cell_conditional=[
                         {'if': {'column_id': 'year'}, 'width': '5%'},
                         {'if': {'column_id': 'author'}, 'width': '25%'},
                         {'if': {'column_id': 'title'}, 'width': '35%'},
                         {'if': {'column_id': 'times-cited'}, 'width': '5%'},
                         {'if': {'column_id': 'journal'}, 'width': '10%'},
                         {'if': {'column_id': 'keywords'}, 'width': '15%'}
                     ],
                     style_cell={"textAlign": 'left'},
                     style_as_list_view=True
                     ),


            ], style={"width": "100%", "display": "flex", "align-items": "center", "justify-content": "center"})


        ], style={"width": "100%", "display": "inline-block"})

    ]),

    dcc.Store(id='scenario_data'),
    dcc.Store(id='final_data'),
    dcc.Store(id='final_dummies'),
    dcc.Store(id='kw_cluster_data')
])


@app.callback(
    Output('scenario_data', 'data'),
    Input('journal-scenario-dropdown', 'value'))
def update_scenario(scenario):
    # data prep - needs to happen after every filter adjustment in the dashapp

    # 1 filter for scenario

    df["journal"] = df["journal"].str.upper()

    if scenario == 3:
        df_scenario = df_scenario_3
    elif scenario == 2:
        df_scenario = df_scenario_2

    journal_message = f"Scenario {scenario}"

    return df_scenario.to_json(orient='split')


@app.callback(
    Output('dd-year-start', 'options'),
    Output('dd-year-start', 'value'),
    Input('scenario_data', 'data'))
def update_year_start(data_scenario):
    df = pd.read_json(data_scenario, orient='split')

    min_year = int(min(df["year"]))
    max_year = int(max(df["year"]))

    return [{'label': i, 'value': i} for i in range(min_year, max_year + 1)], min_year


@app.callback(
    Output('dd-year-end', 'options'),
    Output('dd-year-end', 'value'),
    Input('dd-year-start', 'value'),
    Input('scenario_data', 'data'))
def update_year_end(start_value, data_scenario):
    df = pd.read_json(data_scenario, orient='split')

    min_year = start_value
    max_year = int(max(df["year"]))

    return [{'label': i, 'value': i} for i in range(min_year, max_year + 1)], max_year


@app.callback(
    Output('final_data', 'data'),
    Output('final_dummies', 'data'),
    Output('scenario-dd-container', 'children'),
    Input('scenario_data', 'data'),
    Input('dd-year-start', 'value'),
    Input('dd-year-end', 'value'))
def update_data_year(data_scenario, start_year, end_year):
    df = pd.read_json(data_scenario, orient='split')

    # filter based on year values
    df_final = df[(df["year"] <= end_year) & (df["year"] >= start_year)]

    # get base info from df
    n_docs = len(df_final)  # number of papers in filtered ds
    df_dummy = df_final.iloc[:, 4:-2]  # only dummy variables (keywords)
    max_kw = len(df_dummy.columns)  # max number of keywords

    # drop keywords with zero occurences in current filter setting
    vec_drop = list(df_dummy.sum(axis=0) > 0)

    # apply to dummy df and df with all variables
    df_dummy = df_dummy.loc[:, vec_drop]
    true_kw = len(df_dummy.columns)  # max number of keywords
    df_final = df_final.loc[:, [True, True, True, True] + vec_drop + [True, True]]

    # put together info message
    message = dcc.Markdown(
        f"Number of papers included in analysis: {n_docs} \nNumber of possible keywords: {max_kw} \nNumber of keywords after filtering: {true_kw}",
        style={"white-space": "pre"})

    return df_final.to_json(orient='split'), df_dummy.to_json(orient='split'), message


### number of papers per year (graph/info)

@app.callback(
    Output('year-freq-plot', 'figure'),
    Input('final_data', 'data'))
def yearly_papers(df):
    data = pd.read_json(df, orient='split')
    years_list = data.loc[:, "year"]
    countery = Counter(years_list)
    countery_data = sorted(countery.items())

    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            x=1,
            y=1.2,
            buttons=list([
                dict(
                    args=[{'yaxis.type': 'linear'}],
                    label="Linear",
                    method="relayout"
                ),
                dict(
                    args=[{'yaxis.type': 'log'}],
                    label="Log",
                    method="relayout"
                )
            ])
        ),
    ]

    p_fig = px.bar(countery_data, x=0, y=1, title="Number of papers published per year")

    p_fig.update_layout(yaxis_title=None, xaxis_title=None,
                        width=800, height=400, updatemenus=updatemenus)

    return p_fig


@app.callback(
    Output('kw-distri-plot', 'figure'),
    Input('final_dummies', 'data'))
def update_kw_figure(dt_words):
    # frequency distribution keywords plot

    dt_words = pd.read_json(dt_words, orient='split')
    kw_distri = dt_words.sum(axis=0).sort_values(ascending=True)

    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            x=1,
            y=1.25,
            buttons=list([
                dict(
                    args=[{'yaxis.type': 'linear'}],
                    label="Linear",
                    method="relayout"
                ),
                dict(
                    args=[{'yaxis.type': 'log'}],
                    label="Log",
                    method="relayout"
                )
            ])
        ),
    ]

    kw_fig = px.bar(kw_distri, title="Number of papers per keyword")
    kw_fig.update_xaxes(tickangle=-45)
    kw_fig.update_layout(width=800, height=400, showlegend=False,
                         yaxis_title=None, xaxis_title=None, updatemenus=updatemenus)
    return kw_fig


@app.callback(
    Output('cluster-3d-graph', 'figure'),
    Output('dendogram', 'figure'),
    Output('mds-cluster-info', 'children'),
    Output('cluster-table', 'children'),
    Output('3d-graph-container', 'children'),
    Output('kw_cluster_data','data'),
    Input('cluster-slider', 'value'),
    Input('final_dummies', 'data'),
    Input('distance-measure-dd','value')
)
def update_cluster_figure_table(n_cluster, data,distance_function_value):
    # diagonals become 0 - we want to ignore the cells for the same keywords
    dt_words = pd.read_json(data, orient='split')
    n_docs = len(dt_words)

    M_cooc = dt_words.T.dot(dt_words)
    np.fill_diagonal(M_cooc.values, 0)

    # get correlation based on that
    # calc distance!-------------------------------
    # set distance fucntion
    dist_func = dist_dict[distance_function_value]

    M_dist = squareform(dist_func(M_cooc))
    order_kw = M_cooc.columns
    #-----------------------------------------------
    checknan = str(np.isnan(M_dist).any())

    n_keywords = len(order_kw)

    if checknan == str(False):
        message = f"dense enough data for MDS/Clustering. Using {n_keywords} out of 36 keywords.\n {n_docs} papers."

        embedding = MDS(n_components=3, dissimilarity='precomputed', random_state=3)
        x_transformed = embedding.fit_transform(M_dist)

        # stress
        stress = embedding.stress_
        stress1 = np.sqrt(stress / (0.5 * np.sum((pd.DataFrame(M_dist)).values ** 2)))
        message_stress = f"Kruskal's Stress : {stress1}  [Poor > 0.2 > Fair > 0.1 > Good > 0.05 > Excellent > 0.025 > Perfect > 0.0]"

        ## Cluster
        cl = hierachy_k(M_dist, n_cluster)
        viz_df = pd.DataFrame({"kw": order_kw, "cluster": cl, "dim1": x_transformed[:, 0],
                               "dim2": x_transformed[:, 1], "dim3": x_transformed[:, 2]})

        viz_df["cluster"] = viz_df["cluster"]+1
        viz_df["cluster"] = viz_df["cluster"].astype("string")

        fig = px.scatter_3d(viz_df, x='dim1', y='dim2', z='dim3', text="kw", color="cluster",
                            color_discrete_sequence=px.colors.qualitative.Safe)

        dendo = ff.create_dendrogram(X=M_cooc, labels=order_kw,
                                     orientation="left",
                                     distfun=dist_func,
                                     linkagefun=lambda x: sch.linkage(x, "ward"))

        dendo.update_layout(width=500, height=700)

        fig.update_layout(transition_duration=500, legend_title_text="Cluster")

        # update table

        df_table = viz_df.loc[:, ["kw", "cluster"]]

        c_table = pd.DataFrame(
            df_table.pivot_table(values='kw', index=df_table.index, columns='cluster', aggfunc='first'))
        c_table = c_table.apply(lambda x: pd.Series(x.dropna().values))

        table = dash_table.DataTable(id="c-table",
                                     columns=[{"name": "cluster " + str(i), "id": str(i)} for i in c_table.columns],
                                     data=c_table.to_dict("records"),
                                     export_columns="visible",
                                     export_format="xlsx",
                                     export_headers="display",
                                     style_table={"overflowX": "auto"},
                                     style_cell={'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
                                                 'whiteSpace': 'normal',
                                                 'textAlign': 'left'},

                                     tooltip_data=[
                                         {
                                             column: {'value': str(value), 'type': 'markdown'}
                                             for column, value in row.items()
                                         } for row in c_table.to_dict('records')
                                     ],
                                     tooltip_duration=None,
                                     style_as_list_view=False)

        return fig, dendo, message, table, message_stress, df_table.to_json(orient='split')

    else:

        message = f"too sparse data for MDS/Clustering!"
        ## MDS

        return {}, {}, message, {}, {}, {}


@app.callback(
    Output('docs-table', 'data'),
    Output('docs-table', 'columns'),
    Input('final_data', 'data'),
    Input('kw_cluster_data','data'))
def update_paper_table(data,kw_cluster):
    # make keyword column, for each papaer containing all related keywords

    df_table = pd.read_json(data, orient='split')

    df_table["id"] = [i for i in range(len(df_table))]
    df_table.set_index('id', inplace=True)

    dummies = df_table.iloc[:, 4:-2]

    df_temp = dummies[dummies == 1].stack().reset_index()
    df_temp = df_temp.groupby(["id"])["level_1"].apply(', '.join).reset_index()

    df_table["keywords"] = df_temp["level_1"]

    # add cluster column based on keywords
    df_kw_cluster = pd.read_json(kw_cluster, orient='split')

    def map_kw_to_cluster(kw_string):
        cl_list = []
        kw_list = kw_string.replace(" ", "").split(",")
        for kw in kw_list:
            cl_list.append(kw_dict[kw])
        return [str(i) for i in cl_list]

    kw_dict = {k: v for k, v in zip(df_kw_cluster["kw"], df_kw_cluster["cluster"])}

    cl_col = [",".join(map_kw_to_cluster(i)) for i in df_table["keywords"]]

    df_table.loc[:, "cluster"] = cl_col

    # -----------------------



    pd_table = df_table.loc[:, ["year","times-cited", "journal", "title", "author", "keywords","cluster"]]
    pd_table["year"] = pd.to_numeric(pd_table["year"])


    columns=[{"name": i, "id": i} for i in pd_table.columns]
    data=pd_table.to_dict("records")

    return data,columns


### custom filter system docs table



@app.callback(
    Output('filter-query-input', 'style'),
    Output('filter-query-output', 'style'),
    Input('filter-query-read-write', 'value')
)
def query_input_output(val):
    input_style = {'width': '100%'}
    output_style = {}
    if val == 'read':
        input_style.update(display='none')
        output_style.update(display='inline-block')
    else:
        input_style.update(display='inline-block')
        output_style.update(display='none')
    return input_style, output_style


@app.callback(
    Output('docs-table', 'filter_query'),
    Input('filter-query-input', 'value')
)
def write_query(query):
    if query is None:
        return ''
    return query


@app.callback(
    Output('filter-query-output', 'children'),
    Input('docs-table', 'filter_query')
)
def read_query(query):
    if query is None:
        return "No filter query"
    return dcc.Markdown('`filter_query = "{}"`'.format(query))



if __name__ == '__main__':
    app.run_server(debug=True)
    # app.run_server(mode='inline')

