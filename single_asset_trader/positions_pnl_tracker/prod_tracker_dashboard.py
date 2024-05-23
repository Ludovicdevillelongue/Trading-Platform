import logging
import os
import traceback
import pandas as pd
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
from waitress import serve
import dash
import webbrowser
from time import sleep

class ProductDashboard:
    @staticmethod
    def create_product_strat_value_graph(product_strat_history):
        fig = go.Figure(data=[
            go.Scatter(x=product_strat_history.index, y=product_strat_history['product_value'])
        ])
        fig.update_layout(
            title='Product_strat Value Over Time',
            xaxis_title='Time',
            yaxis_title='Value',
            plot_bgcolor='white',
            paper_bgcolor='lightgray'
        )
        return fig

    @staticmethod
    def create_product_strat_vs_bench_graph(product_strat_history):
        # Assuming 'creturns' and 'cstrategy' are column names in `product_strat_history`
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=product_strat_history.index, y=product_strat_history['creturns'], name='Benchmark'))
        fig.add_trace(go.Scatter(x=product_strat_history.index, y=product_strat_history['cstrategy'], name='Strategy'))

        fig.update_layout(
            title='Product Strat Cumulative Returns vs Benchmark Cumulative Returns',
            xaxis_title='Time',
            yaxis_title='Value',
            plot_bgcolor='white',
            paper_bgcolor='lightgray'
        )
        return fig

    @staticmethod
    def create_data_table(data):
        return dash_table.DataTable(
            data=data.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in data.columns],
            style_data_conditional=[
                {'if': {'row_index': 'odd'},
                 'backgroundColor': 'rgb(248, 248, 248)'},
                {'if': {'column_id': 'Value'},
                 'color': 'red',
                 'fontWeight': 'bold'}
            ],
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )

class ProductManagementApp:
    def __init__(self, platform, symbol, frequency, port):
        self.platform = platform
        self.symbol = symbol
        self.broker_symbol = self.symbol.replace("-", "/") if '-' in self.symbol else self.symbol
        self.frequency = frequency
        self.port = port
        self.app = dash.Dash(__name__, suppress_callback_exceptions=False)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1(f"{self.symbol} Product Management Dashboard", style={'textAlign': 'center'}),
            html.Button("Update Data", id="update-data-button", style={'margin': '10px'}),
            html.Div(id='product-strat-metrics', style={'padding': '10px'}),
            html.Div(id='orders-recap', style={'padding': '10px'}),
            html.Div(id='positions-recap', style={'padding': '10px'}),
            html.Div(id='product-strat-value-graph', style={'padding': '10px'}),
            html.Div(id='product-strat-vs-bench-graph', style={'padding': '10px'}),

            dcc.Interval(id='interval-component', interval=60 * 1000, n_intervals=0),
        ], style={'backgroundColor': '#f5f5f5', 'fontFamily': 'Arial'})

    def setup_callbacks(self):
        @self.app.callback(
            [Output('product-strat-metrics', 'children'),
             Output('orders-recap', 'children'),
             Output('positions-recap', 'children'),
             Output('product-strat-value-graph', 'children'),
             Output('product-strat-vs-bench-graph', 'children')],
            [Input('update-data-button', 'n_clicks'),
             Input('interval-component', 'n_intervals')]
        )
        def update_data(n_clicks, n_intervals):
            try:
                orders_data = self.platform.get_symbol_orders(self.broker_symbol)
                try:
                    positions_data = self.platform.get_symbol_position(self.broker_symbol)
                except Exception as e:
                    positions_data = pd.DataFrame()
                product_strat_history = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                                f'positions_pnl_tracker/{self.symbol}_{self.frequency["interval"]}_strat_history.csv'), header=[0],
                                                    index_col=[0])
                product_strat_history.index = pd.DatetimeIndex(pd.to_datetime(product_strat_history.index, utc=True).
                                                               tz_convert('Europe/Paris'))
                product_strat_history = product_strat_history.sort_index()
                product_strat_metrics = (pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                                  f'positions_pnl_tracker/{self.symbol}_{self.frequency["interval"]}_strat_metric.csv'), header=[0],
                                                     index_col=[0])).round(2)

                value_graph = ProductDashboard.create_product_strat_value_graph(product_strat_history)
                creturns_graph = ProductDashboard.create_product_strat_vs_bench_graph(product_strat_history)
                product_strat_metrics_table = ProductDashboard.create_data_table(product_strat_metrics)
                orders_table = ProductDashboard.create_data_table(orders_data)
                positions_table = ProductDashboard.create_data_table(positions_data)

                return (
                    html.Div([html.H3("Product Traded Metrics"), product_strat_metrics_table]),
                    html.Div([html.H3("Orders Recap"), orders_table]),
                    html.Div([html.H3("Positions Recap"), positions_table]),
                    html.Div([html.H3("Product Traded Value Graph"), dcc.Graph(figure=value_graph)]),
                    html.Div([html.H3("Product Strat vs Bench Graph"), dcc.Graph(figure=creturns_graph)])
                )
            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")
                traceback.print_exc()
                return (html.Div(), html.Div(), html.Div(), html.Div(), html.Div())

    def run_server(self):
        serve(self.app.server, host='0.0.0.0', port=self.port)

    def open_browser(self):
        sleep(1)
        webbrowser.open(f"http://127.0.0.1:{self.port}")
