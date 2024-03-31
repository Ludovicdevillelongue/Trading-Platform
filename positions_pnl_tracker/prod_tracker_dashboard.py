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

class PortfolioDashboard:
    @staticmethod
    def create_portfolio_value_graph(portfolio_history):
        fig = go.Figure(data=[
            go.Scatter(x=portfolio_history.index, y=portfolio_history['ptf_value'])
        ])
        fig.update_layout(
            title='Portfolio Value Over Time',
            xaxis_title='Time',
            yaxis_title='Value',
            plot_bgcolor='white',
            paper_bgcolor='lightgray'
        )
        return fig

    @staticmethod
    def create_portfolio_vs_bench_graph(portfolio_history):
        # Assuming 'creturns' and 'cstrategy' are column names in `portfolio_history`
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_history.index, y=portfolio_history['creturns'], name='Benchmark'))
        fig.add_trace(go.Scatter(x=portfolio_history.index, y=portfolio_history['cstrategy'], name='Strategy'))

        fig.update_layout(
            title='Portfolio Cumulative Returns vs Benchmark Cumulative Returns',
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

class PortfolioManagementApp:
    def __init__(self, platform, symbol, frequency):
        self.platform = platform
        self.symbol=symbol
        self.frequency=frequency
        self.app = dash.Dash(__name__, suppress_callback_exceptions=False)
        self.setup_layout()
        self.setup_callbacks()


    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Portfolio Management Dashboard", style={'textAlign': 'center'}),
            html.Button("Update Data", id="update-data-button", style={'margin': '10px'}),
            html.Div(id='portfolio-overview', style={'padding': '10px'}),
            html.Div(id='portfolio-metrics', style={'padding': '10px'}),
            html.Div(id='orders-recap', style={'padding': '10px'}),
            html.Div(id='positions-recap', style={'padding': '10px'}),
            html.Div(id='portfolio-value-graph', style={'padding': '10px'}),
            html.Div(id='portfolio-vs-bench-graph', style={'padding': '10px'}),

            dcc.Interval(id='interval-component', interval=60 * 1000, n_intervals=0),
        ], style={'backgroundColor': '#f5f5f5', 'fontFamily': 'Arial'})

    def setup_callbacks(self):
        @self.app.callback(
            [Output('portfolio-overview', 'children'),
            Output('portfolio-metrics', 'children'),
             Output('orders-recap', 'children'),
             Output('positions-recap', 'children'),
             Output('portfolio-value-graph', 'children'),
             Output('portfolio-vs-bench-graph', 'children')],
            [Input('update-data-button', 'n_clicks'),
             Input('interval-component', 'n_intervals')]
        )
        def update_data(n_clicks, n_intervals):
            try:
                portfolio_data = self.platform.get_account_info()
                orders_data = self.platform.get_orders()
                positions_data = self.platform.get_all_positions()
                portfolio_history = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                              f'positions_pnl_tracker/{self.symbol}_{self.frequency['interval']}_strat_history.csv'),header=[0],
                                                index_col=[0])
                portfolio_history.index = pd.DatetimeIndex(pd.to_datetime(portfolio_history.index, utc=True).
                tz_convert('Europe/Paris'))
                portfolio_history=portfolio_history.sort_index()
                portfolio_metrics= (pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                              f'positions_pnl_tracker/{self.symbol}_{self.frequency['interval']}_strat_metric.csv'), header=[0],
                                                index_col=[0])).round(2)

                value_graph = PortfolioDashboard.create_portfolio_value_graph(portfolio_history)
                creturns_graph=PortfolioDashboard.create_portfolio_vs_bench_graph(portfolio_history)
                portfolio_metrics_table=PortfolioDashboard.create_data_table(portfolio_metrics)
                portfolio_table = PortfolioDashboard.create_data_table(portfolio_data)
                orders_table = PortfolioDashboard.create_data_table(orders_data)
                positions_table = PortfolioDashboard.create_data_table(positions_data)

                return (
                    html.Div([html.H3("Portfolio Overview"), portfolio_table]),
                    html.Div([html.H3("Portfolio Metrics"), portfolio_metrics_table]),
                    html.Div([html.H3("Orders Recap"), orders_table]),
                    html.Div([html.H3("Positions Recap"), positions_table]),
                    html.Div([html.H3("Portfolio Value Graph"), dcc.Graph(figure=value_graph)]),
                    html.Div([html.H3("Portfolio vs Bench Graph"), dcc.Graph(figure=creturns_graph)])
                )
            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")
                traceback.print_exc()
                return (html.Div, html.Div(), html.Div(), html.Div(), html.Div(), html.Div())

    def run_server(self):
        serve(self.app.server, host='0.0.0.0', port=8070)

    def open_browser(self):
        sleep(1)
        webbrowser.open("http://127.0.0.1:8070")


