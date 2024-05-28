import pandas as pd
import yaml
import sys
import os
sys.path.append('C:\\Users\\Admin\\Documents\\Pro\\projets_code\\python\\trading_platform')

import time
from broker_interaction.broker_metrics import AlpacaPlatform
from broker_interaction.broker_order import GetBrokersConfig
from indicators.performances_indicators import RiskFreeRate
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
from waitress import serve
import webbrowser
from threading import Timer

class PortfolioDashboard:
    def __init__(self, config, initial_amount, frequency):
        self.config = config
        self.platform = AlpacaPlatform(config, frequency)
        self.api = self.platform.get_api_connection()
        self.initial_amount = initial_amount
        self.positions = None
        self.orders = None
        self.account = None
        self.equity_history = []
        self.app = self.create_dash_app()

    def update_data(self):
        risk_free_rate = RiskFreeRate.get_metric()
        try:
            self.positions = self.platform.create_positions_pnl_table()
            self.orders = self.platform.create_orders_table()
            self.portfolio_equity = self.platform.get_all_portfolio_history()
            self.bench_ptf_cumulative_returns, self.key_metrics = self.platform.get_portfolio_metrics(risk_free_rate,
                                                                                                      self.initial_amount)
        except Exception as e:
            self.positions = pd.DataFrame()
            self.orders = pd.DataFrame()
            self.portfolio_equity = pd.DataFrame()
            self.bench_ptf_cumulative_returns = pd.DataFrame()
            self.key_metrics = pd.DataFrame()

    def create_dash_app(self):
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Portfolio Dashboard"), className="mb-2")
            ]),
            dbc.Row([
                dbc.Col(html.H2("Key Metrics"), className="mb-2")
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='key-metrics-table'), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.H2("Positions"), className="mb-2")
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='positions-table'))
            ]),
            dbc.Row([
                dbc.Col(html.H2("Portfolio Value"), className="mb-2")
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='portfolio-equity-graph'), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.H2("Cumulative Returns"), className="mb-2")
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='cumulative-returns-graph'), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.H2("Orders"), className="mb-2")
            ]),
            dbc.Row([
                dbc.Col(html.Div(id='orders-table'))
            ]),
            dcc.Interval(
                id='interval-component',
                interval=60 * 1000,  # in milliseconds
                n_intervals=0
            )
        ], fluid=True)

        @app.callback(
            [Output('positions-table', 'children'),
             Output('orders-table', 'children'),
             Output('portfolio-equity-graph', 'figure'),
             Output('cumulative-returns-graph', 'figure'),
             Output('key-metrics-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            self.update_data()

            positions_table = dbc.Table.from_dataframe(self.positions, striped=True, bordered=True, hover=True)
            orders_table = dbc.Table.from_dataframe(self.orders, striped=True, bordered=True, hover=True)

            equity_fig = go.Figure()
            if not self.portfolio_equity.empty:
                equity_fig.add_trace(
                    go.Scatter(x=self.portfolio_equity.index, y=self.portfolio_equity['equity'], mode='lines',
                               name='Portfolio Equity Evolution'))
                equity_fig.update_layout(title='Portfolio Equity Over Time', xaxis_title='Date', yaxis_title='Equity')

            returns_fig = go.Figure()
            if not self.bench_ptf_cumulative_returns.empty:
                returns_fig.add_trace(go.Scatter(x=self.bench_ptf_cumulative_returns.index,
                                                 y=self.bench_ptf_cumulative_returns['bench_creturns'], mode='lines',
                                                 name='Benchmark Returns'))
                returns_fig.add_trace(go.Scatter(x=self.bench_ptf_cumulative_returns.index,
                                                 y=self.bench_ptf_cumulative_returns['ptf_creturns'], mode='lines',
                                                 name='Portfolio Returns'))
                returns_fig.update_layout(title='Benchmark vs Portfolio Cumulative Returns', xaxis_title='Date',
                                          yaxis_title='Cumulative Returns')

            # Create key metrics table with two rows
            key_metrics_html = []
            for i in range(0, len(self.key_metrics), 2):
                row_data = self.key_metrics.iloc[i:i+2]
                table = dbc.Table.from_dataframe(row_data, striped=True, bordered=True, hover=True)
                key_metrics_html.append(table)
                key_metrics_html.append(html.Br())

            return positions_table, orders_table, equity_fig, returns_fig, key_metrics_html

        return app

    def run_server(self):
        serve(self.app.server, host='0.0.0.0', port=8070)

    def open_browser(self):
        Timer(1, lambda: webbrowser.open("http://127.0.0.1:8070")).start()


if __name__ == '__main__':
    data_provider = 'yfinance'
    with open(r'C:\\Users\\Admin\\Documents\\Pro\\projets_code\\python/trading_platform/config/data_frequency.yml') as file:
        frequency_yaml = yaml.safe_load(file)
    frequency = frequency_yaml[data_provider]['minute']
    broker_config = GetBrokersConfig.key_secret_tc_url()
    initial_amount = 100000
    dashboard = PortfolioDashboard(broker_config, initial_amount, frequency)

    dashboard.open_browser()
    dashboard.run_server()