import webbrowser
from time import sleep

import pandas as pd
import plotly.graph_objs as go
from waitress import serve
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output


class PortfolioManagementApp:
    def __init__(self, platform):
        self.platform = platform
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.app.enable_dev_tools(dev_tools_hot_reload=True)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            dcc.Location(id='url', refresh=False),
            html.Div(id='page-content'),
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Interval set to 60 seconds
                n_intervals=0
            )
        ])

    def setup_callbacks(self):
        @self.app.callback(
            Output('page-content', 'children'),
            [Input('interval-component', 'n_intervals'),
             Input('url', 'pathname')]
        )
        def update_dashboard(n, pathname):
            if pathname == '/portfolio':
                portfolio_data = self.get_portfolio_overview()
                return html.Div([
                    html.H1("Portfolio Overview"),
                    html.Pre(str(portfolio_data))
                ])
            elif pathname == '/trade_recap':
                recap_data = self.get_trade_recap()
                return html.Div([
                    html.H1("Trade Recap"),
                    html.Pre(str(recap_data))
                ])
            elif pathname == '/trade_profit_graph':
                trade_data = self.get_trade_recap()['trades']
                graph = self.create_trade_profit_graph(trade_data)
                return html.Div([
                    html.H1("Trade Profit Graph"),
                    dcc.Graph(figure=graph)
                ])
            elif pathname == '/portfolio_value_graph':
                portfolio_history = self.platform.get_portfolio_history()
                graph = self.create_portfolio_value_graph(portfolio_history)
                return html.Div([
                    html.H1("Portfolio Value Graph"),
                    dcc.Graph(figure=graph)
                ])
            else:
                return '404 Page Not Found'


    def get_portfolio_overview(self):
        account_info = self.platform.get_account_info()
        metrics = self.platform.get_portfolio_metrics()
        return {**account_info, **metrics}

    def get_trade_recap(self):
        return self.platform.get_trade_recap()

    def create_trade_profit_graph(self, trade_data):
        trades_df = pd.DataFrame(trade_data)
        fig = go.Figure(data=[
            go.Bar(x=trades_df['symbol'], y=trades_df['profit'])
        ])
        fig.update_layout(title='Trade Profit per Symbol', xaxis_title='Symbol', yaxis_title='Profit')
        return fig

    def create_portfolio_value_graph(self, portfolio_history):
        fig = go.Figure(data=[
            go.Scatter(x=portfolio_history['timestamp'], y=portfolio_history['portfolio_value'])
        ])
        fig.update_layout(title='Portfolio Value Over Time', xaxis_title='Time', yaxis_title='Value')
        return fig

    def run_server(self):

        portfolio_data = self.get_portfolio_overview()

        recap_data = self.get_trade_recap()

        trade_data = self.get_trade_recap()['trades']
        graph = self.create_trade_profit_graph(trade_data)

        portfolio_history = self.platform.get_portfolio_history()
        graph = self.create_portfolio_value_graph(portfolio_history)
        self.app.run(debug=True, host='0.0.0.0', port=8090)

    def open_browser(self):
        sleep(1)  # Short delay before opening the browser
        # Make sure to change the port number here as well
        webbrowser.open("http://127.0.0.1:8090")



