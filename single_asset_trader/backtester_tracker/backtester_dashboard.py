import logging
import traceback
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
from waitress import serve
from dash import dash_table
import webbrowser
from time import sleep

class BacktestDashboard:
    @staticmethod
    def plot_cumulative_returns(returns):
        fig = px.line(returns, x=returns.index, y=returns.columns,
                      title='Cumulative Returns and Strategies Comparison')
        return fig

    @staticmethod
    def plot_positions(positions):
        figures = []
        for column in positions.columns:
            fig = px.line(positions, x=positions.index, y=column, title=f'{column}', render_mode='svg')
            figures.append(fig)
        return figures

    @staticmethod
    def create_table(best_strat_recap):
        table_data = []
        for strategy_name, strategy_result in best_strat_recap.items():
            table_data.append({'strategy_name': strategy_name,
                               'search_type': str(strategy_result['search_type']),
                               'params': str(strategy_result['params']),
                               **strategy_result['results']})
        table_data = sorted(table_data, key=lambda x: x['sharpe_ratio'], reverse=True)
        return table_data


class BacktestApp:
    def __init__(self, best_strats, comparison_data, symbol, port):
        self.best_strats = best_strats
        self.comparison_data = comparison_data
        self.symbol = symbol
        self.port = port

        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.create_layout()
        self.register_callbacks()

    def create_layout(self):
        self.app.layout = html.Div([
            html.H1(f"{self.symbol} Strategy Dashboard"),

            dcc.Loading(id="loading-output",
                        type="circle",
                        children=[
                            dcc.Graph(id="cumulative-returns-plot"),
                            dash_table.DataTable(id="strategy-table")
                        ]),
            html.Div(id="strategy-plots-container"),
            # Hidden div to trigger the callback
            html.Div(id="hidden-trigger", style={"display": "none"})
        ])

    def register_callbacks(self):
        @self.app.callback(
            [Output("cumulative-returns-plot", "figure"),
             Output("strategy-plots-container", "children"),
             Output("strategy-table", "data")],
            [Input("hidden-trigger", "children")]  # Using hidden div as input to trigger the callback
        )
        def generate_dashboard(_):
            try:
                dashboard = BacktestDashboard()
                cumulative_returns_fig = dashboard.plot_cumulative_returns(self.comparison_data['creturns'])
                positions_figs = dashboard.plot_positions(self.comparison_data['positions'])
                table_data = dashboard.create_table(self.best_strats)

                positions_plots = [dcc.Graph(figure=fig) for fig in positions_figs]

                return cumulative_returns_fig, positions_plots, table_data

            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")
                traceback.print_exc()
                return {}, [], []

    def run_server(self):
        serve(self.app.server, host='0.0.0.0', port=self.port)

    def open_browser(self):
        sleep(1)
        webbrowser.open(f"http://127.0.0.1:{self.port}")