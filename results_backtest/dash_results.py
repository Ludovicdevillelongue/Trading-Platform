import logging
import traceback
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
from waitress import serve
from dash import dash_table
import webbrowser
from time import sleep

class DashboardApp:
    def __init__(self,best_strats, comparison_data, symbol):

        self.best_strats=best_strats
        self.comparison_data=comparison_data
        self.symbol=symbol

        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.create_layout()
        self.register_callbacks()

    def create_layout(self):
        self.app.layout = html.Div([
            html.H1("Strategy Dashboard"),

            html.Label("Symbol"),
            dcc.Input(id="symbol", type="text", value=f"{self.symbol}"),

            html.Label("Start Date"),
            dcc.DatePickerSingle(id="start-date", date=self.comparison_data['returns'].index[0]),

            html.Label("End Date"),
            dcc.DatePickerSingle(id="end-date", date=self.comparison_data['returns'].index[1]),

            html.Button("Generate Dashboard", id="generate-dashboard"),

            dcc.Loading(id="loading-output",
                        type="circle",
                        children=[
                            dcc.Graph(id="cumulative-returns-plot"),
                            dash_table.DataTable(id="strategy-table")
                        ]),
            html.Div(id="strategy-plots-container")
        ])

    def register_callbacks(self):
        @self.app.callback(
            [Output("cumulative-returns-plot", "figure"),
             Output("strategy-plots-container", "children"),  # Container for multiple plots
             Output("strategy-table", "data")],
            [Input("generate-dashboard", "n_clicks")],
            [State("symbol", "value"),
             State("start-date", "date"),
             State("end-date", "date")]
        )
        def generate_dashboard(n_clicks, symbol, start_date, end_date):
            try:
                dashboard = Dashboard()
                cumulative_returns_fig = Dashboard.plot_cumulative_returns(self.comparison_data['returns'])
                positions_figs = Dashboard.plot_positions(self.comparison_data['positions'])
                table_data = Dashboard.create_table(self.best_strats)

                # Convert each figure to a Graph component and store in a list
                positions_plots = [dcc.Graph(figure=fig) for fig in positions_figs]

                return cumulative_returns_fig, positions_plots, table_data


            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")
                traceback.print_exc()
                # Log specific error message
                return {}, []


    def run_server(self):
        serve(self.app.server, host='0.0.0.0', port=8080)

    def open_browser(self):
        sleep(1)  # Short delay before opening the browser
        webbrowser.open("http://127.0.0.1:8080")

class Dashboard:
    @staticmethod
    def plot_cumulative_returns(returns):
        fig = px.line(returns, x=returns.index, y=returns.columns,
                      title='Cumulative Returns and Strategies Comparison')
        return fig

    @staticmethod
    def plot_positions(positions):
        # Initialize a list to store the figures
        figures = []

        # Iterate through each column in the DataFrame
        for column in positions.columns:
            # Generate a line plot for each column
            fig = px.line(positions,
                          x=positions.index,
                          y=column,  # Plot each column individually
                          title=f'{column}')  # Title for each column

            # Append the figure to the list
            figures.append(fig)

        return figures

    @staticmethod
    def create_table(best_strat_recap):
        table_data = []
        for strategy_name, strategy_result in best_strat_recap.items():
            table_data.append({'strategy_name': strategy_name,
                               'params': str(strategy_result['params']),
                               **strategy_result['results']})
            table_data = sorted(table_data, key=lambda x: x['sharpe_ratio'], reverse=True)
        return table_data





