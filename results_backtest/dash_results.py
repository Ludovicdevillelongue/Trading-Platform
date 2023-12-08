import logging
from datetime import datetime, timedelta
import traceback
import dash
import pytz
from dash import dcc, html, Input, Output, State
import plotly.express as px
from dash import dash_table
from backtester.strat_creator import (
    SMAVectorBacktester, LRVectorBacktester, MomVectorBacktester, MRVectorBacktester, TurtleVectorBacktester,
ScikitVectorBacktester)
from backtester.strat_comparator import StrategyRunner

class DashboardApp:
    def __init__(self):
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.create_layout()
        self.register_callbacks()

    def create_layout(self):
        self.app.layout = html.Div([
            html.H1("Strategy Dashboard"),

            html.Label("Symbol"),
            dcc.Input(id="symbol", type="text", value="IBM"),

            html.Label("Start Date"),
            dcc.DatePickerSingle(id="start-date", date="2020-01-01"),

            html.Label("End Date"),
            dcc.DatePickerSingle(id="end-date", date="2022-12-31"),

            html.Button("Generate Dashboard", id="generate-dashboard"),

            dcc.Loading(id="loading-output",
                        type="circle",
                        children=[
                            dcc.Graph(id="cumulative-returns-plot"),
                            dash_table.DataTable(id="strategy-table")  # Use DataTable from dash_table
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
                strategies = {
                    'SMA': SMAVectorBacktester, 'MOM': MomVectorBacktester,
                    'MeanRev': MRVectorBacktester, 'Turtle': TurtleVectorBacktester, 'LinearReg': LRVectorBacktester}
                    # 'ScikitReg':ScikitVectorBacktester}

                # 'LinearReg': LRVectorBacktester,
                #     'ScikitReg': ScikitVectorBacktester}
                param_grids = {
                    'SMA': {'sma_short': (5, 30), 'sma_long': (31, 100)},
                    'MOM': {'momentum': (10, 100)},
                    'MeanRev': {'sma': (5, 50), 'threshold': (0.3, 0.7)},
                    'Turtle': {'window_size': (20, 50)},
                    'LinearReg': {'lags': (3,10), 'train_percent': (0.7, 0.8)},
                    # 'ScikitReg': {'lags': (3, 10), 'train_percent': (0.7, 0.8), 'model': ['logistic']}
                }
                symbol = 'SYM'
                start_date = '2023-11-15 00:00:00'
                end_date = (
                    (datetime.now(pytz.timezone('US/Eastern')) - timedelta(minutes=2)).replace(second=0)).strftime(
                    "%Y-%m-%d %H:%M:%S")
                amount = 10000
                transaction_costs = 0.01
                iterations = 10

                # Run the comparison and optimization
                runner = StrategyRunner(strategies, symbol, start_date, end_date, param_grids, amount,
                                        transaction_costs, iterations)
                logging.info("Optimizing strategies...")
                optimization_results = runner.test_all_search_types()
                logging.info("Optimized results: %s", optimization_results)
                logging.info("\nRunning and comparing strategies...")
                best_strat_recap, comparison_data = runner.run_and_compare_strategies()
                dashboard = Dashboard()
                cumulative_returns_fig = Dashboard.plot_cumulative_returns(comparison_data['returns'])
                positions_figs = Dashboard.plot_positions(comparison_data['positions'])
                table_data = Dashboard.create_table(best_strat_recap)

                # Convert each figure to a Graph component and store in a list
                positions_plots = [dcc.Graph(figure=fig) for fig in positions_figs]

                return cumulative_returns_fig, positions_plots, table_data


            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")
                traceback.print_exc()
                # Log specific error message
                return {}, []

    def run(self):
        self.app.run_server(debug=True)


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
            table_data.append({
                'strategy': strategy_name,
                'absolute_perf': strategy_result['aperf'],
                'over_perf': strategy_result['operf'],
                'sharpe_ratio': strategy_result['sharpe_ratio']
            })
        return table_data


if __name__ == '__main__':
    # Configure logging level (INFO for normal operation)
    logging.basicConfig(level=logging.INFO)

    app = DashboardApp()
    app.run()


