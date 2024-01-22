import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def generic_plot_indicator(dates:pd.DataFrame, lst_indicators:list, lst_stategy_names:list,frequency:str,estimation_period:int,
                           fig_name:str,save_fig=False, path=None,
                 y_axis_name='Indicator Value', legend_name='Indicator Name', no_show=False)-> go:
    """
        Parameters
        ----------
            dates: pd.DataFrame -> dates taken into account in the plot
            lst_indicators : list -> list of chosen metrics to compute
            lst_stategy_names : list -> list of strategies to plot
            frequency : str -> Frequency of data (daily, hourly)
            estimation_period : int -> estimation period for the chosen strategies
            fig_name: str -> name of graph
            save_fig : bool -> option to save figure or not
            path : str -> path of the saved figure
            y_axis_name : str -> value of indicators
            legend_name : str -> name of indicators
            no_show : bool -> option to show figure
        Returns
        -------
            plot of selected indicators for the computed strategies
    """
    lines = []
    for name, indicator in zip(lst_stategy_names, lst_indicators):
        line = go.Scatter(x=dates[estimation_period+1:], y=indicator[estimation_period+1:],
                   mode='lines',
                   name=name)
        lines.append(line)

    if save_fig or (not no_show):
        fig = go.Figure()
        for line in lines:
            fig.add_trace(line)
        fig.update_layout(
            xaxis_title="Dates",
            yaxis_title=y_axis_name,
            legend_title=legend_name,
            title=fig_name)
        if frequency == "daily":
            fig.update_xaxes(
                title_text='Dates',
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label='1M', step='month', stepmode='backward'),
                        dict(count=6, label='6M', step='month', stepmode='backward'),
                        dict(count=1, label='YTD', step='year', stepmode='todate'),
                        dict(count=1, label='1Y', step='year', stepmode='backward'),
                        dict(step='all')])))
        elif frequency == "hourly":
            fig.update_xaxes(
                title_text='Dates',
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label='1h', step='hour', stepmode='backward'),
                        dict(count=12, label='12h', step='hour', stepmode='backward'),
                        dict(count=24, label='24h', step='hour', stepmode='backward'),
                        dict(count=48, label='48h', step='hour', stepmode='backward'),
                        dict(step='all')])))
        if save_fig:
            if path is None:
                raise Exception('Output path missing to save figure')
            fig.write_html(path)
        if not no_show:
            fig.show()
    return fig

def area_chart(data:pd.DataFrame,frequency:str,estimation_period:int,
                           fig_name:str,save_fig=False, path=None,
                 y_axis_name='Indicator Value', legend_name='Indicator Name', no_show=False):

    """
        Parameters
        ----------
            data: pd.DataFrame -> values with dates as index
            frequency : str -> Frequency of data (daily, hourly)
            estimation_period : int -> estimation period for the chosen strategies
            fig_name: str -> name of graph
            save_fig : bool -> option to save figure or not
            path : str -> path of the saved figure
            y_axis_name : str -> value of indicators
            legend_name : str -> name of indicators
            no_show : bool -> option to show figure
        Returns
        -------
            plot of selected indicators for the computed strategies
    """
    c_area = px.area(data.iloc[estimation_period+1:,:], title=fig_name)

    if frequency=="daily":
        c_area.update_xaxes(
            title_text='Dates',
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1M', step='month', stepmode='backward'),
                    dict(count=6, label='6M', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1Y', step='year', stepmode='backward'),
                    dict(step='all')])))
    elif frequency=="hourly":
        c_area.update_xaxes(
        title_text = 'Dates',
        rangeslider_visible = True,
        rangeselector = dict(
            buttons=list([
                dict(count=1, label='1h', step='hour', stepmode='backward'),
                dict(count=12, label='12h', step='hour', stepmode='backward'),
                dict(count=24, label='24h', step='hour', stepmode='backward'),
                dict(count=48, label='48h', step='hour', stepmode='backward'),
                dict(step='all')])))


    c_area.update_yaxes(title_text=y_axis_name, tickprefix='$')
    c_area.update_layout(showlegend=False,
                         title={
                             'text': legend_name,
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'})


    if save_fig:
        if path is None:
            raise Exception('Output path missing to save figure')
        c_area.write_html(path)
    if not no_show:
        c_area.show()

    return c_area