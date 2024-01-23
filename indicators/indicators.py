from abc import abstractmethod
import numpy as np
import time

class Indicator:

    @abstractmethod
    def get_metric(self, day:int):
        """
           Parameters
           ----------
                day: int -> day on which to compute metric
        """
        raise NotImplementedError


class AssetsIndicator(Indicator):
    # returns metrics for a given time on an array of assets
    #    row_index is time (days for example)
    #    column_index is an asset
    @abstractmethod
    def get_metric(self, time:int, data_array:np.array):
        """
           Parameters
           ----------
                time: int -> time as row_index
                data_array : np.array -> array of returns of a portfolio
        """
        raise NotImplementedError


class PerformanceIndicator(Indicator):
    @abstractmethod
    def get_metric(self, data_array:np.array):
        """
           Parameters
           ----------
                data_array : np.array -> array of returns of a portfolio
        """
        return NotImplementedError


class RollingIndicator(Indicator):
    @abstractmethod
    def get_metric(self, data_array:np.array, **kwargs):
        """
           Parameters
           ----------
                data_array : np.array -> array of returns of a portfolio
        """
        return NotImplementedError

    def get_rolling_metric(self, data_array:np.array, period:int, **kwargs)-> np.array:
        """
              Parameters
              ----------
                  data_array : np.array -> array of returns of a portfolio
                  period : int -> period on which results are computed

              Returns
              ----------
                  array of rolling metric
        """
        data_len = len(data_array)
        res = np.empty(data_len)
        for ii in range(period, data_len):
            res[ii] = self.get_metric(data_array[ii-period:ii], **kwargs)
        res[0:period] = res[period] * np.ones(period)
        return res
