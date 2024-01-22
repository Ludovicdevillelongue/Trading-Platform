import numpy as np
import pandas as pd


def compute_returns(c:pd.DataFrame)-> np.array:
    """
        Parameters
        ----------
            c: pd.DataFrame -> close prices
        Returns
        -------
            price returns
    """
    res = np.empty_like(c)
    res[0] = 1.
    res[1:] = (c[1:] - c[:-1]) / c[:-1]

    return res
