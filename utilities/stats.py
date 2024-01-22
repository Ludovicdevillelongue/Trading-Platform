
import numpy as np
from numba import njit


@njit
def quantile(p:float, u:np.array)-> np.array:
    """
        Parameters
        ----------
            p : float -> probability associated to the quantile
            u : np.array -> vector of realizations of the random variable

        Returns
        -------
            p-quantile of u
    """
    if p == 0:
        k = 1
    else:
        k = np.ceil(p * len(u))
    return _quickselect(k, u)


@njit
def superquantile(p:float, u:np.array)-> np.array:

    """
        Parameters
        ----------
           p : float -> probability associated to the superquantile
           u : np.array -> vector of realizations of the random variable

        Returns
        -------
            p-superquantile of u
    """

    if p == 0:
        return np.mean(u)

    elif p >= 1.0 - 1.0/len(u):
        return np.max(u)
    else:
        n = len(u)
        q_p = quantile(p, u)
        higher_data = np.extract(u > q_p, u)
        if len(higher_data) == 0:
            return q_p
        else:
            next_jump = (u <= q_p).sum() / n
            cvar_plus = np.mean(higher_data)
            lmbda = (next_jump - p) / (1.0 - p)
            return lmbda * q_p + (1.0 - lmbda) * cvar_plus


@njit
def _quickselect(k:int, list_of_numbers:list)-> np.array:
    """
        Parameters
        ----------
           k : int -> period selected
           list_of_numbers : list -> numbers selected

        Returns
        -------
            kthsmallest parameter
    """
    return _kthSmallest(list_of_numbers, k, 0, len(list_of_numbers) - 1)


@njit
def _kthSmallest(arr:np.array, k:int, start:int, end:int)-> np.array:
    """
        Parameters
        ----------
           arr:np.array -> array of data on which the quantile will be calculated
           k : int -> period selected
           start : int -> stat of period
           end : int -> end of period

        Returns
        -------
            kthsmallest parameter
    """

    pivot_index = _partition(arr, start, end)

    if pivot_index - start == k - 1:
        return arr[pivot_index]

    if pivot_index - start > k - 1:
        return _kthSmallest(arr, k, start, pivot_index - 1)

    return _kthSmallest(arr, k - pivot_index + start - 1, pivot_index + 1, end)


@njit
def _partition(arr:np.array, l: int, r: int)-> int:
    """
        Parameters
        ----------
            arr:np.array -> array of data on which the quantile will be calculated
            l : int -> first item of array
            r : int -> last item of array

        Returns
        -------
            partition of the data
    """
    pivot = arr[r]
    i = l
    for j in range(l, r):

        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    arr[i], arr[r] = arr[r], arr[i]
    return i
