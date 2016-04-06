from __future__ import print_function, division
import warnings

import numpy
from scipy import stats
import pandas

from wqio.utils import figutils


def _ros_sort(df, result='res', censorship='cen'):
    """
    This function prepares a dataframe for ROS. It sorts ascending with
    left-censored observations on top. Censored results larger than the
    maximum uncensored results are removed from the dataframe.

    Parameters
    ----------
    dataframe : a pandas dataframe with results and qualifiers.
        The qualifiers of the dataframe must have two states:
        detect and non-detect.
    result (default = 'res') : name of the column in the dataframe
        that contains result values.
    censorship (default = 'cen') : name of the column in the dataframe
        that indicates that a result is left-censored.
        (i.e., True -> censored, False -> uncensored)

    Output
    ------
    Sorted pandas DataFrame.

    """

    # separate uncensored data from censored data
    censored = df[df[censorship]].sort_values(by=result)
    uncensored = df[~df[censorship]].sort_values(by=result)

    if censored[result].max() > uncensored[result].max():
        msg = (
            "Dropping censored results greater than "
            "the max uncensored result."
        )
        warnings.warn(msg)
        censored = censored[censored[result] <= uncensored[result].max()]

    return censored.append(uncensored)[[result, censorship]].reset_index(drop=True)


def cohn_numbers(df, result='res', censorship='cen'):
    """
    Computes the Cohn numbers for the detection limits in the
    dataset.

    The Cohn Numbers are:
        + A_j = the number of uncensored obs above the jth threshold.
        + B_j = the number of observations (cen & uncen) below the
          jth threshold.
        + C_j = the number of censored observations at the jth
          threshold.
        + PE_j = the probability of exceeding the jth threshold
        + detection_limit = unique detection limits in the dataset.
        + lower -> a copy of the detection_limit column
        + upper -> lower shifted down 1 step

    """

    def nuncen_above(row):
        """
        The number of uncensored obs above the given threshold (A_j).

        """

        # index of results above the lower DL
        above = df[result] >= row['lower']

        # index of results below the upper DL
        below = df[result] < row['upper']

        # index of non-detect results
        detect = df[censorship] == False

        # return the number of results where all conditions are True
        return df[above & below & detect].shape[0]

    def nobs_below(row):
        """
        The number of observations (cen & uncen) below the given
        threshold (B_j).

        """

        # index of data less than the lower DL
        less_than = df[result] < row['lower']

        # index of data less than or equal to the lower DL
        less_thanequal = df[result] <= row['lower']

        # index of detects, non-detects
        uncensored = df[censorship] == False
        censored = df[censorship] == True

        # number results less than or equal to lower DL and non-detect
        LTE_censored = df[less_thanequal & censored].shape[0]

        # number of results less than lower DL and detected
        LT_uncensored = df[less_than & uncensored].shape[0]

        # return the sum
        return LTE_censored + LT_uncensored

    def ncen_equal(row):
        """
        The number of censored observations at the given threshold
        (C_j).

        """

        censored_index = df[censorship]
        censored_data = df[result][censored_index]
        censored_below = censored_data == row['lower']
        return censored_below.sum()

    def set_upper_limit(cohn):
        if cohn.shape[0] > 1:
            return cohn['DL'].shift(-1).fillna(value=numpy.inf)
        else:
            return [numpy.inf]

    def compute_PE(A, B):
        N = len(A)
        PE = numpy.empty(N, dtype='float64')
        PE[-1] = 0.0
        for j in range(N-2, -1, -1):
            PE[j] = PE[j+1] + (1 - PE[j+1]) * A[j] / (A[j] + B[j])

        return PE


    # unique values
    censored_data = df[censorship]
    cohn = pandas.unique(df.loc[censored_data, result])

    # if there is a results smaller than the minimum detection limit,
    # add that value to the array
    if cohn.shape[0] > 0:
        if df[result].min() < cohn.min():
            cohn = numpy.hstack([df[result].min(), cohn])

        # create a dataframe
        cohn = (
            pandas.DataFrame(cohn, columns=['DL'])
                .assign(lower=lambda df: df['DL'])
                .assign(upper=lambda df: set_upper_limit(df))
                .assign(nuncen_above=lambda df: df.apply(nuncen_above, axis=1))
                .assign(nobs_below=lambda df: df.apply(nobs_below, axis=1))
                .assign(ncen_equal=lambda df: df.apply(ncen_equal, axis=1))
                .reindex(range(cohn.shape[0] + 1))
                .assign(prob_exceedance=lambda df: compute_PE(df['nuncen_above'], df['nobs_below']))
        )

    else:
        dl_cols = ['DL', 'lower', 'upper', 'nuncen_above',
                   'nobs_below', 'ncen_equal', 'prob_exceedance']
        cohn = pandas.DataFrame(numpy.empty((0,7)), columns=dl_cols)

    return cohn


def _detection_limit_index(res, cohn):
    """
    Helper function to create an array of indices for the
    detection  limits (cohn) corresponding to each
    data point.

    """

    if cohn.shape[0] > 0:
        index, = numpy.where(cohn['DL'] <= res)
        det_limit_index = index[-1]
    else:
        det_limit_index = 0

    return det_limit_index


def _ros_group_rank(df, groupcols):
    ranks = (
        df.assign(rank=1)
          .groupby(by=groupcols)['rank']
          .transform(lambda g: g.cumsum())
    )
    return ranks


def _ros_plot_pos(row, cohn, censorship='cen'):
    """
    Helper function to compute the ROS'd plotting position.

    """

    DL_index = row['det_limit_index']
    rank = row['rank']
    censored = row[censorship]

    dl_1 = cohn.iloc[DL_index]
    dl_2 = cohn.iloc[DL_index + 1]
    if censored:
        return (1 - dl_1['prob_exceedance']) * rank / (dl_1['ncen_equal']+1)
    else:
        return (1 - dl_1['prob_exceedance']) + (dl_1['prob_exceedance'] - dl_2['prob_exceedance']) * \
                rank / (dl_1['nuncen_above']+1)


