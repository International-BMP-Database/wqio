import sys
import nose.tools as ntools

import numpy as np
import numpy.testing as npt
from numpy.testing import dec
import pandas.util.testing as pdtest
from wqio import testing

import pandas as pd

from wqio.algo import fros
from statsmodels.compat.python import StringIO

try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except:
    have_matplotlib = False



@ntools.nottest
def load_basic_data():
    df = (
        testing
            .getTestROSData()
            .assign(conc=lambda df: df['res'])
            .assign(censored=lambda df: df['qual'] == 'ND')
    )
    return df


class Test__ros_sort(object):
    def setup(self):
        self.df = load_basic_data()

        self.expected_baseline = pd.DataFrame([
            {'censored': True,  'conc': 5.0},   {'censored': True,  'conc': 5.0},
            {'censored': True,  'conc': 5.5},   {'censored': True,  'conc': 5.75},
            {'censored': True,  'conc': 9.5},   {'censored': True,  'conc': 9.5},
            {'censored': True,  'conc': 11.0},  {'censored': False, 'conc': 2.0},
            {'censored': False, 'conc': 4.2},   {'censored': False, 'conc': 4.62},
            {'censored': False, 'conc': 5.57},  {'censored': False, 'conc': 5.66},
            {'censored': False, 'conc': 5.86},  {'censored': False, 'conc': 6.65},
            {'censored': False, 'conc': 6.78},  {'censored': False, 'conc': 6.79},
            {'censored': False, 'conc': 7.5},   {'censored': False, 'conc': 7.5},
            {'censored': False, 'conc': 7.5},   {'censored': False, 'conc': 8.63},
            {'censored': False, 'conc': 8.71},  {'censored': False, 'conc': 8.99},
            {'censored': False, 'conc': 9.85},  {'censored': False, 'conc': 10.82},
            {'censored': False, 'conc': 11.25}, {'censored': False, 'conc': 11.25},
            {'censored': False, 'conc': 12.2},  {'censored': False, 'conc': 14.92},
            {'censored': False, 'conc': 16.77}, {'censored': False, 'conc': 17.81},
            {'censored': False, 'conc': 19.16}, {'censored': False, 'conc': 19.19},
            {'censored': False, 'conc': 19.64}, {'censored': False, 'conc': 20.18},
            {'censored': False, 'conc': 22.97},
        ])[['conc', 'censored']]

        self.expected_with_warning = self.expected_baseline.iloc[:-1]

    def test_baseline(self):
        result = fros._ros_sort(self.df, result='conc', censorship='censored')
        pdtest.assert_frame_equal(result, self.expected_baseline)

    def test_censored_greater_than_max(self):
        df = self.df.copy()
        max_row = df['conc'].argmax()
        df.loc[max_row, 'censored'] = True
        result = fros._ros_sort(df, result='conc', censorship='censored')
        pdtest.assert_frame_equal(result, self.expected_with_warning)


class Test_cohn_numbers(object):
    def setup(self):
        self.df = load_basic_data()
        self.final_cols = ['DL', 'lower', 'upper', 'nuncen_above', 'nobs_below',
                           'ncen_equal', 'prob_exceedance']

        self.expected_baseline = pd.DataFrame([
            {'DL': 2.0, 'lower': 2.0, 'ncen_equal': 0.0, 'nobs_below': 0.0,
             'nuncen_above': 3.0, 'prob_exceedance': 1.0, 'upper': 5.0},
            {'DL': 5.0, 'lower': 5.0, 'ncen_equal': 2.0, 'nobs_below': 5.0,
             'nuncen_above': 0.0, 'prob_exceedance': 0.77757437070938218, 'upper': 5.5},
            {'DL': 5.5, 'lower': 5.5, 'ncen_equal': 1.0, 'nobs_below': 6.0,
             'nuncen_above': 2.0, 'prob_exceedance': 0.77757437070938218, 'upper': 5.75},
            {'DL': 5.75, 'lower': 5.75, 'ncen_equal': 1.0, 'nobs_below': 9.0,
             'nuncen_above': 10.0, 'prob_exceedance': 0.7034324942791762, 'upper': 9.5},
            {'DL': 9.5, 'lower': 9.5, 'ncen_equal': 2.0, 'nobs_below': 21.0,
             'nuncen_above': 2.0, 'prob_exceedance': 0.37391304347826088, 'upper': 11.0},
            {'DL': 11.0, 'lower': 11.0, 'ncen_equal': 1.0, 'nobs_below': 24.0,
             'nuncen_above': 11.0, 'prob_exceedance': 0.31428571428571428, 'upper': np.inf},
            {'DL': np.nan, 'lower': np.nan, 'ncen_equal': np.nan, 'nobs_below': np.nan,
             'nuncen_above': np.nan, 'prob_exceedance': 0.0, 'upper': np.nan}
        ])[self.final_cols]

    def test_baseline(self):
        result = fros.cohn_numbers(self.df, result='conc', censorship='censored')
        pdtest.assert_frame_equal(result, self.expected_baseline)

    def test_no_NDs(self):
        result = fros.cohn_numbers(self.df.assign(qual=False), result='conc', censorship='qual')
        ntools.assert_tuple_equal(result.shape, (0, 7))


class Test__detection_limit_index(object):
    def setup(self):
        self.cohn = pd.DataFrame([
            {'DL': 2.0, 'lower': 2.0, 'ncen_equal': 0.0, 'nobs_below': 0.0,
             'nuncen_above': 3.0, 'prob_exceedance': 1.0, 'upper': 5.0},
            {'DL': 5.0, 'lower': 5.0, 'ncen_equal': 2.0, 'nobs_below': 5.0,
             'nuncen_above': 0.0, 'prob_exceedance': 0.77757437070938218, 'upper': 5.5},
            {'DL': 5.5, 'lower': 5.5, 'ncen_equal': 1.0, 'nobs_below': 6.0,
             'nuncen_above': 2.0, 'prob_exceedance': 0.77757437070938218, 'upper': 5.75},
            {'DL': 5.75, 'lower': 5.75, 'ncen_equal': 1.0, 'nobs_below': 9.0,
             'nuncen_above': 10.0, 'prob_exceedance': 0.7034324942791762, 'upper': 9.5},
            {'DL': 9.5, 'lower': 9.5, 'ncen_equal': 2.0, 'nobs_below': 21.0,
             'nuncen_above': 2.0, 'prob_exceedance': 0.37391304347826088, 'upper': 11.0},
            {'DL': 11.0, 'lower': 11.0, 'ncen_equal': 1.0, 'nobs_below': 24.0,
             'nuncen_above': 11.0, 'prob_exceedance': 0.31428571428571428, 'upper': np.inf},
            {'DL': np.nan, 'lower': np.nan, 'ncen_equal': np.nan, 'nobs_below': np.nan,
             'nuncen_above': np.nan, 'prob_exceedance': 0.0, 'upper': np.nan}
        ])

        self.empty_cohn = pd.DataFrame(np.empty((0, 7)))

    def test_empty(self):
        ntools.assert_equal(fros._detection_limit_index(None, self.empty_cohn), 0)

    def test_populated(self):
         ntools.assert_equal(fros._detection_limit_index(3.5, self.cohn), 0)
         ntools.assert_equal(fros._detection_limit_index(6.0, self.cohn), 3)
         ntools.assert_equal(fros._detection_limit_index(12.0, self.cohn), 5)

    @ntools.raises(IndexError)
    def test_out_of_bounds(self):
        fros._detection_limit_index(0, self.cohn)


def test__ros_group_rank():
    df = pd.DataFrame({
        'params': list('AABCCCDE') + list('DCBA'),
        'values': list(range(12))
    })

    result = fros._ros_group_rank(df, ['params'])
    expected = pd.Series([1, 2, 1, 1, 2, 3, 1, 1, 2, 4, 2, 3], name='rank')
    pdtest.assert_series_equal(result, expected)

