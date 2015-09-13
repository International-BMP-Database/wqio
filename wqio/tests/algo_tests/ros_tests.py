import os

from nose.tools import *
import numpy as np
import numpy.testing as nptest

from wqio import testing
usetex = testing.compare_versions(utility='latex')

import matplotlib
matplotlib.rcParams['text.usetex'] = usetex

import pandas as pd

from wqio.algo import ros


def test_rosSort():
    raw_data = testing.getTestROSData()
    data = ros.rosSort(raw_data)
    np.unique(data.res[data.qual == 'ND'])

    # masked/not masked indices
    masked = data[data.qual == 'ND']
    not_masked = data[data.qual != 'ND']

    val = -9999.9
    for d in masked.res:
        assert_greater_equal(d, val)
        val = d

    val = -9999.9
    for d in not_masked.res:
        assert_greater_equal(d, val)
        val = d
    pass


class _baseMR_Mixin:
    @nottest
    def makePath(self, filename):
        return os.path.join(self.prefix, filename)

    @raises(ValueError)
    def test_zero_data(self):
        data = self.data.copy()
        data.loc[0, self.mr.rescol] = 0.0
        mr = ros.MR(data, rescol=self.mr.rescol, qualcol=self.mr.qualcol, ndsymbol=self.mr.ndsymbol)

    @raises(ValueError)
    def test_negative_data(self):
        data = self.data.copy()
        data.loc[0, self.mr.rescol] = -1.0
        mr = ros.MR(data, rescol=self.mr.rescol, qualcol=self.mr.qualcol, ndsymbol=self.mr.ndsymbol)

    def test_data(self):
        assert_true(hasattr(self.mr, 'data'))
        assert_is_instance(self.mr.data, pd.DataFrame)
        assert_list_equal(self.mr.data.columns.tolist(), self.known_data_cols)

    def test_debug_attr(self):
        assert_true(hasattr(self.mr, 'debug'))
        assert_is_instance(self.mr.data, pd.DataFrame)
        assert_list_equal(self.mr.debug.columns.tolist(), self.known_debug_cols)

    def test_N_tot(self):
        assert_true(hasattr(self.mr, 'N_tot'))
        assert_equal(self.mr.N_tot, self.data.shape[0])

    def test_N_nd(self):
        assert_true(hasattr(self.mr, 'N_nd'))
        assert_equal(self.mr.N_nd, self.data[self.data[self.mr.qualcol] == self.ndsymbol].shape[0])

    def test_DLs_attr(self):
        assert_true(hasattr(self.mr, 'DLs'))
        assert_equal(type(self.mr.DLs), pd.DataFrame)

    def test_DLs_values(self):
        nptest.assert_array_equal(self.mr.DLs.DL[:-1], np.array([2., 5., 5.5, 5.75, 9.5, 11.]))

    def test_MR_ABC(self):
        known_abc = (0.0, 5.0, 2.0)
        abc = (self.mr.DLs.A[1], self.mr.DLs.B[1], self.mr.DLs.C[1])
        assert_tuple_equal(abc, known_abc)

    def test_MR_rosEstimator(self):
        known_z = np.array([
            -2.06188401, -1.66883254, -1.4335397, -1.25837339, -1.11509471,
            -0.99166098, -0.8817426, -0.78156696, -0.68868392, -0.60139747,
            -0.51847288, -0.4389725, -0.36215721, -0.28742406, -0.21426459,
            -0.14223572, -0.07093824, 0.,         0.07093824,  0.14223572,
            0.21426459,  0.28742406,  0.36215721,  0.4389725,  0.51847288,
            0.60139747,  0.68868392,  0.78156696,  0.8817426,  0.99166098,
            1.11509471,  1.25837339,  1.4335397,  1.66883254,  2.06188401
        ])

        known_fd = np.array([
            2., 3.11029054, 3.60383412, 4.04355908, 4.04355908, 4.2, 4.62,
            4.70773991, 5.57, 5.66, 5.86, 6.13826881, 6.65, 6.78, 6.79,
            6.97698797, 7.5, 7.5, 7.5, 8.63, 8.71, 8.99, 9.85, 10.82, 11.25,
            11.25, 12.2, 14.92, 16.77, 17.81, 19.16, 19.19, 19.64, 20.18, 22.97
        ])

        fdarray = np.array(self.mr.data.final_data)
        fdarray.sort()
        nptest.assert_array_almost_equal(known_fd, fdarray, decimal=2)


    @raises(ValueError)
    def test_dup_index_error(self):
        data = self.data.append(self.data)
        mr = ros.MR(data, rescol=self.mr.rescol, qualcol=self.mr.qualcol, ndsymbol=self.mr.ndsymbol)

    @raises(ValueError)
    def test_non_dataframe_error(self):
        data = self.data.values
        mr = ros.MR(data, rescol=self.mr.rescol, qualcol=self.mr.qualcol, ndsymbol=self.mr.ndsymbol)


class test_MR_baseline(_baseMR_Mixin):
    def setup(self):
        self.ndsymbol = 'ND'
        self.data = testing.getTestROSData()
        self.mr = ros.MR(self.data, ndsymbol=self.ndsymbol)
        self.known_data_cols = ['final_data', 'res', 'qual']
        self.known_debug_cols = ['res', 'qual', 'DLIndex', 'Norm Ranks', 'Avg Ranks',
                                 'plot_pos', 'Zprelim', 'modeled_data', 'final_data']


class test_MR_weirdNDsymbol(_baseMR_Mixin):
    def setup(self):
        self.ndsymbol = '<<'
        self.data = testing.getTestROSData()
        self.data['qual'] = self.data['qual'].str.replace('ND', self.ndsymbol)
        self.mr = ros.MR(self.data, ndsymbol=self.ndsymbol)
        self.known_data_cols = ['final_data', 'res', 'qual']
        self.known_debug_cols = ['res', 'qual', 'DLIndex', 'Norm Ranks', 'Avg Ranks',
                                 'plot_pos', 'Zprelim', 'modeled_data', 'final_data']


class test_MR_weirdEverything(_baseMR_Mixin):
    def setup(self):
        self.ndsymbol = '<<'
        self.data = testing.getTestROSData().rename(columns={'res': 'conc', 'qual': 'cen'})
        self.data['cen'] = self.data['cen'].str.replace('ND', self.ndsymbol)
        self.mr = ros.MR(self.data, ndsymbol=self.ndsymbol, rescol='conc', qualcol='cen')
        self.known_data_cols = ['final_data', 'conc', 'cen']
        self.known_debug_cols = ['conc', 'cen', 'DLIndex', 'Norm Ranks', 'Avg Ranks',
                                 'plot_pos', 'Zprelim', 'modeled_data', 'final_data']

