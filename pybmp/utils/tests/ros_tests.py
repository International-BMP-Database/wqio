import os

from nose.tools import *
import numpy as np
import numpy.testing as nptest

import testing
usetex = testing.compare_versions(utility='latex')

import matplotlib
matplotlib.rcParams['text.usetex'] = usetex

import pandas as pd

from .. import ros


def test_getTestData():
    known_data = np.array([
        2., 4.2, 4.62, 5., 5., 5.5, 5.57, 5.66, 5.75, 5.86, 6.65, 6.78, 6.79,
        7.5, 7.5, 7.5, 8.63, 8.71, 8.99, 9.50, 9.50, 9.85, 10.82, 11.00, 11.25,
        11.25, 12.2, 14.92, 16.77, 17.81, 19.16, 19.19, 19.64, 20.18, 22.97
    ])
    known_qual = np.array([
        '=', '=', '=', 'ND', 'ND', 'ND', '=', '=', 'ND', '=', '=', '=',
        '=', '=', '=', '=', '=', '=', '=', 'ND', 'ND', '=', '=', 'ND', '=',
        '=', '=', '=', '=', '=', '=', '=', '=', '=', '='
    ])

    test_data = ros.getTestData()

    nptest.assert_array_almost_equal(known_data, np.array(test_data.res))
    nptest.assert_array_equal(known_qual, np.array(test_data.qual))
    pass


def test_rosSort():
    raw_data = ros.getTestData()
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
        data['res'].loc[0] = 0.0
        mr = ros.MR(data)

    @raises(ValueError)
    def test_negative_data(self):
        data = self.data.copy()
        data['res'].loc[0] = -1.0
        mr = ros.MR(data)

    def test_data(self):
        assert_true(hasattr(self.mr, 'data'))
        assert_is_instance(self.mr.data, pd.DataFrame)

    def test_data_cols(self):
        assert_list_equal(self.mr.data.columns.tolist(),
                          ['final_data', 'res', 'qual'])

    def test_debug_attr(self):
        assert_true(hasattr(self.mr, 'debug'))
        assert_is_instance(self.mr.data, pd.DataFrame)

    def test_debug_cols(self):
        assert_list_equal(self.mr.debug.columns.tolist(),
                          ['res', 'qual', 'DLIndex', 'Norm Ranks', 'Avg Ranks',
                           'plot_pos', 'Zprelim', 'modeled_data', 'final_data'])

    def test_N_tot(self):
        assert_true(hasattr(self.mr, 'N_tot'))
        assert_equal(self.mr.N_tot, self.data.shape[0])

    def test_N_nd(self):
        assert_true(hasattr(self.mr, 'N_nd'))
        assert_equal(self.mr.N_nd, self.data[self.data.qual == self.ndsymbol].shape[0])

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

        # zarray = np.array(self.mr.data.Z)
        # zarray.sort()

        # nptest.assert_array_almost_equal(known_z, zarray, decimal=2)
        nptest.assert_array_almost_equal(known_fd, fdarray, decimal=2)

    #@nptest.dec.skipif(not usetex)
    #def test_MR_plot(self):
    #    assert_true(hasattr(self.mr, 'plot'))
    #    self.mr.plot(self.makePath('test_plot.png'))

    @raises(ValueError)
    def test_dup_index_error(self):
        data = self.data.append(self.data)
        mr = ros.MR(data)

    @raises(ValueError)
    def test_non_dataframe_error(self):
        data = self.data.values
        mr = ros.MR(data)


class test_MR_std(_baseMR_Mixin):
    def setup(self):
        self.ndsymbol = 'ND'
        self.data = ros.getTestData()
        self.mr = ros.MR(self.data, ndsymbol=self.ndsymbol)
        if os.path.split(os.getcwd())[-1] == 'src':
            self.prefix = os.path.join('.', 'utils', 'tests', 'result_images')
        else:
            self.prefix = os.path.join('..', 'utils', 'tests', 'result_images')


class test_MR_weirdNDsymbol(_baseMR_Mixin):
    def setup(self):
        self.ndsymbol = '<<'
        self.data = ros.getTestData()
        self.data.qual[self.data.qual == 'ND'] = self.ndsymbol
        self.mr = ros.MR(self.data, ndsymbol=self.ndsymbol)
        if os.path.split(os.getcwd())[-1] == 'src':
            self.prefix = os.path.join('.', 'utils', 'tests', 'result_images')
        else:
            self.prefix = os.path.join('..', 'utils', 'tests', 'result_images')


class junk_arrayMR:
    def setup(self):
        self.data = ros.getTestData()
        self.mr = ros.arrayMR(self.data)

    def test_data(self):
        assert_true(hasattr(self.mr, 'data'))
        assert_is_instance(self.mr.data, pd.DataFrame)
        assert_list_equal(self.mr.data.columns.tolist(),
                          ['res', 'qual', 'DLIndex', 'nranks', 'aranks'])

    def test_N(self):
        assert_true(hasattr(self.mr, 'N'))
        assert_equal(self.mr.N, self.data.shape[0])

    def test_DLs(self):
        assert_true(hasattr(self.mr, 'DLs'))
        assert_equal(type(self.mr.DLs), np.ndarray)
        assert_less(self.mr.DLs.shape[0], self.mr.data.shape[0])
        assert_equal(len(self.mr.DLs.shape), 1)
        nptest.assert_array_equal(self.mr.DLs, np.array([2., 5., 5.5, 5.75, 9.5, 11.]))

    def test_DLIndex(self):
        assert_true(hasattr(self.mr, 'DLIndex'))
        assert_is_instance(self.mr.DLIndex, np.ndarray)
        assert_equal(self.mr.DLIndex.shape[0], self.mr.data.shape[0])
        assert_equal(len(self.mr.DLIndex.shape), 1)

        known_DLIndex = np.array([
            1, 1, 2, 3, 4, 4, 5, 0, 0, 0, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5
        ])
        nptest.assert_array_equal(known_DLIndex, self.mr.DLIndex)

    def test_nranks(self):
        assert_true(hasattr(self.mr, 'nranks'))
        assert_is_instance(self.mr.nranks, np.ndarray)
        assert_equal(self.mr.nranks.shape[0], self.mr.data.shape[0])
        assert_equal(len(self.mr.nranks.shape), 1)

        known_nranks = np.array([
            1.,   2.,   1.,   1.,   1.,   2.,   1.,   1.,   2.,   3.,   1.,
            2.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
            1.,   2.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,
            10.,  11.
        ])
        nptest.assert_array_equal(known_nranks, self.mr.nranks)

    def test_aranks(self):
        assert_true(hasattr(self.mr, 'aranks'))
        assert_is_instance(self.mr.aranks, np.ndarray)
        assert_equal(self.mr.aranks.shape[0], self.mr.data.shape[0])
        assert_equal(len(self.mr.aranks.shape), 1)

        known_aranks = np.array([
            1.,   2.,   1.,   1.,   1.,   2.,   1.,   1.,   2.,
            3.,   1.,   2.,   1.,   2.,   3.,   4.,   6.,   6.,
            6.,   8.,   9.,  10.,   1.,   2.,   1.5,  1.5,  3.,
            4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.
        ])
        nptest.assert_array_equal(known_aranks, self.mr.aranks)

    def test_plot_pos(self):
        assert_true(hasattr(self.mr, 'plot_pos'))
        assert_is_instance(self.mr.plot_pos, np.ndarray)
        assert_equal(self.mr.plot_pos.shape[0], self.mr.data.shape[0])
        assert_equal(len(self.mr.plot_pos.shape), 1)

    def test_Z(self):
        assert_true(hasattr(self.mr, 'Z'))
        assert_is_instance(self.mr.Z, np.ndarray)
        assert_equal(self.mr.Z.shape[0], self.mr.data.shape[0])
        assert_equal(len(self.mr.Z.shape), 1)

    def test_final_data(self):
        assert_true(hasattr(self.mr, 'final_data'))
        assert_is_instance(self.mr.final_data, np.ndarray)
        assert_equal(self.mr.final_data.shape[0], self.mr.data.shape[0])
        assert_equal(len(self.mr.final_data.shape), 1)

    def test_MR_ABC(self):
        known_abc = (0.0, 5.0, 2.0)
        abc = self.mr._compute_ABC(self.mr.DLs[1:3])
        assert_tuple_equal(abc, known_abc)

    @nptest.dec.skipif(True)
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

        fdarray = np.array(self.mr.final_data)
        fdarray.sort()

        zarray = np.array(self.mr.Z)
        zarray.sort()

        nptest.assert_array_almost_equal(known_z, zarray)
        nptest.assert_array_almost_equal(known_fd, fdarray)
