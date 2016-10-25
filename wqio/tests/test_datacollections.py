from textwrap import dedent
from io import StringIO

import numpy
import pandas

import pytest
import pandas.util.testing as pdtest
from wqio.tests import helpers

from wqio.features import Location, Dataset
from wqio.datacollections import DataCollection


class _base_DataCollecionMixin(object):
    known_rescol = 'ros_res'
    known_raw_rescol = 'res'
    known_roscol = 'ros_res'
    known_qualcol = 'qual'
    known_stationcol = 'loc'
    known_paramcol = 'param'
    known_ndval = 'ND'
    expected_comp_index = pandas.MultiIndex.from_tuples([
        ('A', 'Inflow', 'Outflow'),
        ('A', 'Inflow', 'Reference'),
        ('A', 'Outflow', 'Inflow'),
        ('A', 'Outflow', 'Reference'),
        ('A', 'Reference', 'Inflow'),
        ('A', 'Reference', 'Outflow'),
        ('B', 'Inflow', 'Outflow'),
        ('B', 'Inflow', 'Reference'),
        ('B', 'Outflow', 'Inflow'),
        ('B', 'Outflow', 'Reference'),
        ('B', 'Reference', 'Inflow'),
        ('B', 'Reference', 'Outflow'),
    ], names=['param', 'loc_1', 'loc_2'])

    def prep_comp_result(self, df):
        return df.query("param in ['A', 'B']")

    def prep_comp_expected(self, data):
        return pandas.DataFrame(data, index=self.expected_comp_index)

    def test__raw_rescol(self):
        assert self.dc._raw_rescol == self.known_raw_rescol

    def test_data(self):
        assert isinstance(self.dc.data, pandas.DataFrame)

    def test_roscol(self):
        assert self.dc.roscol == self.known_roscol

    def test_rescol(self):
        assert self.dc.rescol == self.known_rescol

    def test_qualcol(self):
        assert self.dc.qualcol == self.known_qualcol

    def test_stationncol(self):
        assert self.dc.stationcol == self.known_stationcol

    def test_paramcol(self):
        assert self.dc.paramcol == self.known_paramcol

    def test_ndval(self):
        assert self.dc.ndval == [self.known_ndval]

    def test_bsiter(self):
        assert self.dc.bsiter == self.known_bsiter

    def test_groupby(self):
        assert self.dc.groupcols == self.known_groupcols

    def test_columns(self):
        assert self.dc.columns == self.known_columns

    def test_filterfxn(self):
        assert hasattr(self.dc, 'filterfxn')

    def test_tidy(self):
        assert isinstance(self.dc.tidy, pandas.DataFrame)
        tidycols = self.dc.tidy.columns.tolist()
        knowncols = self.known_columns + [self.known_roscol]
        assert sorted(tidycols) == sorted(knowncols)

    @helpers.seed
    def test_means(self):
        pdtest.assert_frame_equal(
            self.dc.means.round(3),
            self.known_means.round(3),
            check_names=False,
            check_less_precise=True
        )

    @helpers.seed
    def test_medians(self):
        pdtest.assert_frame_equal(
            self.dc.medians.round(3),
            self.known_medians.round(3),
            check_names=False,
            check_less_precise=True
        )

    def test_shaprio(self):
        expected_data = {
            ('Inflow', 'pvalue'): [
                1.35000277e-07, 1.15147395e-06, 2.84853918e-08, 7.25922699e-09,
                4.04597494e-06, 5.37404680e-08, 6.98694420e-05, 2.09285303e-06,
            ],
            ('Outflow', 'shapiro'): [
                0.68407392, 0.74749016, 0.897427022, 0.55306959,
                0.78847879, 0.88128829, 0.778801679, 0.71769607,
            ],
            ('Reference', 'shapiro'): [
                0.64490425, 0.66652446, 0.497832655, 0.65883779,
                0.57689213, 0.39423871, 0.583670616, 0.78648996,
            ],
            ('Inflow', 'shapiro'): [
                0.59733164, 0.67138719, 0.537596583, 0.48065340,
                0.71066832, 0.56260156, 0.789419353, 0.69042921,
            ],
            ('Outflow', 'pvalue'): [
                1.71045326e-06, 1.4417389138543513e-05, 0.0099637247622013092, 4.2077829220943386e-08,
                6.73221293e-05, 0.0042873905040323734, 4.616594742401503e-05, 5.1183847062929999e-06,
            ],
            ('Reference', 'pvalue'): [
                5.18675221e-07, 9.91817046e-07, 1.08479882e-08, 7.85432064e-07,
                7.80523379e-08, 1.08831232e-09, 9.34287243e-08, 6.22547740e-05,
            ],
        }
        result = self.dc.shapiro
        expected = pandas.DataFrame(expected_data, index=result.index)
        expected.columns.names = ['station', 'result']
        pdtest.assert_frame_equal(result, expected, check_less_precise=True)

    def test_shapiro_log(self):
        expected_data = {
            ('Inflow', 'log-shapiro'): [
                0.962131, 0.983048, 0.974349, 0.971814,
                0.953440, 0.986218, 0.970588, 0.980909,
            ],
            ('Inflow', 'pvalue'): [
                0.3912280, 0.916009, 0.700492, 0.630001,
                0.2414900, 0.964459, 0.596430, 0.871599,
            ],
            ('Outflow', 'log-shapiro'): [
                0.969009, 0.915884, 0.9498569, 0.972526,
                0.973539, 0.964473, 0.9782290, 0.982024,
            ],
            ('Outflow', 'pvalue'): [
                0.554274, 0.027473, 0.196495, 0.649692,
                0.677933, 0.442456, 0.806073, 0.895776,
            ],
            ('Reference', 'log-shapiro'): [
                0.980489, 0.946262, 0.986095, 0.991098,
                0.966805, 0.958543, 0.975473, 0.971864,
            ],
            ('Reference', 'pvalue'): [
                0.861968, 0.159437, 0.963002, 0.996583,
                0.498018, 0.321844, 0.731752, 0.631380,
            ]
        }
        result = self.dc.shapiro_log
        expected = pandas.DataFrame(expected_data, index=result.index)
        expected.columns.names = ['station', 'result']
        pdtest.assert_frame_equal(result, expected, check_less_precise=True)

    def test_lillifors(self):
        expected_data = {
            ('Inflow', 'lillifors'): [
                0.27843886, 0.24802344, 0.33527872, 0.37170719,
                0.22752955, 0.28630622, 0.20595218, 0.27953043
            ],
            ('Inflow', 'pvalue'): [
                6.24148409e-06, 0.000119332, 8.62060186e-09, 6.08463945e-11,
                0.000695739, 2.72633652e-06, 0.003660092, 5.57279417e-06
            ],
            ('Outflow', 'lillifors'): [
                0.31416123, 0.23332031, 0.13066911, 0.32460209,
                0.20829954, 0.18088983, 0.25659462, 0.22520102
            ],
            ('Outflow', 'pvalue'): [
                1.17145253e-07, 0.000430609, 0.248687678, 3.30298603e-08,
                0.003085194, 0.019562376, 5.41005765e-05, 0.000840341
            ],
            ('Reference', 'lillifors'): [
                0.28351398, 0.24930868, 0.29171088, 0.22593144,
                0.32020481, 0.42722648, 0.32208719, 0.16592420
            ],
            ('Reference', 'pvalue'): [
                3.66921686e-06, 0.000106205, 1.51965475e-06, 0.000792213,
                5.66187072e-08, 1.06408145e-14, 4.49998108e-08, 0.046766975
            ],
        }
        result = self.dc.lillifors
        expected = pandas.DataFrame(expected_data, index=result.index)
        expected.columns.names = ['station', 'result']
        pdtest.assert_frame_equal(result, expected, check_less_precise=True)

    def test_lillifors_log(self):
        expected_data = {
            ('Inflow', 'log-lillifors'): [
                0.10979498, 0.08601191, 0.13721881, 0.123985937,
                0.12767141, 0.09254070, 0.11319291, 0.107527896,
            ],
            ('Inflow', 'pvalue'): [
                0.51940912, 0.95582245, 0.18986954, 0.321330151,
                0.27964797, 0.82837543, 0.46666969, 0.556327244,
            ],
            ('Outflow', 'log-lillifors'): [
                0.10661711, 0.12321271, 0.15586768, 0.144033649,
                0.09536450, 0.11033031, 0.08012353, 0.089479881,
            ],
            ('Outflow', 'pvalue'): [
                0.57153020, 0.33058849, 0.07956179, 0.140596426,
                0.77423131, 0.51088991, 1.07047546, 0.887890467,
            ],
            ('Reference', 'log-lillifors'): [
                0.08280941, 0.12379594, 0.10251285, 0.070249172,
                0.08910822, 0.14824279, 0.10622305, 0.109876879,
            ],
            ('Reference', 'pvalue'): [
                1.01845443, 0.32358845, 0.64250052, 1.251670597,
                0.89515574, 0.11562015, 0.57817185, 0.518100907,
            ],
        }
        result = self.dc.lillifors_log
        expected = pandas.DataFrame(expected_data, index=result.index)
        expected.columns.names = ['station', 'result']
        pdtest.assert_frame_equal(result, expected, check_less_precise=True)

    def test_anderson_darling(self):
        with pytest.raises(NotImplementedError):
            self.dc.anderson_darling

    def test_anderson_darling_log(self):
        with pytest.raises(NotImplementedError):
            self.dc.anderson_darling_log

    @helpers.seed
    def test__generic_stat(self):
        result = self.dc._generic_stat(numpy.min, use_bootstrap=False)
        pdtest.assert_frame_equal(
            numpy.round(result, 3),
            self.known_genericstat,
            check_names=False
        )

    def test_mann_whitney(self):
        expected_data = {
            'pvalue': [
                0.9934626, 0.1029978, 0.9934626, 0.0701802, 0.1029978, 0.07018023,
                0.3214884, 0.0930252, 0.3214884, 0.5174506, 0.0930252, 0.51745067,
            ],
            'mann_whitney': [
                391.0, 492.0, 393.0, 503.0, 292.0, 281.0,
                453.0, 495.0, 331.0, 432.0, 289.0, 352.0,
            ]
        }

        pdtest.assert_frame_equal(
            self.prep_comp_result(self.dc.mann_whitney),
            self.prep_comp_expected(expected_data),
        )

    def test_t_test(self):
        expected_data = {
            't_test': [
                0.5069615, 1.7827515, -0.5069615, 1.6629067, -1.7827515, -1.6629067,
                0.2994807, 0.9528661, -0.2994807, 0.7150517, -0.9528661, -0.7150517,
            ],
            'pvalue': [
                0.6145228, 0.0835157, 0.6145228, 0.1038642, 0.0835157, 0.1038642,
                0.7657606, 0.3452425, 0.7657606, 0.4776926, 0.3452425, 0.4776926,
            ]
        }
        pdtest.assert_frame_equal(
            self.prep_comp_result(self.dc.t_test),
            self.prep_comp_expected(expected_data),
        )

    def test_levene(self):
        expected_data = {
            'levene': [
                0.3312645, 2.8502753, 0.3312645, 2.1520503, 2.8502753, 2.1520503,
                0.0024326, 0.2962336, 0.0024325, 0.4589038, 0.2962336, 0.4589038,
            ],
            'pvalue': [
                0.5673062, 0.0971261, 0.5673062, 0.1481797, 0.0971261, 0.1481797,
                0.9608452, 0.5884937, 0.9608452, 0.5010288, 0.5884937, 0.5010288
            ]
        }
        pdtest.assert_frame_equal(
            self.prep_comp_result(self.dc.levene),
            self.prep_comp_expected(expected_data),
            check_less_precise=True,
        )

    def test_wilcoxon(self):
        expected_data = {
            'wilcoxon': [
                183., 118., 183., 130., 118., 130.,
                183., 154., 183., 180., 154., 180.,
            ],
            'pvalue': [
                0.6488010, 0.052920, 0.6488010, 0.096449, 0.052920, 0.096449,
                0.6488010, 0.264507, 0.6488010, 0.600457, 0.264507, 0.600457,
            ]
        }
        pdtest.assert_frame_equal(
            self.prep_comp_result(self.dc.wilcoxon),
            self.prep_comp_expected(expected_data),
            check_less_precise=True,
        )

    def test_kendall(self):
        expected_data = {
            'kendalltau': [
                0.04232804,  0.06349206,  0.04232804,  0.02645503,  0.06349206,
                0.02645503, -0.06878307, -0.04232804, -0.06878307,  0.09523810,
                -0.04232804,  0.0952381
            ],
            'pvalue': [
                0.75192334,  0.63538834,  0.75192334,  0.84338527,  0.63538834,
                0.84338527,  0.60748308,  0.75192334,  0.60748308,  0.47693882,
                0.75192334,  0.47693882
            ]
        }
        pdtest.assert_frame_equal(
            self.prep_comp_result(self.dc.kendall),
            self.prep_comp_expected(expected_data),
            check_less_precise=True,
        )

    def test_spearman(self):
        expected_data = {
            'spearmanrho': [
                0.066776, 0.133004, 0.066776, 0.029556, 0.133004, 0.029556,
                -0.119868, -0.079365, -0.119868, 0.122058, -0.079365, 0.122058,
            ],
            'pvalue': [
                0.735654,  0.499856,  0.735654,  0.881316,  0.499856,  0.881316,
                0.543476,  0.688090,  0.543476,  0.536084,  0.688090,  0.536084,
            ]
        }
        pdtest.assert_frame_equal(
            self.prep_comp_result(self.dc.spearman),
            self.prep_comp_expected(expected_data),
            check_less_precise=True,
        )

    def test__comparison_stat(self):
        result = self.dc._comparison_stat(helpers.comp_statfxn, statname='Tester')

        expected_data = {
            'pvalue': [
                8.947202, 9.003257,  6.018320,  6.032946,
                3.276959, 3.235531, 10.306480, 10.331898,
                5.552678, 5.584064,  5.795812,  5.801780,
            ],
            'Tester': [
                35.788809, 36.013028, 24.073280, 24.131785,
                13.107839, 12.942125, 41.225920, 41.327592,
                22.210713, 22.336259, 23.183249, 23.207123,
            ]
        }

        expected = pandas.DataFrame(expected_data, index=self.expected_comp_index)
        pdtest.assert_frame_equal(
            self.prep_comp_result(result),
            expected
        )

    def test_locations(self):
        for l in self.dc.locations:
            assert isinstance(l, Location)
        assert len(self.dc.locations) == 24
        assert self.dc.locations[0].definition == {'loc': 'Inflow', 'param': 'A'}
        assert self.dc.locations[1].definition == {'loc': 'Inflow', 'param': 'B'}
        assert self.dc.locations[6].definition == {'loc': 'Inflow', 'param': 'G'}
        assert self.dc.locations[8].definition == {'loc': 'Outflow', 'param': 'A'}

    def test_datasets(self):
        for d in self.dc.datasets:
            assert isinstance(d, Dataset)
        assert len(self.dc.datasets) == 8
        assert self.dc.datasets[0].definition == {'param': 'A'}
        assert self.dc.datasets[1].definition == {'param': 'B'}
        assert self.dc.datasets[6].definition == {'param': 'G'}
        assert self.dc.datasets[7].definition == {'param': 'H'}


def load_known_dc_stat(csv_data):
    df = (
        pandas.read_csv(StringIO(dedent(csv_data)))
        .set_index(['param', 'station'])
        .unstack(level='station')
    )
    df.columns = df.columns.swaplevel(0, 1)
    return df.sort_index(axis='columns')


class Test_DataCollection_baseline(_base_DataCollecionMixin):
    def setup(self):
        self.data = helpers.make_dc_data(ndval=self.known_ndval, rescol=self.known_raw_rescol,
                                         qualcol=self.known_qualcol)
        self.dc = DataCollection(self.data, paramcol='param', stationcol='loc',
                                 ndval=self.known_ndval, rescol=self.known_raw_rescol,
                                 qualcol=self.known_qualcol, pairgroups=['state', 'bmp'])

        self.known_groupcols = ['loc', 'param']
        self.known_columns = ['loc', 'param', self.known_raw_rescol, '__censorship']
        self.known_bsiter = 10000
        self.known_means = load_known_dc_stat("""\
            param,station,lower,mean,upper
            A,Inflow,2.748679,5.881599,9.2116
            A,Outflow,2.600673,4.80762,7.197304
            A,Reference,1.416864,2.555152,3.839059
            B,Inflow,3.908581,6.96897,10.273159
            B,Outflow,3.710928,6.297485,9.189436
            B,Reference,2.56929,4.918943,7.457957
            C,Inflow,1.608618,3.674344,6.010133
            C,Outflow,2.351154,3.32731,4.358206
            C,Reference,1.855405,3.8916,6.490131
            D,Inflow,1.755983,4.240805,7.240196
            D,Outflow,1.98268,4.450713,7.433584
            D,Reference,2.270379,3.818759,5.648175
            E,Inflow,2.385389,3.877227,5.495141
            E,Outflow,2.5154,4.01144,5.589268
            E,Reference,1.240479,2.587013,4.085286
            F,Inflow,2.641011,5.35544,8.614032
            F,Outflow,2.487351,3.514617,4.608282
            F,Reference,1.532666,4.987571,9.699604
            G,Inflow,2.267453,3.907819,5.639204
            G,Outflow,1.680899,2.758258,3.920394
            G,Reference,1.877497,3.618892,5.698217
            H,Inflow,1.920636,3.252142,4.726754
            H,Outflow,1.549154,2.632812,3.841772
            H,Reference,1.717932,2.515394,3.400027
        """)

        self.known_medians = load_known_dc_stat("""\
            param,station,lower,median,upper
            A,Inflow,1.329239,2.82959,3.485278
            A,Outflow,1.23428,2.499703,3.553598
            A,Reference,0.694529,1.292849,2.051181
            B,Inflow,1.717358,3.511042,6.326503
            B,Outflow,0.88851,2.484602,5.727451
            B,Reference,0.833077,2.118402,4.418128
            C,Inflow,0.895116,1.409523,2.011819
            C,Outflow,1.318304,3.075015,3.833705
            C,Reference,0.937387,1.832793,2.829857
            D,Inflow,0.766254,1.873097,2.934091
            D,Outflow,0.877327,1.613252,1.929779
            D,Reference,1.190175,2.186729,3.718543
            E,Inflow,1.327971,2.883189,4.113432
            E,Outflow,1.341962,2.101454,3.51228
            E,Reference,0.611767,1.210565,1.916643
            F,Inflow,1.17812,2.629969,4.345175
            F,Outflow,1.49992,2.495085,4.226673
            F,Reference,0.775816,1.473314,2.190129
            G,Inflow,0.735196,1.84397,3.840832
            G,Outflow,0.817122,1.558962,2.420768
            G,Reference,1.058775,2.008404,2.512609
            H,Inflow,1.2963,1.729254,2.828794
            H,Outflow,0.635636,1.604207,2.387034
            H,Reference,0.976348,1.940175,2.846231
        """)

        self.known_genericstat = pandas.DataFrame({
            ('Reference', 'stat'): {
                'D': 0.209, 'A': 0.119, 'B': 0.307, 'H': 0.221,
                'G': 0.189, 'F': 0.099, 'E': 0.211, 'C': 0.135,
            },
            ('Inflow', 'stat'): {
                'D': 0.107, 'A': 0.178, 'B': 0.433, 'H': 0.210,
                'G': 0.096, 'F': 0.157, 'E': 0.236, 'C': 0.116,
            },
            ('Outflow', 'stat'): {
                'D': 0.118, 'A': 0.344, 'B': 0.409, 'H': 0.128,
                'G': 0.124, 'F': 0.300, 'E': 0.126, 'C': 0.219,
            }
        })


class Test_DataCollection_customNDval(Test_DataCollection_baseline):
    known_raw_rescol = 'conc'
    known_roscol = 'ros_conc'
    known_rescol = 'ros_conc'
    known_qualcol = 'anote'
    known_stationcol = 'loc'
    known_paramcol = 'param'
