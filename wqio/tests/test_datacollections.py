from distutils.version import LooseVersion
from textwrap import dedent
from io import StringIO

import numpy
import scipy
import pandas

from unittest import mock
import pytest
import pandas.util.testing as pdtest
from wqio.tests import helpers

from wqio.features import Location, Dataset
from wqio.datacollections import DataCollection


OLD_SCIPY = LooseVersion(scipy.version.version) < LooseVersion('0.19')


def check_stat(expected_csv, result, comp=False):
    index_col = [0]
    if comp:
        index_col += [1]

    file_obj = StringIO(dedent(expected_csv))
    expected = pandas.read_csv(file_obj, header=[0, 1], index_col=index_col)

    if comp:
        expected = expected.stack(level=-1)

    pdtest.assert_frame_equal(expected, result, check_less_precise=True)


def remove_g_and_h(group):
    return group.name[1] not in ['G', 'H']


@pytest.fixture
def dc():
    df = helpers.make_dc_data_complex()
    dc = DataCollection(df, rescol='res', qualcol='qual',
                        stationcol='loc', paramcol='param',
                        ndval='<', othergroups=None,
                        pairgroups=['state', 'bmp'],
                        useros=True, filterfxn=remove_g_and_h,
                        bsiter=10000)

    return dc


@pytest.fixture
def dc_noNDs():
    df = helpers.make_dc_data_complex()
    dc = DataCollection(df, rescol='res', qualcol='qual',
                        stationcol='loc', paramcol='param',
                        ndval='junk', othergroups=None,
                        pairgroups=['state', 'bmp'],
                        useros=True, filterfxn=remove_g_and_h,
                        bsiter=10000)

    return dc


def test_basic_attr(dc):
    assert dc._raw_rescol == 'res'
    assert isinstance(dc.data, pandas.DataFrame)
    assert dc.roscol == 'ros_res'
    assert dc.rescol == 'ros_res'
    assert dc.qualcol == 'qual'
    assert dc.stationcol == 'loc'
    assert dc.paramcol == 'param'
    assert dc.ndval == ['<']
    assert dc.bsiter == 10000
    assert dc.groupcols == ['loc', 'param']
    assert dc.tidy_columns == ['loc', 'param', 'res', '__censorship']
    assert hasattr(dc, 'filterfxn')


def test_data(dc):
    assert isinstance(dc.data, pandas.DataFrame)
    assert dc.data.shape == (519, 8)
    assert 'G' in dc.data['param'].unique()
    assert 'H' in dc.data['param'].unique()


def test_tidy(dc):
    assert isinstance(dc.tidy, pandas.DataFrame)
    assert dc.tidy.shape == (388, 5)
    assert 'G' not in dc.tidy['param'].unique()
    assert 'H' not in dc.tidy['param'].unique()
    collist = ['loc', 'param', 'res', '__censorship', 'ros_res']
    assert dc.tidy.columns.tolist() == collist


def test_paired(dc):
    assert isinstance(dc.paired, pandas.DataFrame)
    assert dc.paired.shape == (164, 6)
    assert 'G' not in dc.paired.index.get_level_values('param').unique()
    assert 'H' not in dc.paired.index.get_level_values('param').unique()
    dc.paired.columns.tolist() == [('res', 'Inflow'),
                                   ('res', 'Outflow'),
                                   ('res', 'Reference'),
                                   ('__censorship', 'Inflow'),
                                   ('__censorship', 'Outflow'),
                                   ('__censorship', 'Reference')]


def test_count(dc):
    known_csv = """\
        station,Inflow,Outflow,Reference
        result,Count,Count,Count
        param,,,
        A,21,22,20
        B,24,22,19
        C,24,24,25
        D,24,25,21
        E,19,16,20
        F,21,24,17
    """
    check_stat(known_csv, dc.count)


@helpers.seed
def test_median(dc):
    known_csv = """\
        station,Inflow,Inflow,Inflow,Outflow,Outflow,Outflow,Reference,Reference,Reference
        result,lower,median,upper,lower,median,upper,lower,median,upper
        param,,,,,,,,,
        A,0.334506,1.197251,2.013994,0.883961,2.231058,2.626023,1.073386,1.639472,1.717293
        B,1.366948,2.773989,3.294859,0.238161,1.546499,2.579206,0.204164,1.565076,2.137915
        C,0.169047,0.525957,0.68024,0.247769,0.396984,0.540742,0.136462,0.412693,0.559458
        D,0.374122,1.201892,2.117927,0.516989,1.362759,1.827087,0.314655,0.882695,1.24545
        E,0.240769,1.070858,1.152887,0.287914,0.516746,1.477551,0.366824,0.80716,2.040739
        F,0.05667,0.832488,1.282846,0.425237,1.510942,2.209813,0.162327,0.745993,1.992513
    """
    check_stat(known_csv, dc.median)


@helpers.seed
def test_mean(dc):
    known_csv = """\
        station,Inflow,Inflow,Inflow,Outflow,Outflow,Outflow,Reference,Reference,Reference
        result,lower,mean,upper,lower,mean,upper,lower,mean,upper
        param,,,,,,,,,
        A,1.244875,2.646682,4.290582,1.952764,5.249281,9.015144,1.501006,3.777974,6.293246
        B,3.088871,7.647175,12.670781,1.597771,6.863835,12.65965,1.006978,4.504255,9.489099
        C,0.372328,0.513248,0.657034,0.418296,1.004637,1.708248,0.355404,0.541962,0.742944
        D,1.298897,3.021235,5.011268,1.269134,2.318808,3.456552,1.006727,1.945828,2.912135
        E,0.841728,1.914696,3.086477,0.581081,1.098241,1.647634,1.115033,2.283292,3.5266
        F,0.730967,9.825404,25.303784,1.47824,3.450184,5.660159,1.001284,2.491708,4.141191
    """
    check_stat(known_csv, dc.mean)


@helpers.seed
def test_std_dev(dc):
    known_csv = """\
        station,Inflow,Outflow,Reference
        result,std. dev.,std. dev.,std. dev.
        param,,,
        A,3.58649,8.719371,5.527633
        B,12.360099,13.60243,10.759285
        C,0.353755,1.691208,0.493325
        D,4.811938,2.849393,2.248178
        E,2.55038,1.096698,2.789238
        F,34.447565,5.361033,3.398367
    """
    check_stat(known_csv, dc.std_dev)


@helpers.seed
def test_percentile_25(dc):
    known_csv = """\
        station,Inflow,Outflow,Reference
        result,pctl 25,pctl 25,pctl 25
        param,,,
        A,0.522601,0.906029,1.094721
        B,1.472541,0.251126,0.314226
        C,0.164015,0.267521,0.136462
        D,0.35688,0.516989,0.383895
        E,0.364748,0.311508,0.394658
        F,0.120068,0.406132,0.224429
    """
    check_stat(known_csv, dc.percentile(25))


@helpers.seed
def test_percentile_75(dc):
    known_csv = """\
        station,Inflow,Outflow,Reference
        result,pctl 75,pctl 75,pctl 75
        param,,,
        A,2.563541,3.838021,2.650648
        B,4.728871,2.849948,2.261847
        C,0.776388,0.853535,0.792612
        D,3.04268,2.79341,3.611793
        E,1.532775,1.59183,3.201534
        F,1.792985,2.80979,2.742249
    """
    check_stat(known_csv, dc.percentile(75))


@helpers.seed
def test_logmean(dc):
    known_csv = """\
        station,Inflow,Inflow,Inflow,Outflow,Outflow,Outflow,Reference,Reference,Reference
        result,Log-mean,lower,upper,Log-mean,lower,upper,Log-mean,lower,upper
        param,,,,,,,,,
        A,0.140559,-0.542266,0.655481,0.733004,0.083331,1.219697,0.545205,-0.095031,1.012713
        B,1.026473,0.379605,1.527193,0.105106,-0.910899,0.862624,0.068638,-0.937859,0.663884
        C,-0.963004,-1.316093,-0.652167,-0.83221,-1.47521,-0.406836,-1.088377,-1.551392,-0.70957
        D,0.062317,-0.674413,0.579414,0.185757,-0.367616,0.591613,-0.063507,-0.670086,0.426025
        E,-0.103655,-0.773771,0.399788,-0.456202,-1.101804,0.022327,-0.068135,-0.778097,0.516806
        F,-0.442721,-1.890026,0.310913,0.211658,-0.48962,0.747611,-0.253352,-1.121651,0.494673
    """
    check_stat(known_csv, dc.logmean)


@helpers.seed
def test_logstd_dev(dc):
    known_csv = """\
        station,Inflow,Outflow,Reference
        result,Log-std. dev.,Log-std. dev.,Log-std. dev.
        param,,,
        A,1.374026,1.343662,1.225352
        B,1.430381,2.07646,1.662001
        C,0.818504,1.263631,1.057177
        D,1.530871,1.187246,1.277927
        E,1.264403,1.121038,1.474431
        F,2.324063,1.516331,1.701596
    """
    check_stat(known_csv, dc.logstd_dev)


@helpers.seed
def test_geomean(dc):
    known_csv = """\
        station,Inflow,Inflow,Inflow,Outflow,Outflow,Outflow,Reference,Reference,Reference
        Geo-mean,Log-mean,lower,upper,Log-mean,lower,upper,Log-mean,lower,upper
        param,,,,,,,,,
        A,1.150917,0.581429,1.926069,2.081323,1.086901,3.386162,1.724962,0.909344,2.75306
        B,2.791205,1.461706,4.605233,1.110829,0.402163,2.369369,1.071049,0.391465,1.942321
        C,0.381744,0.268181,0.520916,0.435087,0.228731,0.665753,0.336763,0.211953,0.491856
        D,1.064299,0.509456,1.784993,1.204129,0.692383,1.8069,0.938467,0.511665,1.53116
        E,0.901536,0.46127,1.491509,0.633686,0.332271,1.022578,0.934134,0.459279,1.676664
        F,0.642286,0.151068,1.36467,1.235726,0.612859,2.111949,0.776195,0.325742,1.639961
    """
    check_stat(known_csv, dc.geomean)


@helpers.seed
def test_geostd_dev(dc):
    known_csv = """\
        station,Inflow,Outflow,Reference
        Geo-std. dev.,Log-std. dev.,Log-std. dev.,Log-std. dev.
        param,,,
        A,3.951225,3.833055,3.405365
        B,4.180294,7.976181,5.269843
        C,2.267105,3.538244,2.878234
        D,4.622199,3.278041,3.589191
        E,3.540977,3.068036,4.368548
        F,10.217099,4.55548,5.48269
    """
    check_stat(known_csv, dc.geostd_dev)


@helpers.seed
def test_shapiro(dc):
    known_csv = """\
        station,Inflow,Inflow,Outflow,Outflow,Reference,Reference
        result,pvalue,shapiro,pvalue,shapiro,pvalue,shapiro
        param,,,,,,
        A,1.8405e-05,0.685783088,7.41e-07,0.576068878,4.478e-06,0.617349863
        B,5.35e-07,0.594411373,2.6e-07,0.530962467,9.6e-08,0.414710045
        C,0.028773842,0.905906141,1.65e-07,0.546625972,0.002789602,0.860373497
        D,1.125e-06,0.622915387,1.4798e-05,0.72237432,0.000201738,0.765179813
        E,1.6584e-05,0.654137194,0.004895572,0.818813443,0.000165472,0.749170363
        F,4e-09,0.292915702,1.543e-06,0.63467145,0.000167156,0.713968396
    """
    check_stat(known_csv, dc.shapiro)


@helpers.seed
def test_shapiro_log(dc):
    known_csv = """\
        station,Inflow,Inflow,Outflow,Outflow,Reference,Reference
        result,log-shapiro,pvalue,log-shapiro,pvalue,log-shapiro,pvalue
        param,,,,,,
        A,0.983521938,0.96662426,0.979861856,0.913820148,0.939460814,0.234214202
        B,0.957531095,0.390856266,0.97048676,0.722278714,0.967978418,0.735424638
        C,0.906479359,0.029602444,0.974698305,0.78197974,0.967106879,0.572929323
        D,0.989704251,0.995502174,0.990663111,0.997093379,0.964812279,0.617747009
        E,0.955088913,0.479993254,0.95211035,0.523841977,0.963425279,0.61430341
        F,0.97542423,0.847370088,0.982230783,0.933124721,0.966197193,0.749036908
    """
    check_stat(known_csv, dc.shapiro_log)


@helpers.seed
def test_lilliefors(dc):
    known_csv = """\
        station,Inflow,Inflow,Outflow,Outflow,Reference,Reference
        result,lilliefors,pvalue,lilliefors,pvalue,lilliefors,pvalue
        param,,,,,,
        A,0.3081305983,1.42209e-05,0.3405937884,3.324e-07,0.3644530472,1.337e-07
        B,0.3676403629,3.3e-09,0.4203432704,0.0,0.4171653285,1.2e-09
        C,0.1667986537,0.082737184,0.3247330623,4.463e-07,0.1617532493,0.0904548308
        D,0.2730119678,6.65111e-05,0.2403110942,0.0006650862,0.2969189853,3.74178e-05
        E,0.3413982663,2.6564e-06,0.2393138433,0.0148623981,0.2337732366,0.005474318
        F,0.419545454,1e-10,0.3313152839,2.197e-07,0.2842488753,0.0007414141
    """
    check_stat(known_csv, dc.lilliefors)


@helpers.seed
def test_lilliefors_log(dc):
    known_csv = """\
        station,Inflow,Inflow,Outflow,Outflow,Reference,Reference
        result,log-lilliefors,pvalue,log-lilliefors,pvalue,log-lilliefors,pvalue
        param,,,,,,
        A,0.08548109,0.2,0.15443943,0.18686121,0.20141389,0.03268737
        B,0.16162839,0.10419352,0.12447902,0.2,0.15934334,0.2
        C,0.16957278,0.07248915,0.12388174,0.2,0.11746642,0.2
        D,0.06885549,0.2,0.06067356,0.2,0.13401954,0.2
        E,0.13506577,0.2,0.14552341,0.2,0.09164876,0.2
        F,0.14420794,0.2,0.08463267,0.2,0.08586933,0.2
    """
    check_stat(known_csv, dc.lilliefors_log)


@helpers.seed
def test_anderson_darling(dc):
    with helpers.raises(NotImplementedError):
        _ = dc.anderson_darling


@helpers.seed
def test_anderson_darling_log(dc):
    with helpers.raises(NotImplementedError):
        _ = dc.anderson_darling_log


@helpers.seed
def test_mann_whitney(dc):
    known_csv = """\
        ,,mann_whitney,mann_whitney,mann_whitney,pvalue,pvalue,pvalue
        loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
        param,loc_1,,,,,,
        A,Inflow,,180.0,179.0,,0.2198330905,0.4263216587
        A,Outflow,282.0,,248.0,0.2198330905,,0.488580368
        A,Reference,241.0,192.0,,0.4263216587,0.488580368,
        B,Inflow,,345.0,317.0,,0.0766949991,0.0304383994
        B,Outflow,183.0,,216.0,0.0766949991,,0.8650586835
        B,Reference,139.0,202.0,,0.0304383994,0.8650586835,
        C,Inflow,,282.0,323.0,,0.9097070273,0.6527104406
        C,Outflow,294.0,,323.0,0.9097070273,,0.6527104406
        C,Reference,277.0,277.0,,0.6527104406,0.6527104406,
        D,Inflow,,285.0,263.0,,0.7718162376,0.8111960975
        D,Outflow,315.0,,293.0,0.7718162376,,0.5082395211
        D,Reference,241.0,232.0,,0.8111960975,0.5082395211,
        E,Inflow,,164.0,188.0,,0.7033493939,0.9663820218
        E,Outflow,140.0,,132.0,0.7033493939,,0.3813114322
        E,Reference,192.0,188.0,,0.9663820218,0.3813114322,
        F,Inflow,,201.0,172.0,,0.2505911218,0.8601783903
        F,Outflow,303.0,,236.0,0.2505911218,,0.4045186043
        F,Reference,185.0,172.0,,0.8601783903,0.4045186043
    """
    check_stat(known_csv, dc.mann_whitney, comp=True)


@helpers.seed
def test_t_test(dc):
    known_csv = """\
        ,,pvalue,pvalue,pvalue,t_test,t_test,t_test
        loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
        param,loc_1,,,,,,
        A,Inflow,,0.2178424157,0.4563196599,,-1.2604458127,-0.7539785777
        A,Outflow,0.2178424157,,0.5240147979,1.2604458127,,0.643450194
        A,Reference,0.4563196599,0.5240147979,,0.7539785777,-0.643450194,
        B,Inflow,,0.8430007638,0.3898358794,,0.1992705833,0.869235357
        B,Outflow,0.8430007638,,0.5491097882,-0.1992705833,,0.6043850808
        B,Reference,0.3898358794,0.5491097882,,-0.869235357,-0.6043850808,
        C,Inflow,,0.1847386316,0.8191392537,,-1.3639360123,-0.2300373632
        C,Outflow,0.1847386316,,0.2179907667,1.3639360123,,1.2615982727
        C,Reference,0.8191392537,0.2179907667,,0.2300373632,-1.2615982727,
        D,Inflow,,0.5484265023,0.344783812,,0.6056706932,0.9582600001
        D,Outflow,0.5484265023,,0.6299742693,-0.6056706932,,0.4851636024
        D,Reference,0.344783812,0.6299742693,,-0.9582600001,-0.4851636024,
        E,Inflow,,0.2304569921,0.6770414622,,1.2287029977,-0.4198288251
        E,Outflow,0.2304569921,,0.1023435465,-1.2287029977,,-1.6935358498
        E,Reference,0.6770414622,0.1023435465,,0.4198288251,1.6935358498,
        F,Inflow,,0.422008391,0.3549979666,,0.8190789273,0.9463539528
        F,Outflow,0.422008391,,0.4988994144,-0.8190789273,,0.6826435968
        F,Reference,0.3549979666,0.4988994144,,-0.9463539528,-0.6826435968
    """
    check_stat(known_csv, dc.t_test, comp=True)


@helpers.seed
def test_levene(dc):
    known_csv = """\
        ,,levene,levene,levene,pvalue,pvalue,pvalue
        loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
        param,loc_1,,,,,,
        A,Inflow,,1.176282059,0.293152155,,0.284450688,0.591287419
        A,Outflow,1.176282059,,0.397705309,0.284450688,,0.531863542
        A,Reference,0.293152155,0.397705309,,0.591287419,0.531863542,
        B,Inflow,,0.003559637,0.402002411,,0.952694449,0.529578712
        B,Outflow,0.003559637,,0.408938588,0.952694449,,0.526247443
        B,Reference,0.402002411,0.408938588,,0.529578712,0.526247443,
        C,Inflow,,1.965613561,0.679535532,,0.167626459,0.413910674
        C,Outflow,1.965613561,,1.462364363,0.167626459,,0.232602352
        C,Reference,0.679535532,1.462364363,,0.413910674,0.232602352,
        D,Inflow,,0.643364813,0.983777911,,0.426532092,0.32681669
        D,Outflow,0.643364813,,0.116830634,0.426532092,,0.734124856
        D,Reference,0.983777911,0.116830634,,0.32681669,0.734124856,
        E,Inflow,,0.961616536,0.410491665,,0.333914902,0.525668596
        E,Outflow,0.961616536,,2.726351564,0.333914902,,0.107912818
        E,Reference,0.410491665,2.726351564,,0.525668596,0.107912818,
        F,Inflow,,0.841984453,0.734809611,,0.363948105,0.396999375
        F,Outflow,0.841984453,,0.25881357,0.363948105,,0.613802541
        F,Reference,0.734809611,0.25881357,,0.396999375,0.613802541,
    """
    check_stat(known_csv, dc.levene, comp=True)


@helpers.seed
def test_wilcoxon(dc):
    known_csv = """\
        ,,pvalue,pvalue,pvalue,wilcoxon,wilcoxon,wilcoxon
        loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
        param,loc_1,,,,,,
        A,Inflow,,0.0351569731,0.4074344561,,32.0,59.0
        A,Outflow,0.0351569731,,0.2552905052,32.0,,46.0
        A,Reference,0.4074344561,0.2552905052,,59.0,46.0,
        B,Inflow,,0.6001794871,0.1823383542,,38.0,22.0
        B,Outflow,0.6001794871,,0.8588630128,38.0,,31.0
        B,Reference,0.1823383542,0.8588630128,,22.0,31.0,
        C,Inflow,,0.1592244176,0.5840564537,,75.0,120.0
        C,Outflow,0.1592244176,,0.4470311612,75.0,,113.0
        C,Reference,0.5840564537,0.4470311612,,120.0,113.0,
        D,Inflow,,0.5936182408,0.5302845968,,44.0,31.0
        D,Outflow,0.5936182408,,0.9721253297,44.0,,45.0
        D,Reference,0.5302845968,0.9721253297,,31.0,45.0,
        E,Inflow,,0.8589549227,0.3862707204,,21.0,19.0
        E,Outflow,0.8589549227,,0.0711892343,21.0,,16.0
        E,Reference,0.3862707204,0.0711892343,,19.0,16.0,
        F,Inflow,,0.4924592975,0.952765022,,62.0,22.0
        F,Outflow,0.4924592975,,0.6566419343,62.0,,28.0
        F,Reference,0.952765022,0.6566419343,,22.0,28.0
    """
    with pytest.warns(UserWarning):
        check_stat(known_csv, dc.wilcoxon, comp=True)


@helpers.seed
def test_ranksums(dc):
    known_csv = """\
        ,,pvalue,pvalue,pvalue,rank_sums,rank_sums,rank_sums
        loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
        param,loc_1,,,,,,
        A,Inflow,,0.2153009,0.4187782,,-1.2391203,-0.8085428
        A,Outflow,0.2153009,,0.4807102,1.2391203,,0.7051607
        A,Reference,0.4187782,0.4807102,,0.8085428,-0.7051607,
        B,Inflow,,0.0748817,0.029513,,1.781188,2.1765661
        B,Outflow,0.0748817,,0.8547898,-1.781188,,0.1830104
        B,Reference,0.029513,0.8547898,,-2.1765661,-0.1830104,
        C,Inflow,,0.9015386,0.6455162,,-0.1237179,0.46
        C,Outflow,0.9015386,,0.6455162,0.1237179,,0.46
        C,Reference,0.6455162,0.6455162,,-0.46,-0.46,
        D,Inflow,,0.7641772,0.8023873,,-0.3,0.2502587
        D,Outflow,0.7641772,,0.5011969,0.3,,0.6726078
        D,Reference,0.8023873,0.5011969,,-0.2502587,-0.6726078,
        E,Inflow,,0.6911022,0.9551863,,0.3973597,-0.0561951
        E,Outflow,0.6911022,,0.3727144,-0.3973597,,-0.8914004
        E,Reference,0.9551863,0.3727144,,0.0561951,0.8914004,
        F,Inflow,,0.2459307,0.8486619,,-1.1602902,-0.190826
        F,Outflow,0.2459307,,0.3971011,1.1602902,,0.8468098
        F,Reference,0.8486619,0.3971011,,0.190826,-0.8468098,
    """
    check_stat(known_csv, dc.ranksums, comp=True)


@helpers.seed
@pytest.mark.xfail(OLD_SCIPY, reason='Scipy < 0.19')
def test_kendall(dc):
    known_csv = """\
        ,,kendalltau,kendalltau,kendalltau,pvalue,pvalue,pvalue
        loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
        param,loc_1,,,,,,
        A,Inflow,,-0.051661,-0.00738,,0.772893,0.967114
        A,Outflow,-0.051661,,-0.083333,0.772893,,0.652548
        A,Reference,-0.00738,-0.083333,,0.967114,0.652548,
        B,Inflow,,0.441351,0.298246,,0.015267,0.119265
        B,Outflow,0.441351,,0.559855,0.015267,,0.004202
        B,Reference,0.298246,0.559855,,0.119265,0.004202,
        C,Inflow,,0.280223,0.084006,,0.078682,0.578003
        C,Outflow,0.280223,,-0.1417,0.078682,,0.352394
        C,Reference,0.084006,-0.1417,,0.578003,0.352394,
        D,Inflow,,0.403469,0.095299,,0.020143,0.634826
        D,Outflow,0.403469,,0.318337,0.020143,,0.094723
        D,Reference,0.095299,0.318337,,0.634826,0.094723,
        E,Inflow,,0.114286,0.640703,,0.673337,0.004476
        E,Outflow,0.114286,,0.167944,0.673337,,0.449603
        E,Reference,0.640703,0.167944,,0.004476,0.449603,
        F,Inflow,,0.0,0.07231,,1.0,0.763851
        F,Outflow,0.0,,0.388889,1.0,,0.063
        F,Reference,0.07231,0.388889,,0.763851,0.063,
    """
    check_stat(known_csv, dc.kendall, comp=True)


@helpers.seed
def test_spearman(dc):
    known_csv = """\
        ,,pvalue,pvalue,pvalue,spearmanrho,spearmanrho,spearmanrho
        loc_2,,Inflow,Outflow,Reference,Inflow,Outflow,Reference
        param,loc_1,,,,,,
        A,Inflow,,0.7574884491,0.9627447553,,-0.0809319588,0.012262418
        A,Outflow,0.7574884491,,0.7617330788,-0.0809319588,,-0.0823529412
        A,Reference,0.9627447553,0.7617330788,,0.012262418,-0.0823529412,
        B,Inflow,,0.0110829791,0.0775159774,,0.5831305575,0.4537313433
        B,Outflow,0.0110829791,,0.0024069317,0.5831305575,,0.6850916941
        B,Reference,0.0775159774,0.0024069317,,0.4537313433,0.6850916941,
        C,Inflow,,0.1330504059,0.6063501968,,0.3387640122,0.1134228342
        C,Outflow,0.1330504059,,0.3431640379,0.3387640122,,-0.2070506455
        C,Reference,0.6063501968,0.3431640379,,0.1134228342,-0.2070506455,
        D,Inflow,,0.0195715066,0.4751861062,,0.4935814032,0.1858231711
        D,Outflow,0.0195715066,,0.1263974782,0.4935814032,,0.363209462
        D,Reference,0.4751861062,0.1263974782,,0.1858231711,0.363209462,
        E,Inflow,,0.9828818202,0.0013596162,,0.0084033613,0.8112988341
        E,Outflow,0.9828818202,,0.3413722947,0.0084033613,,0.3012263814
        E,Reference,0.0013596162,0.3413722947,,0.8112988341,0.3012263814,
        F,Inflow,,0.9645303744,0.6759971848,,-0.0106277141,0.1348767061
        F,Outflow,0.9645303744,,0.0560590794,-0.0106277141,,0.5028571429
        F,Reference,0.6759971848,0.0560590794,,0.1348767061,0.5028571429
    """
    check_stat(known_csv, dc.spearman, comp=True)


@helpers.seed
def test_theilslopes(dc):
    with helpers.raises(NotImplementedError):
        _ = dc.theilslopes


def test_inventory(dc):
    known_csv = StringIO(dedent("""\
    loc,param,Count,Non-Detect
    Inflow,A,21,3
    Inflow,B,24,6
    Inflow,C,24,0
    Inflow,D,24,11
    Inflow,E,19,4
    Inflow,F,21,8
    Outflow,A,22,1
    Outflow,B,22,9
    Outflow,C,24,4
    Outflow,D,25,12
    Outflow,E,16,2
    Outflow,F,24,8
    Reference,A,20,2
    Reference,B,19,6
    Reference,C,25,4
    Reference,D,21,12
    Reference,E,20,3
    Reference,F,17,7
    """))
    expected = pandas.read_csv(known_csv, index_col=[0, 1]).astype(int)
    pdtest.assert_frame_equal(expected, dc.inventory.astype(int),
                              check_names=False,
                              check_less_precise=True)


def test_inventory_noNDs(dc_noNDs):
    known_csv = StringIO(dedent("""\
    loc,param,Count,Non-Detect
    Inflow,A,21,0
    Inflow,B,24,0
    Inflow,C,24,0
    Inflow,D,24,0
    Inflow,E,19,0
    Inflow,F,21,0
    Outflow,A,22,0
    Outflow,B,22,0
    Outflow,C,24,0
    Outflow,D,25,0
    Outflow,E,16,0
    Outflow,F,24,0
    Reference,A,20,0
    Reference,B,19,0
    Reference,C,25,0
    Reference,D,21,0
    Reference,E,20,0
    Reference,F,17,0
    """))
    expected = pandas.read_csv(known_csv, index_col=[0, 1]).astype(int)
    pdtest.assert_frame_equal(expected, dc_noNDs.inventory.astype(int),
                              check_names=False,
                              check_less_precise=True)


@helpers.seed
def test_stat_summary(dc):
    known_csv = StringIO(dedent("""\
    ros_res,loc,A,B,C,D,E,F
    Count,Inflow,21,24,24,24,19,21
    Count,Outflow,22,22,24,25,16,24
    Count,Reference,20,19,25,21,20,17
    Non-Detect,Inflow,3.0,6.0,0.0,11.0,4.0,8.0
    Non-Detect,Outflow,1.0,9.0,4.0,12.0,2.0,8.0
    Non-Detect,Reference,2.0,6.0,4.0,12.0,3.0,7.0
    mean,Inflow,2.64668,7.64717,0.51325,3.02124,1.9147,9.8254
    mean,Outflow,5.24928,6.86384,1.00464,2.31881,1.09824,3.45018
    mean,Reference,3.77797,4.50425,0.54196,1.94583,2.28329,2.49171
    std,Inflow,3.67506,12.62594,0.36136,4.91543,2.62027,35.29825
    std,Outflow,8.92456,13.92253,1.72758,2.90815,1.13267,5.47634
    std,Reference,5.67123,11.05411,0.5035,2.3037,2.8617,3.50296
    min,Inflow,0.0756,0.17404,0.10213,0.05365,0.08312,0.00803
    min,Outflow,0.11177,0.02106,0.03578,0.11678,0.07425,0.06377
    min,Reference,0.15575,0.04909,0.04046,0.08437,0.05237,0.03445
    10%,Inflow,0.1772,0.45233,0.13467,0.15495,0.1763,0.03548
    10%,Outflow,0.44852,0.08297,0.08222,0.26949,0.19903,0.18008
    10%,Reference,0.38448,0.13467,0.08241,0.19355,0.12777,0.09457
    25%,Inflow,0.5226,1.47254,0.16401,0.35688,0.36475,0.12007
    25%,Outflow,0.90603,0.25113,0.26752,0.51699,0.31151,0.40613
    25%,Reference,1.09472,0.31423,0.13646,0.3839,0.39466,0.22443
    50%,Inflow,1.19725,2.77399,0.52596,1.20189,1.07086,0.83249
    50%,Outflow,2.23106,1.5465,0.39698,1.36276,0.51675,1.51094
    50%,Reference,1.63947,1.56508,0.41269,0.8827,0.80716,0.74599
    75%,Inflow,2.56354,4.72887,0.77639,3.04268,1.53278,1.79299
    75%,Outflow,3.83802,2.84995,0.85354,2.79341,1.59183,2.80979
    75%,Reference,2.65065,2.26185,0.79261,3.61179,3.20153,2.74225
    90%,Inflow,6.02835,24.40655,0.99293,8.00691,6.28345,8.51706
    90%,Outflow,12.43052,23.90022,2.43829,5.66731,2.30348,10.32829
    90%,Reference,12.58278,6.67125,1.2205,4.78255,7.72012,8.57303
    max,Inflow,13.87664,45.97893,1.26657,21.75505,8.88365,163.01001
    max,Outflow,36.58941,47.49381,8.04948,12.39894,4.19118,23.29367
    max,Reference,21.22363,48.23615,1.94442,7.67751,8.75609,10.5095
    """))

    expected = pandas.read_csv(known_csv, index_col=[0, 1]).T
    pdtest.assert_frame_equal(expected, dc.stat_summary(),
                              check_names=False,
                              check_less_precise=True,
                              check_dtype=False)


def test_locations(dc):
    for l in dc.locations:
        assert isinstance(l, Location)
    assert len(dc.locations) == 18
    assert dc.locations[0].definition == {'loc': 'Inflow', 'param': 'A'}
    assert dc.locations[1].definition == {'loc': 'Inflow', 'param': 'B'}


def test_datasets(dc):
    _ds = []
    for d in dc.datasets('Inflow', 'Outflow'):
        assert isinstance(d, Dataset)
        _ds.append(d)
    assert len(_ds) == 6
    assert _ds[0].definition == {'param': 'A'}
    assert _ds[1].definition == {'param': 'B'}


# this sufficiently tests dc._filter_collection
def test_selectLocations(dc):
    locs = dc.selectLocations(param='A', loc=['Inflow', 'Outflow'])
    assert len(locs) == 2
    for n, loc in enumerate(locs):
        assert isinstance(loc, Location)
        assert loc.definition['param'] == 'A'
        if n == 0:
            assert loc.definition['loc'] == 'Inflow'
        elif n == 1:
            assert loc.definition['loc'] == 'Outflow'


def test_selectLocations_squeeze_False(dc):
    locs = dc.selectLocations(param='A', loc=['Inflow'], squeeze=False)
    assert len(locs) == 1
    for n, loc in enumerate(locs):
        assert isinstance(loc, Location)
        assert loc.definition['param'] == 'A'
        assert loc.definition['loc'] == 'Inflow'


def test_selectLocations_squeeze_True(dc):
    loc = dc.selectLocations(param='A', loc=['Inflow'], squeeze=True)
    assert isinstance(loc, Location)
    assert loc.definition['param'] == 'A'
    assert loc.definition['loc'] == 'Inflow'


def test_selectLocations_squeeze_True_None(dc):
    loc = dc.selectLocations(param='A', loc=['Junk'], squeeze=True)
    assert loc is None


# since the test_selectLocations* tests stress _filter_collection
# enough, we'll mock it out for datasets:
def test_selectDatasets(dc):
    with mock.patch.object(dc, '_filter_collection') as _fc:
        with mock.patch.object(dc, 'datasets', return_value=['A', 'B']) as _ds:
            dc.selectDatasets('Inflow', 'Reference', foo='A', bar='C')
            _ds.assert_called_once_with('Inflow', 'Reference')
            _fc.assert_called_once_with(['A', 'B'], foo='A', bar='C',
                                        squeeze=False)
