import datetime
import warnings

import nose.tools as nt
import pandas

from wqio.utils import dateutils


class _base_getSeason(object):
    def setup(self):
        self.winter = self.makeDate('1998-12-25')
        self.spring = self.makeDate('2015-03-22')
        self.summer = self.makeDate('1965-07-04')
        self.autumn = self.makeDate('1982-11-24')

    def test_winter(self):
        nt.assert_equal(dateutils.getSeason(self.winter), 'winter')

    def test_spring(self):
        nt.assert_equal(dateutils.getSeason(self.spring), 'spring')

    def test_summer(self):
        nt.assert_equal(dateutils.getSeason(self.summer), 'summer')

    def test_autumn(self):
        nt.assert_equal(dateutils.getSeason(self.autumn), 'autumn')


class test_getSeason_Datetime(_base_getSeason):
    @nt.nottest
    def makeDate(self, date_string):
        return datetime.datetime.strptime(date_string, '%Y-%m-%d')


class test_getSeason_Timestamp(_base_getSeason):
    @nt.nottest
    def makeDate(self, date_string):
        return pandas.Timestamp(date_string)


class test_makeTimestamp(object):
    def setup(self):
        warnings.resetwarnings()
        warnings.simplefilter("always")
        self.known_tstamp = pandas.Timestamp('2012-05-25 16:54')
        self.known_tstamp_fbdate = pandas.Timestamp('1901-01-01 16:54')
        self.known_tstamp_fbtime = pandas.Timestamp('2012-05-25 00:00')
        self.known_tstamp_fbboth = pandas.Timestamp('1901-01-01 00:00')

    def teardown(self):
        warnings.resetwarnings()

    def test_default_cols(self):
        row = {'sampledate': '2012-05-25', 'sampletime': '16:54'}
        tstamp = dateutils.makeTimestamp(row)
        nt.assert_equal(self.known_tstamp, tstamp)

    def test_custom_cols(self):
        row = {'mydate': '2012-05-25', 'mytime': '16:54'}
        tstamp = dateutils.makeTimestamp(row, datecol='mydate', timecol='mytime')
        nt.assert_equal(self.known_tstamp, tstamp)

    def test_fallback_date(self):
        row = {'sampledate': None, 'sampletime': '16:54'}
        tstamp = dateutils.makeTimestamp(row)
        nt.assert_equal(self.known_tstamp_fbdate, tstamp)

    def test_fallback_time(self):
        row = {'sampledate': '2012-05-25', 'sampletime': None}
        tstamp = dateutils.makeTimestamp(row)
        nt.assert_equal(self.known_tstamp_fbtime, tstamp)

    def test_fallback_both(self):
        row = {'sampledate': None, 'sampletime': None}
        tstamp = dateutils.makeTimestamp(row)
        nt.assert_equal(self.known_tstamp_fbboth, tstamp)


class test_getWaterYear(object):
    def setup(self):
        self.earlydate = datetime.datetime(2005, 10, 2)
        self.latedate = datetime.datetime(2006, 9, 2)
        self.known_wateryear = '2005/2006'

    def test_early_dt(self):
        nt.assert_equal(dateutils.getWaterYear(self.earlydate), self.known_wateryear)

    def test_late_dt(self):
        nt.assert_equal(dateutils.getWaterYear(self.latedate), self.known_wateryear)

    def test_early_tstamp(self):
        date = pandas.Timestamp(self.earlydate)
        nt.assert_equal(dateutils.getWaterYear(date), self.known_wateryear)

    def test_late_tstamp(self):
        date = pandas.Timestamp(self.latedate)
        nt.assert_equal(dateutils.getWaterYear(date), self.known_wateryear)
