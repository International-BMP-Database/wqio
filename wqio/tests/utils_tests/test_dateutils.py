from datetime import datetime
import warnings

import pytest
import pandas

from wqio.utils import dateutils


@pytest.mark.parametrize(
    "datemaker", [pandas.Timestamp, lambda x: datetime.strptime(x, "%Y-%m-%d")]
)
@pytest.mark.parametrize(
    ("datestring", "expected"),
    [
        ("1998-12-25", "winter"),
        ("2015-03-22", "spring"),
        ("1965-07-04", "summer"),
        ("1982-11-24", "autumn"),
    ],
)
def test_getSeason(datemaker, datestring, expected):
    date = datemaker(datestring)
    season = dateutils.getSeason(date)
    assert season == expected


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"sampledate": "2012-05-25", "sampletime": "16:54"}, "2012-05-25 16:54"),
        ({"sampledate": None, "sampletime": "16:54"}, "1901-01-01 16:54"),
        ({"sampledate": "2012-05-25", "sampletime": None}, "2012-05-25 00:00"),
        ({"sampledate": None, "sampletime": None}, "1901-01-01 00:00"),
    ],
)
def test_makeTimestamp_basic(row, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        tstamp = dateutils.makeTimestamp(row)
        assert tstamp == pandas.Timestamp(expected)


def test_makeTimestamp_customcols():
    row = {"mydate": "2012-05-25", "mytime": "16:54"}
    tstamp = dateutils.makeTimestamp(row, datecol="mydate", timecol="mytime")
    assert tstamp == pandas.Timestamp("2012-05-25 16:54")


@pytest.mark.parametrize(
    "datemaker", [pandas.Timestamp, lambda x: datetime.strptime(x, "%Y-%m-%d")]
)
@pytest.mark.parametrize("datestring", ["2005-10-02", "2006-09-02"])
def test_getWaterYear(datemaker, datestring):
    date = datemaker(datestring)
    wateryear = dateutils.getWaterYear(date)
    assert wateryear == "2005/2006"
