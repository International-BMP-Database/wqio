import wqio
import nose.tools as nt

def test_api():
    nt.assert_true(hasattr(wqio, 'ros'))
    nt.assert_true(hasattr(wqio, 'bootstrap'))
    nt.assert_true(hasattr(wqio, 'Location'))
    nt.assert_true(hasattr(wqio, 'Dataset'))
    nt.assert_true(hasattr(wqio, 'DataCollection'))
    nt.assert_true(hasattr(wqio, 'utils'))
