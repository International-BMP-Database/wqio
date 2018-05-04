import sys
import matplotlib
from matplotlib import style


matplotlib.use('agg')
style.use('classic')

import wqio


if '--strict' in sys.argv:
    sys.argv.remove('--strict')
    status = wqio.teststrict(*sys.argv[1:])
else:
    status = wqio.test(*sys.argv[1:])

sys.exit(status)
