import sys
import matplotlib
matplotlib.use('agg')

import wqio
status = wqio.test(*sys.argv[1:])
sys.exit(status)
