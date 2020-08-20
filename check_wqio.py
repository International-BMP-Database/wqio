import sys
import matplotlib
from matplotlib import style

import wqio

if "--strict" in sys.argv:
    sys.argv.remove("--strict")
    tester = wqio.teststrict
else:
    tester = wqio.test

sys.exit(tester(*sys.argv[1:]))
