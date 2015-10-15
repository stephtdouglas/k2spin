import logging, os

import matplotlib 

#logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.DEBUG)

if os.path.exists("/home/stephanie/Dropbox/")==True:
    base_path = "/home/stephanie/code/python/k2spin/"
    logging.warning("k2spin on jaina")
else:
    base_path = "/vega/astro/users/sd2706/k2/"
    logging.warning("Working on Yeti")
    matplotlib.use("agg")
