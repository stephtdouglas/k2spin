"""Test computation of lightcurve statistics."""

import k2spin.stats as stats

def test_stats():
    time, flux, u_flux = np.arange(10),np.ones(10),np.ones(10)

    med, std = stats.stats(time, flux, u_flux)

    if med!=1:
        logging.warning("stats median failed")
    if std!=1:
        logging.warning("stats stdev failed")


