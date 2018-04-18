
import logging, itertools, os
from datetime import date

import astropy.io.ascii as at
import matplotlib.pyplot as plt

from k2spin.config import *
from k2spin import plot

today = date.today().isoformat()

def plot_list(results_list):
    """
    """
    res = at.read(base_path+"tables/"+results_list)

    f = open("/home/stephanie/my_papers/hyadesk2/figure_sets/f9.tbl","w")

    count = 1
    for i, epic in enumerate(res["EPIC"]):
        logging.info(epic)
        outfilename = "ktwo{0}-c04_lc_analysis.png".format(epic)
        plot.paper_lcs(epic,res[i])
        plt.savefig(base_path+"plot_outputs/"+outfilename,bbox_inches="tight")

        if ((epic==210736105) or (epic==2107361050) or 
            (epic==210963067) or (epic==2109630670)):
            continue
        elif epic==2107361051:
            save_epic = 210736105
        elif epic==2109630671:
            save_epic = 210963067
        else:
            save_epic = epic
        figsetname = "f9_{0}.eps".format(count)
        f.write("{0} & EPIC {1}\n".format(figsetname,save_epic))
        plt.savefig("/home/stephanie/my_papers/hyadesk2/figure_sets/"+figsetname,bbox_inches="tight")
        plt.close("all")
        count += 1

    f.close()
        

if __name__=="__main__":

    plot_list("c4_lcs_aps_results_2015-10-17_comments.csv")

"""
    lc_file = "ktwo210408563-c04.csv"
    epic = "210408563"
    ap = 5

    res = at.read(base_path+"tables/c4_lcs_aps_results_2015-10-17.csv")
    plot.paper_lcs(epic,res[4])
    plt.savefig("/home/stephanie/my_papers/hyadesk2/sample_lc.eps",
                bbox_inches="tight")
    plt.savefig("/home/stephanie/Dropbox/plots_for_sharing/sample_lc.png",
                bbox_inches="tight")
"""
