import uproot as up
import awkward as ak
import numpy as np
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
import hist
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import mplhep as hep
import argparse
import pathlib
import shutil
import PIL


parser = argparse.ArgumentParser(description="manual to this script")
parser.add_argument("-i", "--input", help="input coffea file", default="rlts.coffea")
parser.add_argument("-o", "--output", help="output folder", default="outplots")
parser.add_argument("-r", "--rebin", help="rebin histogram", default="1")
args = parser.parse_args()


def get_ratio(plt_nom, plt_denom):
    val_nom = plt_nom.values()
    var_nom = plt_nom.variances()
    val_denom = plt_denom.values()
    var_denom = plt_denom.variances()

    ratio_val_denom = val_denom / val_denom
    ratio_val_denom = np.clip(ak.nan_to_num(ratio_val_denom, nan=1), -9999.0, 9999.0)
    ratio_err_denom = np.sqrt(var_denom) / val_denom
    ratio_err_denom = np.clip(ak.nan_to_num(ratio_err_denom, nan=1), -9999.0, 9999.0)

    ratio_val_nom = val_nom / val_denom
    ratio_val_nom = np.clip(ak.nan_to_num(ratio_val_nom, nan=1), -9999.0, 9999.0)
    ratio_err_nom = np.sqrt(var_nom) / val_denom
    ratio_err_nom = np.clip(ak.nan_to_num(ratio_err_denom, nan=1), -9999.0, 9999.0)

    return ratio_val_denom, ratio_err_denom, ratio_val_nom, ratio_err_nom


def plotting(plts, outfolder, rebin):
    for i in plts["official"]:
        # if not i=="h_gen_j1_mass":
        #     continue
        print("[ Plotting ] ===>", i)
        f, ax = plt.subplots(
            2,
            1,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": [3.3, 0.7]},
            sharex=True,
        )
        plt.style.use(hep.style.CMS)
        if (plts_normed["official"][i].axes.size[0] > 5 * int(rebin)) and (
            plts_normed["official"][i].axes.size[0]
        ) % int(rebin) == 0:
            plts_normed["official"][i] = plts_normed["official"][i][
                :: hist.rebin(int(rebin))
            ]
            plts_normed["condor"][i] = plts_normed["condor"][i][
                :: hist.rebin(int(rebin))
            ]
        plts_normed["official"][i].plot(
            ax=ax[0], label="Central Prod.", flow=False, color="r"
        )
        plts_normed["condor"][i].plot(
            ax=ax[0], label="Private Prod.", flow=False, color="b"
        )
        ratio_val_denom, ratio_err_denom, ratio_val_nom, ratio_err_nom = get_ratio(
            plts_normed["condor"][i], plts_normed["official"][i]
        )
        xaxis_edges = plts_normed["official"][i].axes.edges[0]
        # print("smr1")
        # print(len(ratio_val_denom),ratio_val_denom)
        # print(len(ratio_err_denom),ratio_err_denom)
        # print(len(ratio_val_nom),ratio_val_nom)
        # print(len(ratio_err_nom),ratio_err_nom)
        # print(len(xaxis_edges),xaxis_edges)
        new_ratio_val_denom = []
        new_edges = []
        new_up = []
        new_down = []
        for j, jval in enumerate(ratio_val_denom):
            new_ratio_val_denom.append(jval)
            new_ratio_val_denom.append(jval)
            new_up.append(jval + np.abs(ratio_err_denom[j]))
            new_up.append(jval + np.abs(ratio_err_denom[j]))
            new_down.append(jval - np.abs(ratio_err_denom[j]))
            new_down.append(jval - np.abs(ratio_err_denom[j]))

            new_edges.append(xaxis_edges[j])
            new_edges.append(xaxis_edges[j + 1])
        # print("smr2")
        # print(len(new_edges),new_edges)
        # print(len(new_up),new_up)
        # print(len(new_down),new_down)
        ax[1].plot(new_edges, new_up, color="r", linewidth=0.5, alpha=0.6)
        ax[1].plot(new_edges, new_down, color="r", linewidth=0.5, alpha=0.6)
        ax[1].fill_between(new_edges, new_up, new_down, color="r", alpha=0.3)

        ax[0].legend()
        ax[1].set_xlabel(ax[0].get_xlabel())
        ax[0].set_xlabel("")
        ax[0].set_ylabel("A.U.")
        ax[0].set_xlim(xaxis_edges[0], xaxis_edges[-1])

        ax[1].axhline(y=1, color="r", linestyle="--")
        ax[1].set_ylim(0.972, 1.028)
        # hep.histplot(ratio_val_denom,bins=xaxis_edges,yerr=np.abs(ratio_err_denom), histtype='errorbar', stack=False, color="r", ax=ax[1], marker='.', markersize=0,elinewidth=0.5)
        hep.histplot(
            ratio_val_nom,
            bins=xaxis_edges,
            yerr=np.abs(ratio_err_nom),
            histtype="errorbar",
            stack=False,
            color="b",
            ax=ax[1],
            marker="o",
            markersize=2,
            elinewidth=1,
        )
        hep.cms.label(
            label="Preliminary", loc=0, data=False, ax=ax[0], com=13.6
        )  # Preliminary
        plt.subplots_adjust(hspace=0.03)
        # plt.tight_layout()
        print("[ Plotting ]", f"{outfolder}/{i}.png")
        # https://stackoverflow.com/questions/3938676/python-save-matplotlib-figure-on-an-pil-image-object
        # https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image
        f.canvas.draw()
        pil_image = PIL.Image.frombytes(
            "RGB", f.canvas.get_width_height(), f.canvas.tostring_rgb()
        )
        pil_image.save(f"{outfolder}/{i}.png")
        pil_image.save(f"{outfolder}/{i}.pdf")
        # plt.savefig(f"{outfolder}/{i}.png", bbox_inches="tight")
        # plt.savefig(f"{outfolder}/{i}.pdf", bbox_inches="tight")
        # break
    return True


if __name__ == "__main__":
    rlt = coffea.util.load(args.input)
    plt_keys = [i for i in rlt.keys() if i.startswith("h_")]
    other_keys = [i for i in rlt.keys() if not i.startswith("h_")]

    norm_off = rlt["count"]["official"]["neff"]
    norm_loc = rlt["count"]["condor"]["neff"]
    print(
        """[ Plotting ] Official:
    {}
    """.format(
            rlt["count"]["official"]
        )
    )
    print(
        """[ Plotting ] Private:
    {}
    """.format(
            rlt["count"]["condor"]
        )
    )

    plts_normed = {}
    plts_normed["official"] = {}
    plts_normed["condor"] = {}
    for i in plt_keys:
        # print(i,rlt[i])
        try:
            plts_normed["official"][i] = rlt[i][{"dataset": "official"}] / norm_off
            plts_normed["condor"][i] = rlt[i][{"dataset": "condor"}] / norm_loc
        except:
            pass

    p = pathlib.Path(args.output)
    try:
        p.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("[ Plotting ] Recreating folder {}".format(args.output))
        shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=False)
    else:
        pass

    plotting(plts_normed, args.output, args.rebin)

# python doPlot.py -i rlts_full3.coffea -o plots
