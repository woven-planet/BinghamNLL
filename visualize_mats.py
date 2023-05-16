# Copyright 2023 TOYOTA MOTOR CORPORATION

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from os.path import basename, splitext


def get_mat_fnames():
    return glob.glob("outputs/mats/*.mat")


def draw_KLD(ax, BNLL_KLD, QCQP_KLD):
    data_len = BNLL_KLD.shape[0]

    ax.grid(axis="y")
    ax.set_xlim(0, data_len)

    ax.plot(np.arange(data_len), BNLL_KLD, label="BNLL", zorder=3)
    ax.plot(np.arange(data_len), QCQP_KLD, label="QCQP", zorder=2)
    ax.plot([0, data_len], [0, 0], "r--", label="Ground Truth", zorder=1)
    ax.legend(bbox_to_anchor=(1, 1), loc="upper right")
    ax.set_xlabel("Number of Iteration", fontsize=13)
    ax.set_ylabel("KL Divergence", fontsize=13)


def draw_bingham_images(ax, BNLL_KLD, QCQP_KLD, init_img, true_img, BNLL_img, QCQP_img):
    init_KLD = BNLL_KLD[0]
    bnll_KLD = BNLL_KLD[-1]
    qcqp_KLD = QCQP_KLD[-1]

    data_len = BNLL_KLD.shape[0]

    # Draw arrows from figure to line
    xys = [(0, init_KLD), (data_len, bnll_KLD), (data_len, qcqp_KLD), (0, 0)]
    xy_boxes = [
        (-data_len * 0.36, init_KLD * 4 / 5),
        (data_len * 1.25, init_KLD * 1 / 5),
        (data_len * 1.25, init_KLD * 4 / 5),
        (-data_len * 0.36, init_KLD * 1 / 5),
    ]
    arr_imgs = [init_img, BNLL_img, QCQP_img, true_img]

    box_colors = ["#3e3e3e", "C0", "C1", "red"]
    arrow_colors = ["#3e3e3e", "C0", "C1", "red"]

    # Add figures
    bbox_extra_artists = []
    for xy, xy_box, arr_img, box_color, arrow_color in zip(
        xys, xy_boxes, arr_imgs, box_colors, arrow_colors
    ):
        arr_img = arr_img[10:-60, 10:-40]

        zoom = 0.3
        imagebox = OffsetImage(arr_img, zoom=zoom)
        imagebox.image.axes = ax

        ab = AnnotationBbox(
            imagebox,
            xy,
            xybox=xy_box,
            xycoords="data",
            pad=0.25,
            arrowprops=dict(arrowstyle="-|>", connectionstyle="bar", color=arrow_color),
            bboxprops=dict(edgecolor=box_color),
        )

        ax.add_artist(ab)
        bbox_extra_artists.append(ab.offsetbox)

    return bbox_extra_artists


def make_KLD_figure(BNLL_KLD, QCQP_KLD, init_img, true_img, BNLL_img, QCQP_img):
    """
    create figure of KLD transition and distribution
    """

    # setting figures
    plt.rc("axes", axisbelow=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_KLD(ax, BNLL_KLD, QCQP_KLD)
    bbox_extra_artists = draw_bingham_images(
        ax, BNLL_KLD, QCQP_KLD, init_img, true_img, BNLL_img, QCQP_img
    )

    return fig, bbox_extra_artists


def save_plot(idx, fname):
    loaded_mat = scipy.io.loadmat(fname)
    randomseed = loaded_mat["common"]["random_seed"][0][0].item()
    # Extract baseline
    fig, bbox_extra_artists = make_KLD_figure(
        loaded_mat["BNLL"]["KLD"][0][0][0],
        loaded_mat["QCQP"]["KLD"][0][0][0],
        loaded_mat["common"]["init_img"][0][0],
        loaded_mat["common"]["truth_img"][0][0],
        loaded_mat["BNLL"]["final_img"][0][0],
        loaded_mat["QCQP"]["final_img"][0][0],
    )

    # Extract arguments
    N_sample = loaded_mat["common"]["gt_quats"][0][0].shape[0]
    data_len = loaded_mat["BNLL"]["KLD"][0][0][0].shape[0]
    # Save figure to outputs
    # NOTE: bbox_extra_artists is only available in savefig
    ofname = "outputs/pdfs/kld_plot_matname{}_N{}_iter{}_seed{}.pdf".format(
        splitext(basename(fname))[0], N_sample, data_len, randomseed
    )
    fig.savefig(ofname, bbox_inches="tight", bbox_extra_artists=bbox_extra_artists, dpi=400)
    print("  saved: ", ofname)


def save_all():
    mat_fnames = get_mat_fnames()
    len_fnames = len(mat_fnames)
    for idx, fname in enumerate(mat_fnames):
        print('{}/{}: processing "{}"...'.format(idx + 1, len_fnames, fname))
        save_plot(idx, fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "-I", metavar="inputname", type=str, default=None, help="load fname of matfile"
    )

    args = parser.parse_args()
    if args.I is not None:
        save_plot(-1, args.I)
    else:
        save_all()
