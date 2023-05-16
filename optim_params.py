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

import bingham.visualize.SO3s as vSO3
import copy
import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import time
import torch
from bingham.distribution import BinghamDistribution
from bingham.losses.nll import BinghamLoss
from bingham.losses.qcqp import QCQPLoss
from bingham.math.kld import calc_KLD
from bingham.math.quaternion import quat_mat
from bingham.math.sampler import BinghamSampler
from bingham.utils.reshape import from10D_to4x4_numpy, from10D_to4x4_torch
from PIL import Image
from torch.autograd import Variable
from visualize_mats import draw_KLD, make_KLD_figure


def optim_unit(optimizer, varA_torch, A_truth_ndarray, lossfunc, q_targs):
    optimizer.zero_grad()
    var_A_padded = varA_torch.repeat(q_targs.shape[0], 1)
    loss = lossfunc.forward(var_A_padded, q_targs)

    loss.backward()

    A_estim_torch, _ = from10D_to4x4_torch(varA_torch.detach(), reduced=True)
    KLD = calc_KLD(A_truth_ndarray, A_estim_torch.numpy())

    optimizer.step()

    return loss, KLD


def create_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%H%M%S_%Y%m%d")


def main(
    N_sampling=20,
    fname=None,
    rand_seed=None,
    dryrun=False,
    show=False,
    init_scale=10,
    maxiter=20000,
    device="cuda",
):
    bs = BinghamSampler(dim=3)

    # rand_seed is set only for specifying the truth distribution
    next_seed = np.random.randint(2**32 - 1)
    if rand_seed is None:
        # explicitly set random seed for reproducibility
        rand_seed = np.random.randint(2**32 - 1)

    np.random.seed(rand_seed)

    V = quat_mat(np.random.randn(4))
    Z = np.cumsum(np.random.rand(4)) * 500
    Z = np.diag(Z - Z.max())
    A_truth = V @ Z @ V.T
    init_A = np.random.rand(10) * init_scale
    init_A_saved = copy.copy(init_A)

    # Randomize seed here.
    # Note that next_seed is randomly generated integer
    # *before* specifying the seed.
    np.random.seed(next_seed)

    q_targs_array = bs(A_truth, N_sampling)[0]

    # Define lossfuncs
    blossfunc = BinghamLoss()
    qlossfunc = QCQPLoss()

    # Convert to Pytorch tensor
    q_targs = Variable(torch.from_numpy(q_targs_array)).to(device).type(torch.DoubleTensor)

    # Initialize Parameters
    A0 = torch.from_numpy(init_A)
    A_bnll = Variable(A0).to(device).type(torch.DoubleTensor).requires_grad_()
    A_qcqp = Variable(A0 * 1.0).to(device).type(torch.DoubleTensor).requires_grad_()

    # Initialize optimizer
    optimizer_bnll = torch.optim.RMSprop([A_bnll])
    optimizer_qcqp = torch.optim.RMSprop([A_qcqp])

    Z_INIT, M_INIT = np.linalg.eigh(from10D_to4x4_numpy(A0.detach().numpy(), reduced=True)[0])
    Z_TRUE, M_TRUE = np.linalg.eigh(A_truth)
    print("True Z = ", Z_TRUE)
    print("True M = ", M_TRUE)

    A_bnlls = []
    A_qcqps = []
    bloss_KLDs = []
    qloss_KLDs = []
    blosses = []
    qlosses = []

    for t in range(maxiter):
        try:
            bloss, bloss_KLD = optim_unit(optimizer_bnll, A_bnll, A_truth, blossfunc, q_targs)
            qloss, qloss_KLD = optim_unit(optimizer_qcqp, A_qcqp, A_truth, qlossfunc, q_targs)

            if t % 200 == 0:
                print(t, ": BNLL", bloss.item(), "KLD", bloss_KLD)
                print(t, ": QCQP", qloss.item(), "KLD", qloss_KLD)

            A_bnlls.append(copy.copy(A_bnll.detach().numpy()))
            A_qcqps.append(copy.copy(A_qcqp.detach().numpy()))
            bloss_KLDs.append(bloss_KLD)
            qloss_KLDs.append(qloss_KLD)
            blosses.append(bloss.item())
            qlosses.append(qloss.item())

        except KeyboardInterrupt:
            break

    print("=== final result ===")
    print(t, ": BNLL", bloss.item(), "KLD", bloss_KLD)
    print(t, ": QCQP", qloss.item(), "KLD", qloss_KLD)
    print("===")
    Z_BNLL, M_BNLL = np.linalg.eigh(from10D_to4x4_numpy(A_bnll.detach().numpy(), reduced=True)[0])
    Z_QCQP, M_QCQP = np.linalg.eigh(from10D_to4x4_numpy(A_qcqp.detach().numpy(), reduced=True)[0])
    print("TRUE Z = ", Z_TRUE)
    print("TRUE M = \n", M_TRUE)
    print("BNLL Z = ", Z_BNLL)
    print("BNLL M = \n", M_BNLL)
    print("QCQP Z = ", Z_QCQP)
    print("QCQP M = \n", M_QCQP)

    print("*** NOTE ***")
    print("- Bingham distribution p(x) is propotional to exp(x^T A x).")
    print("- M, Z satisfies that A = M diag(Z) M^T, where")
    print("  - A is 4x4 symmetric matrix.")
    print("  - M is 4x4 orthogonal matrix.")
    print("  - diag(Z) is 4x4 diagnonal matrix whose entries are Z.")
    print("************")

    if show or not dryrun:
        timestampstr = create_timestamp()

        qt_gt = M_TRUE[:, -1]
        bd_init = BinghamDistribution(Z=Z_INIT, M=M_INIT)
        bd_true = BinghamDistribution(Z=Z_TRUE, M=M_TRUE)
        bd_bnll = BinghamDistribution(Z=Z_BNLL, M=M_BNLL)
        bd_qcqp = BinghamDistribution(Z=Z_QCQP, M=M_QCQP)

        # save images
        if not dryrun:
            init_imgarray = vSO3.draw_bingham_3d(bd_init, qt_gt, num_samples=1000, probability=1.0)
            truth_imgarray = vSO3.draw_bingham_3d(bd_true, qt_gt, num_samples=1000, probability=1.0)
            bnll_imgarray = vSO3.draw_bingham_3d(bd_bnll, qt_gt, num_samples=1000, probability=1.0)
            qcqp_imgarray = vSO3.draw_bingham_3d(bd_qcqp, qt_gt, num_samples=1000, probability=1.0)
            Image.fromarray(init_imgarray).save("outputs/pngs/init.png")
            Image.fromarray(truth_imgarray).save("outputs/pngs/truth.png")
            Image.fromarray(bnll_imgarray).save("outputs/pngs/bnll.png")
            Image.fromarray(qcqp_imgarray).save("outputs/pngs/qcqp.png")
            print("figs saved to outputs/pngs/*.png")
        else:
            print("figs save skipped.")

        # save mat
        if not dryrun:
            if fname is None:
                fname = "{}.mat".format(timestampstr)
            matpath = "outputs/mats/" + fname

            results = {
                "common": {
                    "gt_quats": q_targs_array,
                    "truth_img": truth_imgarray,
                    "init_paramA": init_A_saved,
                    "init_img": init_imgarray,
                    "truth_paramA": A_truth,
                    "random_seed": rand_seed,
                },
                "BNLL": {
                    "paramA": A_bnlls,
                    "KLD": bloss_KLDs,
                    "loss": blosses,
                    "final_img": bnll_imgarray,
                },
                "QCQP": {
                    "paramA": A_qcqps,
                    "KLD": qloss_KLDs,
                    "loss": qlosses,
                    "final_img": qcqp_imgarray,
                },
            }

            scipy.io.savemat(matpath, results)
            print("mat saved to {}".format(matpath))
        else:
            print("mat save skipped")

        if not dryrun:
            # KLD figure
            fig, bbox_extra_artists = make_KLD_figure(
                np.asanyarray(bloss_KLDs),
                np.asanyarray(qloss_KLDs),
                init_imgarray,
                truth_imgarray,
                bnll_imgarray,
                qcqp_imgarray,
            )
            pdfpath = "outputs/pdfs/kld_plot_matname{}_N{}_iter{}_seed{}.pdf".format(
                timestampstr, N_sampling, maxiter, rand_seed
            )
            fig.savefig(
                pdfpath, bbox_inches="tight", bbox_extra_artists=bbox_extra_artists, dpi=400
            )
            plt.close(fig)
            print("pdf saved to {}".format(pdfpath))
        else:
            print("pdf save skipped.")

        if show:
            fig = plt.figure(figsize=(20, 10))
            ax_init = fig.add_subplot(2, 5, 1, projection="3d")
            ax_init.set_title("Init")
            ax_truth = fig.add_subplot(2, 5, 6, projection="3d")
            ax_truth.set_title("Ground Truth")
            ax_qcqp = fig.add_subplot(2, 5, 5, projection="3d")
            ax_qcqp.set_title("QCQP")
            ax_bnll = fig.add_subplot(2, 5, 10, projection="3d")
            ax_bnll.set_title("BNLL")
            ax_kld = fig.add_subplot(2, 5, (2, 9))

            vSO3.draw_bingham_distribution(
                ax_init, bd_init, qt_gt, num_samples=1000, probability=1.0
            )
            vSO3.draw_bingham_distribution(
                ax_truth, bd_true, qt_gt, num_samples=1000, probability=1.0
            )
            vSO3.draw_bingham_distribution(
                ax_qcqp, bd_qcqp, qt_gt, num_samples=1000, probability=1.0
            )
            vSO3.draw_bingham_distribution(
                ax_bnll, bd_bnll, qt_gt, num_samples=1000, probability=1.0
            )
            draw_KLD(ax_kld, np.asanyarray(bloss_KLDs), np.asanyarray(qloss_KLDs))

            plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-N", metavar="N_sampling", type=int, default=20, help="sampling number of quaternion"
    )
    parser.add_argument(
        "-O", metavar="output_fname", type=str, default=None, help="save name of matfile"
    )
    parser.add_argument(
        "--init_scale",
        metavar="init_scale",
        type=int,
        default=10,
        help="scale factor for initial A",
    )
    parser.add_argument(
        "--maxiter", metavar="max_iteration", type=int, default=20000, help="max iteration"
    )
    parser.add_argument(
        "--seed", metavar="random_seed", type=int, default=None, help="seed for randoms"
    )
    parser.add_argument("--dryrun", action="store_true", help="save matfile or not")
    parser.add_argument("--show", action="store_true", help="show result or not")
    parser.add_argument("--gpu", action="store_true", help="use cuda")

    args = parser.parse_args()
    if args.gpu:
        device = "cuda"
    else:
        device = "cpu"

    if args.gpu:
        torch.cuda.synchronize()
    tm0 = time.time()

    main(
        N_sampling=args.N,
        fname=args.O,
        rand_seed=args.seed,
        dryrun=args.dryrun,
        show=args.show,
        init_scale=args.init_scale,
        maxiter=args.maxiter,
        device=device,
    )

    if args.gpu:
        torch.cuda.synchronize()
    tm1 = time.time()
    print("elappsed time: ", tm1 - tm0)
