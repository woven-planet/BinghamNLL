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

"""
Kullback-Leibler Divergence for Bingham distribution
"""

import bingham.math.normconst as nc
import numpy as np


def calc_KLD(A_truth, A_estim):
    """Kullback-Leibler Divergence

    ln(C(Lambda_estim) / C(Lambda_truth)) - tr((A_estim - A_truth) * S)

    S = M_truth *
    diag(1/C(Lambda_truth) * d/dLambda C(Lambda_truth)) * M_truth.T

    Args:
        A_truth (numpy.array): Ground truth
        A_estim (numpy.array): Estimated

    Returns:
        float: Kullback-Leibler Divergence

    """

    # Diagonalize
    Z_truth, M_truth = np.linalg.eigh(A_truth)
    Z_estim, _ = np.linalg.eigh(A_estim)

    # Calculate normalizing constants
    def calc_NC_shifted(Z):
        return nc.calc_constant(Z - Z.max()).item()

    NC_truth_shifted = calc_NC_shifted(Z_truth)
    NC_estim_shifted = calc_NC_shifted(Z_estim)

    # Calculate covariance part
    omegas_truth = nc.calc_Dconstant(Z_truth - Z_truth.max())[0] / NC_truth_shifted
    cov_part = M_truth @ np.diag(omegas_truth) @ M_truth.T
    NC_ratio = np.log(NC_estim_shifted / NC_truth_shifted) + (Z_estim.max() - Z_truth.max())

    A_diff = A_estim - A_truth
    return NC_ratio - np.trace(A_diff @ cov_part)
