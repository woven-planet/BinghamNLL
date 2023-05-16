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
Quaternion functions
"""

import numpy as np


def quat_mat(qt):
    """Convert from quaternion to skew symmetry matrix

    Args:
        qt (numpy.array): Quaternion

    Returns:
        numpy.array: Skew symmetry matrix

    """
    w, x, y, z = qt / np.linalg.norm(qt)
    V = np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])
    return V


def diff_quats(q_estim, q_truth):
    """Difference of quaternions

    Args:
        q_estim (numpy.array): Quaternion
        q_truth (numpy.array): Quaternion

    Returns:
        float: Difference of quaternions

    """

    # check np.dot(q_estim, q_truth) <= 1.0
    iprod = np.abs(np.dot(q_estim, q_truth))
    if iprod > 1.0:
        iprod = 1.0
    return 2 * np.arccos(iprod)


def average_quat_Frob(sampled_quats):
    """average quaternion that minimizes Frobenius norm

    https://users.cecs.anu.edu.au/~hartley/Papers/PDF/Hartley-Trumpf:Rotation-averaging:IJCV.pdf

    Args:
        sampled_quats (numpy.array): Quaternions

    Returns:
        numpy.array: Average quaternion

    """
    qi_qiT = np.einsum("bi,bj->bij", sampled_quats, sampled_quats)
    M = np.sum(qi_qiT.reshape(-1, 16), axis=0).reshape(4, 4)
    _, V = np.linalg.eigh(M)
    return V[:, -1]
