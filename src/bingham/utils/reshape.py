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
Reshape functions for Bingham distribution A 10D <-> 4x4
"""

import numpy as np
import torch


def convert_4x4mat_to_10array(symmat_4x4):
    """Convert from 4x4 matrix to 10D array

    Args:
        symmat_4x4 (numpy.array): 4x4 matrix

    Returns:
        numpy.array: 10D array
    """
    i_idces, j_idces = np.triu_indices(4)
    return symmat_4x4[i_idces, j_idces]


def convert_10array_to_4x4mat(array_10param):
    """Convert from 10D array to 4x4 matrix

    Args:
        array_10param (numpy.array):  10D array

    Returns:
        numpy.array: 4x4 matrix
    """
    batchsize = array_10param.shape[0]
    i_idces, j_idces = np.triu_indices(4)

    resulting_mat = np.zeros((batchsize, 4, 4))
    resulting_mat[:, i_idces, j_idces] = array_10param
    resulting_mat[:, j_idces, i_idces] = array_10param

    return resulting_mat


def from10D_to4x4_numpy(array_10param, evls=False, shift=False, reduced=False):
    """Convert from 10D array to 4x4 matrix

    Args:
        array_10param (numpy.array): array of 10D array
        evls (boolean): If True, this function returns eigenvalues
        shift (boolean): If True, maximum eigenvalue is subtracted from A
        reduced (boolean): If True, this function returns single matrix

    Returns:
        (numpy.array, numpy.array):
            array of 4x4 matrix A, and its eigenvalues (If evls)

    """
    # add first dim if batchsize is zero
    array_10param = unsqueeze_numpy(array=array_10param)

    if evls or shift or reduced:
        resulting_mat, evls = from10D_to4x4_torch(
            torch.as_tensor(array_10param), evls=evls, shift=shift, reduced=reduced
        )
        resulting_mat = resulting_mat.detach().numpy()
        evls = evls.detach().numpy() if evls is not None else None
    else:
        resulting_mat = convert_10array_to_4x4mat(array_10param)
        evls = None

    return resulting_mat, evls


def from10D_to4x4_torch(A_vec, evls=False, shift=False, reduced=False):
    """Convert from 10D array to 4x4 matrix

    Args:
        array_10param (torch.tensor): array of 10D array
        evls (boolean): If True, this function returns eigenvalues
        shift (boolean): If True, maximum eigenvalue is subtracted from A
        reduced (boolean): If True, this function returns single matrix

    Returns:
        (torch.tensor, torch.tensor):
            array of 4x4 matrix A, and its eigenvalues (If evls)

    """

    def Avec_to_Amat(A_vec):
        A_vec = unsqueeze_torch(array=A_vec)
        idx = torch.triu_indices(4, 4)
        A = A_vec.new_zeros((A_vec.shape[0], 4, 4))
        A[:, idx[0], idx[1]] = A_vec
        A[:, idx[1], idx[0]] = A_vec
        return A

    A_result = Avec_to_Amat(A_vec)

    if shift or evls:
        eigvals, _ = torch.linalg.eigh(A_result)
        evl_max = eigvals[:, -1].unsqueeze(1)
        if shift:
            A_result = A_result - torch.diag_embed(torch.ones_like(eigvals) * evl_max)

    if reduced:
        A_result = A_result.squeeze()

    if evls:
        return A_result, eigvals - evl_max
    else:
        return A_result, None


def unsqueeze_numpy(array=None, mats=None):
    """Add first dim if batchsize is zero,
    like torch.unspueeze.

    NOTE:
        This function is exclusive and returns either an array or a matrix
        which argument is not None.

    Args:
        array (numpy.array): 10D array
        mats (numpy.array): 4x4 matrix

    Returns:
        numpy.array: Unsqueezed array

    """
    if array is not None and (len(array.shape) < 2):
        return array[None, :]
    if mats is not None and (len(mats.shape) < 3):
        return mats[None, :]


def unsqueeze_torch(array=None, mats=None):
    """Add first dim if batchsize is zero,
    this is equivalent to torch.unspueeze.

    NOTE:
        This function is exclusive and returns either an array or a matrix
        which argument is not None.

    Args:
        array (torch.tensor): 10D array
        mats (torch.tensor): 4x4 matrix

    Returns:
        torch.tensor: Unsqueezed array

    """
    if array is not None:
        if array.dim() < 2:
            return array.unsqueeze(dim=0)
        else:
            return array
    if mats is not None:
        if mats.dim() < 3:
            return mats.unsqueeze(dim=0)
        else:
            return mats
