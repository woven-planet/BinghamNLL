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
Loss Function (negative log-likelihood of Bingham distribution)
"""

import bingham.math.normconst as bingham_normconst
import numpy as np
import torch
from bingham.utils.reshape import from10D_to4x4_torch


def detach(tensor):
    return tensor.detach().cpu()


class BinghamNormConst(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Lambdas):
        ctx.save_for_backward(Lambdas)
        return torch.as_tensor(
            bingham_normconst.calc_constant(np.asarray(detach(Lambdas))), device=Lambdas.device
        )

    @staticmethod
    def backward(ctx, grad_output):
        Lambdas = ctx.saved_tensors[0]
        return (
            torch.as_tensor(
                bingham_normconst.calc_Dconstant(np.asarray(detach(Lambdas))), device=Lambdas.device
            )
            * grad_output
        )


class BinghamLoss(torch.nn.Module):
    def forward(self, A_vec_pred, q_target, reduce=True):
        """
        Args:
            A_vec_pred (torch.tensor): (size: B x 10)
            q_target (torch.tensor): (size: B x 4)
        Returns:
            torch.tensor: loss (size: 0 or B)
        """
        if A_vec_pred.dim() < 2:
            A_vec_pred = A_vec_pred.unsqueeze(0)
        if q_target.dim() < 2:
            q_target = q_target.unsqueeze(0)

        A_pred, Lambdas = from10D_to4x4_torch(A_vec_pred, reduced=False, evls=True, shift=True)
        normalizing_constant = BinghamNormConst.apply(Lambdas)
        qT_A_q = torch.einsum("bn,bnm,bm->b", q_target, A_pred, q_target)
        losses = -1.0 * qT_A_q + torch.log(normalizing_constant).squeeze()
        loss = losses.mean() if reduce else losses

        return loss
