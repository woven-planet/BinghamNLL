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
Loss Function (Quadratically Constraind Quadratic Program)
This module uses third-party library.
https://github.com/utiasSTARS/bingham-rotation-learning
"""

import losses as brl_losses
import qcqp_layers as brl_qcqp_layers
import torch


class QCQPLoss(torch.nn.Module):
    def forward(self, A_vec_pred, q_target, reduce=True):
        """
        Args:
            A_vec_pred (torch.tensor): (size: B x 10)
            q_target (torch.tensor): (size: B x 4)
        Returns:
            torch.tensor: loss (size: 0 or B)
        """
        # extract eigenvector corresponding to the maximum eigenvalue
        q_pred = brl_qcqp_layers.A_vec_to_quat(-A_vec_pred)
        losses = brl_losses.quat_chordal_squared_loss(q_pred, q_target, reduce=reduce)
        loss = losses.mean() if reduce else losses
        return loss
