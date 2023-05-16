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

import torch
import unittest
from bingham.losses.nll import BinghamLoss
from bingham.losses.qcqp import QCQPLoss

from .bingham_base import BinghamTestBase


def check_installation(lossfunc):
    """
    Check if the installation is successful
    """
    # Generate random A and q_gt
    A_vec_rand, q_rand = torch.rand(2, 10), torch.rand(2, 4)
    q_rand = q_rand / torch.norm(q_rand)
    print("=====")
    # Calculate loss
    # Non-batch case
    loss = lossfunc.forward(A_vec_rand[0], q_rand[0]).item()
    print("non-batch case: loss = {}".format(loss))
    # Batch case
    loss = lossfunc.forward(A_vec_rand, q_rand).item()
    print("batch case:     loss = {}".format(loss))

    # The following message will be shown if no error occurs
    print("*** If you see this message," " the installation seemed to be successful.")


class BinghamLossTestCase(BinghamTestBase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_installation(self):
        check_installation(BinghamLoss())
        check_installation(QCQPLoss())


if __name__ == "__main__":
    unittest.main()
