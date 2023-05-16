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

import numpy as np
import torch
import unittest
from bingham.utils.reshape import (
    convert_4x4mat_to_10array,
    convert_10array_to_4x4mat,
    from10D_to4x4_numpy,
    from10D_to4x4_torch,
)

from .bingham_base import BinghamTestBase


class BinghamUtilsTestCase(BinghamTestBase):
    def setUp(self):
        self.init_A = np.random.rand(10) * 10

    def tearDown(self):
        pass

    def test_reshape_numpy(self):
        a10d = convert_4x4mat_to_10array(self.A_truth)
        A_estim = convert_10array_to_4x4mat(a10d)
        np.testing.assert_allclose(self.A_truth, A_estim[0])

    def test_reshape_torch(self):
        A0 = torch.from_numpy(self.init_A)
        A0_np = A0.detach().numpy()

        A_estim, evls = from10D_to4x4_numpy(A0_np, reduced=False, evls=False, shift=False)
        A_pred, Lambdas = from10D_to4x4_torch(A0, reduced=False, evls=False, shift=False)
        self.assertEqual(evls, None)
        self.assertEqual(Lambdas, None)
        np.testing.assert_allclose(A_estim, A_pred.detach().numpy())

        A_estim, evls = from10D_to4x4_numpy(A0_np, reduced=False, evls=True, shift=False)
        A_pred, Lambdas = from10D_to4x4_torch(A0, reduced=False, evls=True, shift=False)
        np.testing.assert_allclose(evls, Lambdas.detach().numpy())

        A_estim, evls = from10D_to4x4_numpy(A0_np, reduced=False, evls=True, shift=True)
        A_pred, Lambdas = from10D_to4x4_torch(A0, reduced=False, evls=True, shift=True)
        np.testing.assert_allclose(A_estim, A_pred.detach().numpy())

        A_estim, evls = from10D_to4x4_numpy(A0_np, reduced=True, evls=True, shift=True)
        A_pred, Lambdas = from10D_to4x4_torch(A0, reduced=True, evls=True, shift=True)
        np.testing.assert_allclose(A_estim, A_pred.detach().numpy())


if __name__ == "__main__":
    unittest.main()
