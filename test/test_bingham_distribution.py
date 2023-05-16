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
import unittest
from bingham.distribution import BinghamDistribution, MultiplyBingham, RotateBingham
from bingham.math.quaternion import quat_mat

from .bingham_base import BinghamTestBase


class BinghamDistributionTestCase(BinghamTestBase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_a(self):
        bd = BinghamDistribution(A=self.A_truth)
        np.testing.assert_allclose(self.A_truth, bd.A)
        np.testing.assert_allclose(self.Z.diagonal(), bd.Z)

    def test_m_z(self):
        bd = BinghamDistribution(M=self.V, Z=self.Z.diagonal())
        np.testing.assert_allclose(self.A_truth, bd.A)

    def test_bd_multiply(self):
        V2 = quat_mat(np.random.randn(4))
        Z2 = np.cumsum(np.random.rand(4)) * 500
        Z2 = np.diag(Z2 - Z2.max())

        q = np.quaternion(0.0, 1.0, 0.0, 0.0)

        bd1 = BinghamDistribution(M=self.V, Z=self.Z.diagonal())
        bd2 = BinghamDistribution(M=V2, Z=Z2.diagonal())

        res0 = bd1 @ bd2
        res1 = bd1 * bd2
        res2 = bd2 @ bd1
        res3 = bd2 * bd1
        np.testing.assert_allclose(res0.A, res1.A)
        np.testing.assert_allclose(res0.A, res2.A)
        np.testing.assert_allclose(res1.A, res3.A)

        bd_r = bd1 @ q
        bd_r = bd_r @ q
        bd_l = q @ bd1
        bd_l = q @ bd_l
        np.testing.assert_allclose(bd_r.A, bd_l.A)

    def test_multiplyer(self):
        multiplyer = MultiplyBingham()
        vv, zz = multiplyer(self.V, self.Z.diagonal(), self.V, self.Z.diagonal())
        np.testing.assert_equal(vv.shape, self.V.shape)
        np.testing.assert_equal(zz.shape, self.Z.diagonal().shape)

    def test_rotate(self):
        q = np.quaternion(0.0, 1.0, 0.0, 0.0)
        rotator = RotateBingham()
        nA = rotator(self.A_truth, left_quat=q)
        A1 = rotator(nA, right_quat=q)
        A2 = rotator(self.A_truth, left_quat=q, right_quat=q)
        np.testing.assert_almost_equal(A1, A2)


if __name__ == "__main__":
    unittest.main()
