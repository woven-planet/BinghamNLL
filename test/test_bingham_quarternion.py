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


import math
import numpy as np
import unittest
from bingham.math.quaternion import average_quat_Frob, diff_quats, quat_mat
from bingham.math.sampler import BinghamSampler

from .bingham_base import BinghamTestBase


class BinghamQuaternionTestCase(BinghamTestBase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_quat_mat(self):
        bs = BinghamSampler()
        qs = bs(self.A_truth, 20)[0]
        for q in qs:
            mat = quat_mat(q)
            np.testing.assert_equal(mat.shape, (4, 4))
            for i in range(4):
                self.assertAlmostEqual(q[0], mat[i, i])

    def test_diff(self):
        bs = BinghamSampler()
        qs = bs(self.A_truth, 20)[0]
        q0 = qs[0]
        diff = diff_quats(q0, q0)
        self.assertAlmostEqual(diff, 0.0)
        for q in qs[1:]:
            diff = diff_quats(q0, q)
            self.assertTrue(diff >= 0.0)
            self.assertTrue(diff <= 2.0 * math.pi)

    def test_average(self):
        bs = BinghamSampler()
        qs = bs(self.A_truth, 20)[0]
        quat = average_quat_Frob(qs)
        np.testing.assert_equal(quat.shape, (4,))


if __name__ == "__main__":
    unittest.main()
