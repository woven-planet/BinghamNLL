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
from bingham.math.sampler import BinghamSampler

from .bingham_base import BinghamTestBase


class BinghamSamplerTestCase(BinghamTestBase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_optim_b(self):
        bs = BinghamSampler()

        # From BinghamSampler.samplig_from_bingham()
        Lambda, _ = np.linalg.eigh(self.A_truth)
        if not np.isclose(np.min(Lambda), 0):
            Lambda = Lambda - np.min(Lambda)
        b = bs.calc_optim_b(Lambda)

        # Eq. (3.6)
        res = np.sum([1.0 / (b + 2.0 * eigval) for eigval in Lambda])
        self.assertAlmostEqual(res, 1.0, delta=1e-5)

    def test_sampling(self):
        bs = BinghamSampler()
        q_targs_array = bs(self.A_truth, 20)[0]
        np.testing.assert_equal(q_targs_array.shape, (20, 4))


if __name__ == "__main__":
    unittest.main()
