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
from bingham.math.normconst import calc_constant
from bingham.utils.naive_calc import normconst_naive

from .bingham_base import BinghamTestBase


class BinghamNormconstTestCase(BinghamTestBase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_c(self):
        norm_c = calc_constant(np.diag(self.Z))
        print(norm_c)
        naive_c = normconst_naive(np.diag(self.Z))
        print(naive_c)

        self.assertAlmostEqual(norm_c[0, 0], naive_c[0])


if __name__ == "__main__":
    unittest.main()
