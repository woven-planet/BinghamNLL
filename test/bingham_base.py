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
from bingham.math.quaternion import quat_mat


# https://gist.github.com/twolfson/13f5f5784f67fd49b245
class BinghamTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (cls is not BinghamTestBase) and (cls.setUp is not BinghamTestBase.setUp):
            orig_setUp = cls.setUp

            def setUpOverride(self, *args, **kwargs):
                BinghamTestBase.setUp(self)
                return orig_setUp(self, *args, **kwargs)

            cls.setUp = setUpOverride

    def setUp(self):
        self.V = quat_mat(np.random.randn(4))
        self.Z = np.cumsum(np.random.rand(4)) * 500
        self.Z = np.diag(self.Z - self.Z.max())
        self.A_truth = self.V @ self.Z @ self.V.T
