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
The normalizing constant of Bingham distribution
using naive calculation.

The implementation is theoretically based on
Chen--Tanaka (2020; https://arxiv.org/abs/2004.14660.pdf).
"""

import numpy as np
from scipy import integrate


def spherical_integral(func_over_S3):
    """Common integral on spherical surface

    Args:
        func_over_S3 (function): Ihtegrand function

    Returns:
        complex: Integral value

    """

    def f_polar(theta, phi, eta):
        # Polar to S3 euclid coordinate
        s1, c1 = np.sin(theta), np.cos(theta)
        s2, c2 = np.sin(phi), np.cos(phi)
        s3, c3 = np.sin(eta), np.cos(eta)
        J = s2 * (s1**2)  # volume element dV
        q_polar = np.array([s1 * s2 * s3, s1 * s2 * c3, s1 * c2, c1])
        return func_over_S3(q_polar) * J

    return integrate.tplquad(
        lambda p, q, r: f_polar(p, q, r),
        0,
        np.pi,
        lambda x: 0,
        lambda x: np.pi,
        lambda x, y: 0,
        lambda x, y: 2 * np.pi,
    )


def normconst_naive(Z):
    """Normalizing constant C using naive calcuration,
    defined in section 1.1 of [Chen 2020].

    C = integral(exp(sigma(Zx^2))) for all unit 3-spherical surface

    NOTE:
        sign of Z is inverted from theta in [Chen 2020],
        see also math/normconst.py

    Args:
        Z (numpy.array): eigenvalues

    Returns:
        complex: Normalize constant C

    """

    def integrand(x):
        return np.exp(np.einsum("i,i->", Z, x**2))

    return spherical_integral(integrand)
