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
Sampling from Bingham distribution on S^p
This code is based on John et al. (2013, https://arxiv.org/abs/1310.8110 )
"""

import math
import numpy as np


class BinghamSampler:
    """Sample points on Bingham distribution

    Examples:
        >>> sampler = BinghamSampler()
        >>> points = sampler(As, sampling_N)

    Args:
        As (numpy.array): array of Bingham distribution (size: B x q x q)
        sampling_N (int): number of sampling points
    Returns:
        numpy.array: sampled points (size: B x sampling_N)
    """

    def __init__(self, dim=3):
        """Constructor

        Args:
            dim(int): dimension, for rotation quaternion dim = 3 (default).

        """
        self.p = dim
        self.q = dim + 1

    def __call__(self, As, sampling_N):
        if len(As.shape) == 2:
            As = As[None, :, :]

        results = [self.sampling_from_bingham(A, sampling_N=sampling_N) for A in As]
        return np.array(results)

    def sampling_from_bingham(self, A, sampling_N=10000):
        """Quaternion sampling on Bingham distribution.
        Based on rejection sampling method.

        Args:
            A (numpy.array): Bingham distribution A
            sampling_N (int): (optional) N of sampling, default=10000

        Returns:
            numpy.array: quaternions (4 x sampling_N)

        """
        # To make consistent with paper and our notation
        # (Our definition of Bingham distr. is p(x) \propto exp(x^T A x),
        # while the sampling code's definition is p(x) \propto exp(-x^T A x).)
        A = -A

        # Calc Lambda and b from A
        Lambda, _ = np.linalg.eigh(A)
        if not np.isclose(np.min(Lambda), 0):
            A = A - np.min(Lambda) * np.eye(A.shape[0])
            Lambda = Lambda - np.min(Lambda)
        b = self.calc_optim_b(Lambda)

        # Calc Omega from A and b
        # Omega is defined in the mid part of p.5
        Omega = np.eye(A.shape[0]) + 2 * A / b
        invsqrt_Omega = np.linalg.cholesky(np.linalg.inv(Omega))
        # = cACG in Eq. 3.3

        q = self.q
        # Former part of left side of Eq. 3.4
        M_star = np.exp(-(q - b) / 2.0) * (q / b) ** (q / 2)

        result = None
        sample_from_proposed = 2 * sampling_N
        # The acception rate is about 0.5,
        # so we sample 2 * sampling_N points from the proposed distribution.
        while True:
            # sampling from ACG (proposed distribution)
            X = self.sampling_from_ACG(invsqrt_Omega, sampling_N)
            ACG_X_unnorm = self.pdf_ACG_unnormalized(X, Omega, q)
            Bingham_X_unnorm = self.pdf_Bingham_unnormalized(X, A)
            # Eq. 3.4: Bingham_X_unnorm <= M_star * ACG_X_unnorm

            # rejection part
            # Section 2, Step 1
            uniforms = np.random.rand(sampling_N)
            # Section 2, Step 2
            accepted = X[uniforms < Bingham_X_unnorm / (M_star * ACG_X_unnorm)]

            result = accepted if result is None else np.append(result, accepted, axis=0)
            sample_from_proposed = 2 * (sampling_N - result.shape[0])
            if not sample_from_proposed > 0:
                break
        return result[:sampling_N]

    def estim_alpha(self, A, prob, sampling_N=100000):
        """
        find `alpha` (0 < `alpha` < 1) that satisfies

        -p(x) > `alpha` <==>
        \\int_{x\\in p^{-1}([`alpha`,1])} p(x) dx = `prob`

        In other words, if sampled x satisfies p(x) > `alpha`,
        x occurs with a probability of `prob`.

        """
        sampled_points = self.sampling_from_bingham(A, sampling_N=sampling_N)

        # find alpha by binary search
        alpha_bounds = np.array([0.0, 1.0])
        prev_prob_estim = -1
        while True:
            alpha = np.mean(alpha_bounds)
            prob_densities = self.pdf_Bingham_unnormalized(sampled_points, A)

            prob_estim = np.sum(prob_densities > alpha) / sampling_N

            if prev_prob_estim == prob_estim:
                break

            prev_prob_estim = prob_estim
            if prob_estim > prob:
                alpha_bounds[0] = alpha
            else:
                alpha_bounds[1] = alpha
        return alpha, sampled_points[prob_densities > alpha]

    def is_above_alpha(self, A, quat, alpha):
        """Check probabiltiy of quaternion > alpha"""
        prob_densities = self.pdf_Bingham_unnormalized(quat, A)
        return prob_densities > alpha

    @staticmethod
    def calc_optim_b(Lambda):
        """
        John et al. suggested that b is optimal if it satisfies
        - \\sum_{i=1}^q \\frac{1}{b + 2*\\lambda_i} = 1
        where \\lambda_i is a eigenvalue of A
        """

        def func(b):
            return (np.sum(1 / (b + 2 * Lambda)) - 1) ** 2

        def dfunc_db(b):
            return -2 * np.sum(1 / (b + 2 * Lambda) ** 2) * (np.sum(1 / (b + 2 * Lambda)) - 1)

        # optimize using Newton method
        b = 1.0
        for _ in range(16):
            bstep = func(b) / dfunc_db(b)
            if math.fabs(bstep) < 1e-6:
                break
            b = b - bstep
        return b

    @staticmethod
    def pdf_ACG_unnormalized(x, Omega, q):
        """f*ACG, defined in Eq. 3.3"""
        if len(x.shape) == 2:
            return np.einsum("bm,mn,bn->b", x, Omega, x) ** (-q / 2)
        else:
            return np.einsum("m,mn,n->", x, Omega, x) ** (-q / 2)

    @staticmethod
    def pdf_Bingham_unnormalized(x, A):
        """f*Bing(x), defined in Eq. 3.1"""
        if len(x.shape) == 2:
            return np.exp(-np.einsum("bm,mn,bn->b", x, A, x))
        else:
            return np.exp(-np.einsum("m,mn,n->", x, A, x))

    @staticmethod
    def sampling_from_ACG(invsqrt_Omega, N):
        """angular central Gaussian sampling
        described in the middle part of p.5
        """

        # if invsqrt_Omega is None:
        #     invsqrt_Omega = np.linalg.cholesky(np.linalg.inv(Omega))
        q = invsqrt_Omega.shape[0]
        y = invsqrt_Omega.dot(np.random.randn(q, N))
        return (y / np.linalg.norm(y, axis=0)).T

    def genA(self, variance=3, seed=None):
        """generate a random symmetric matrix

        Args:
            variance (float): variance of Bingham distribution
            seed (int): random seed, default = None

        Returns:
            numpy.array: symmetric matrix A

        """
        next_seed = np.random.randint(1e7)
        if seed is not None:
            np.random.seed(seed=seed)
        A = 1 - 2 * np.random.randn(self.q, self.q) * variance
        np.random.seed(next_seed)
        A = A.T @ A
        return A
