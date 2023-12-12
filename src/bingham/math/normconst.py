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
This is a code for calculating
the normalizing constant of Bingham distribution.
The implementation is theoretically based on
Chen--Tanaka (2020; https://arxiv.org/abs/2004.14660.pdf).
"""

import numpy as np
from scipy import special


def preprocess_vars(Lambda, gamma=None):
    """Preprocess Lambda and gamma,
    add first dim if batchsize is zero,
    like torch.unspueeze.

    Args:
        Lambda (numpy.array): Eigen values
        gamma (numpy.array): === 0 in this case

    Returns:
        (numpy.array, numpy.array):
            Unsqueezed Lambda and gamma,
            gamma = 0 if gamma is None.

    """
    # Lambda = all elements are non positive
    # if B=0, then B=1
    if gamma is None:
        gamma = Lambda * 0.0
    Lambda = Lambda[None, :] if len(Lambda.shape) == 1 else Lambda
    gamma = gamma[None, :] if len(gamma.shape) == 1 else gamma
    return Lambda, gamma


def funcF(t, Lambda, c, gamma=None):
    """A(t, theta, gamma), defined in Theorem 1 (p4)

    NOTE:
        We modified sign of arguments and replace theta -> Lambda, t0 -> c,
        so the function A is redefined as follows:

        A = prod(
        exp((gamma_i ^ 2) / 4 (-Lambda_i + t j + c))
        /
        sqrt(-Lambda_i + t j + c))

    Args:
        t (float): Parameter
        Lambda (numpy.array): Eigen values
        c (float): = -t0 in definition
        gamma (numpy.array): === 0 in this case

    Returns:
        numpy.array: A

    """
    Lambda, gamma = preprocess_vars(Lambda, gamma)
    gamma_part = np.exp(gamma[:, :, None] ** 2 * (4 * (-Lambda[:, :, None] + 1j * t + c)) ** -1)
    return np.prod(gamma_part * (-Lambda[:, :, None] + 1j * t + c) ** -0.5, axis=1, keepdims=True)


def dfuncF_dLambda(t, Lambda, c, gamma=None):
    """d/d theta A(t, theta, gamma)

    NOTE:
        This is defined in section 4 but it includes some errata.
        We verified d/d Lamda A (see also funcF) as follows:

        d/d Lambda A = A * (
        (gamma_i ^ 2 / 4) * (-Lambda_i + t j + c) ^ -2 +
        1 / 2(-Lambda_i + t j + c))

        In this case, always gamma_i = 0
        then the function is simply described as

        d/d Lambda A = A * (0.5 / (-Lambda_i + t j + c))

    Args:
        t (float): parameter
        Lambda (numpy.array): eigenvalues
        c (float): = -t0 in definition
        gamma (numpy.array): always None in this case

    Returns:
        numpy.array: d/d Lamda A

    """
    if gamma is not None:
        # TODO: implement this
        raise NotImplementedError
    Lambda, gamma = preprocess_vars(Lambda, gamma)
    return (0.5 / (-Lambda[:, :, None] + 1j * t + c)) * funcF(t, Lambda, c)


def integral_common(integrand, Lambda, gamma=None, N=200):
    """Integration part of Fisher-Bingham distribution

    Args:
        integrand(function): reference to the integrand function
        Lambda (numpy.array): eigenvalues
        gamma (numpy.array): mean * deviation, always None in this case
        N (int): number of divisions

    Returns:
        numpy.array: Integral values

    """
    # Any N larger than Nmin is acceptable
    # set hyperparameters described in Section 3.1 (p.4)
    # Cond. 1: w_d <= 1 <= w_u
    # Cond. 2: w_d/w_u <= 1/2
    # NOTE: we defined r = w_u / w_d for simplicity,
    # then conditions are rewritten as follows:
    # Cond. 1: 1/r <= w_d <= 1
    # Cond. 2: r >= 2
    w_d = 0.5
    r = 2.5

    # In section 3.1 lower bound of N is defined as
    # Nmin = 2d * r^2 * (1+r) * w_d/pi
    # For practical use, it is more convinient to choose Nmin firstly,
    # then the equation is rewriten to solve d:
    # d = (Nmin * pi)/(2 * r^2 * (1+r) * w_d)
    # Also d shall satisfy following condition:
    # d <= min(|Lambda_i - t0|)
    # In this case, min(Lambda_i) = 0 so the condition is simply rewritten
    # d <= |t0|
    # We define c = -t0, and choose d = c/2.0 as we will describe later.
    # ==> c = (Nmin * pi)/(r^2 * (1+r) * w_d)
    # if Nmin is too small, c is so close to zero (= max Lambda)
    #  that the computation may become unstable.
    Nmin = 15
    c = (Nmin * np.pi) / (r**2 * (1 + r) * w_d)

    # Shift Lambda for stable computation
    Lambda, gamma = preprocess_vars(Lambda, gamma)
    Lambda_max = Lambda.max(axis=1, keepdims=True)
    Lambda_shifted = Lambda - Lambda_max
    shift_constant = np.exp(Lambda_max)

    # For stable computation, we assume any pole of funcF
    # (as well as dfuncF_dLambda)
    # is not in the range |Im z| < d.
    # (In Tanaka 2014, they assume that
    #  "f is square integrable over R" in assumption 2.)
    # Note that poles of both functions can be written as
    # t == j*(Lambda_i + c).
    # (solved "Lambda_i + jt - t0 == 0" for t, and used t0 == -c)
    # In our situation,
    # since Lambda_i >= 0 for all i, all poles are in |Im z| > c.
    # One can choose any positive number smaller than c for d.
    # We choose d = c/2 here.
    d = c / 2.0

    # From Chen--Tanaka (2021) p.4, Tanaka (2014) Eq. 3.15.
    # h: step size of integration
    h = np.sqrt(2 * np.pi * d * (1 + r) / (w_d * N))
    # p1, p2: constant for weight function (weight function is defined below)
    p1 = np.sqrt(N * h / w_d)
    p2 = np.sqrt(w_d * N * h / 4.0)

    # Weight function for the continuous Euler transformation (Ooura 2001).
    def w(x, p1=p1, p2=p2):
        return 0.5 * special.erfc(x / p1 - p2)

    # Integration Part
    # set integrand = funcF to calc constant
    # set integrand = dfuncF_dLambda to calc dconstant/dLambda
    ns = np.arange(-N - 1, N + 1)

    # Summention part of Eq. 8
    sumelems = (
        w(np.abs(ns * h), p1, p2) * integrand(ns * h, Lambda_shifted, c=c) * np.exp(1j * ns * h)
    )
    sumres = np.sum(sumelems, axis=2) * shift_constant

    # Complete result of Eq. 8
    # np.pi**(dim/2.0 - 1) == np.pi since dim == 4 here
    complex_res = np.pi * np.exp(c) * h * sumres

    # Im(complex_res) is very close to zero, but not strictly.
    # We cut off the imaginary part of complex_res.
    return np.real(complex_res)


def calc_constant(Lambda, N=200):
    """Normalizing constant C

    Args:
        Lambda (np.array):
            Array of 4 non-positive numbers with shape Bx4
            where B is batch size (corresponds to eigenvalues)
        N (int): Number of division

    Returns:
        numpy.array:
            Normalizing constant of the distribution
            parametrized by given Lambda

    """
    return integral_common(funcF, Lambda, N=N)


def calc_Dconstant(Lambda, N=200):
    """Derivative of normalizing constant C

    Args:
        Lambda (np.array):
            Array of 4 non-positive numbers with shape Bx4
            where B is batch size (corresponds to eigenvalues)
        N (int): Number of division

    Returns:
        numpy.array:
            Derivative of normalizing constant at given Lambda

    """
    return integral_common(dfuncF_dLambda, Lambda, N=N)
