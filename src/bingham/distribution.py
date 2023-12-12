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

import bingham.math.normconst as nc
import bingham.math.sampler as bs
import numpy as np
import quaternion
import warnings


class BinghamDistribution:
    """Bingham distribution in SO(3)
    Manage Bingham distribution using symmetric matrix A

    Attributes:
        A (numpy.array): Symmetric matrix (4x4)
        Z (numpy.array): Eigenvalues of A - (minimum eigenvalue) (4)
        M (numpy.array): Eigenvectors of A (4x4)

    Examples:

        BinghamDistribution is described as a 4x4 matrix A = M diag(Z) M.T.
        It can be initilized using both A or Z, M.

        >>> bd = BinghamDistribution(A=a)

        or

        >>> bd = BinghamDistribution(Z=z, M=m)

        See also __init__ function.

        To multiply two BinghaDistribution,
        use `@` (mutmul) or `*` (mul) operator.

        >>> bd0 = BinghaDistribution(Z=z0, M=m0)
        >>> bd1 = BinghaDistribution(Z=z1, M=m1)
        >>> bd2 = bd0 @ bd1
        >>> bd3 = bd0 * bd1

        Note that bd2 == bd3 and (bd0 @ bd1) == (bd1 @ bd0)

        BinghamDistribution can be rotated by np.quaternion using `@` (mutmul).

        >>> bd = BinghaDistribution(Z=z, M=m)
        >>> q = np.quaternion(w, x, y, z)
        >>> bd_r = bd @ q
        >>> bd_l = q @ bd

        Note that bd_r != bd_l (unless q == (1, 0, 0, 0)).


    """

    def __init__(self, **kwargs):
        """Constructor

        Args:
            A (numpy.array): Symmetric matrix (4x4)
            Z (numpy.array): Eigenvalues of A - (minimum eigenvalue) (4)
            M (numpy.array): Eigenvectors of A (4x4)

        Examples:

            >>> bd = BinghamDistribution(A=a)

            When distribution is initialized using A,
            Z and M are calculated using eigh.

            >>> bd = BinghamDistribution(Z=z, M=m)

            When distribution is initialized using Z and M,
            A is calculated as M diag(Z) M.T.

        Note:
            A and (Z, M) are exclusive and
            cannot be specified simultaneously.
            (If specified at the same time, A takes precedence.)

            In both cases, eigenvalues Z are subtravted
            from the minimum eigenvalue,

            Z = {0, z[1] - z[0], z[2] - z[0], z[3] - z[0]}

            also A is recalculated as M diag(Z) M.T.
            So in case 1 A may be not equivalent to input a.

        """

        # Check inputs
        dim = 3
        keywords = kwargs.keys()
        if "A" in keywords:
            A = kwargs["A"]
            if not np.all(np.isclose(A, A.T)):
                warnings.warn(
                    "Input matrix A doesn't seem to be symmetric. "
                    "A will be replaced with 0.5*(A.T + A).",
                    UserWarning,
                )
                assert len(A.shape) == 2
                A = 0.5 * (A.T + A)
            # Set original value to *_raw
            self.A_raw = A
            self.Z_raw, self.M_raw = np.linalg.eigh(A)
        elif ("M" in keywords) and ("Z" in keywords):
            self.M_raw = kwargs["M"]
            # Check orthogonality
            if not np.all(np.isclose(self.M_raw.T @ self.M_raw, np.eye(dim + 1))):
                raise ValueError("Input matrix M must be orthogonal.")
            self.Z_raw = kwargs["Z"]
            self.A_raw = self.calc_A_from_MZ(self.M_raw, self.Z_raw)
        else:
            raise TypeError("You must specify a 4x4 sym matrix A or a pair (M, Z).")
        # Shift values
        self.M = self.M_raw[:, np.argsort(self.Z_raw)]
        self.Z = np.sort(self.Z_raw - self.Z_raw.max())
        self.A = self.calc_A_from_MZ(self.M, self.Z)
        # Define sampler
        self.sampler = bs.BinghamSampler(dim=dim)
        self.multiplyer = MultiplyBingham()
        self.rotator = RotateBingham()
        self.sample_buf = None

    @staticmethod
    def calc_A_from_MZ(M, Z):
        """M diag(Z) M.T -> A"""
        assert M.shape[0] == Z.shape[0]
        assert len(Z.shape) == 1
        return M @ np.diag(Z) @ M.T

    def density(self, quats, normalized=True):
        """Calc density for quaternions"""
        quats = self.check_normalized(quats)
        quats_array = quaternion.as_float_array(quats)
        N = nc.calc_constant(self.Z).item() if normalized else 1.0
        if len(quats_array.shape) == 1:
            inside_of_exp = np.einsum("i,ij,j->", quats_array, self.A, quats_array)
        else:
            inside_of_exp = np.einsum("bi,ij,bj->b", quats_array, self.A, quats_array)
        return np.exp(inside_of_exp) / N

    def update_sample(self, N_sample=None):
        """Update quaternion samples on Bingham distribution(A)

        Args:
            N_sample (int): (optional) N of samples, default=None

        Returns:
            numpy.array: quaternion (4) when N_sample is None,
            else quaternions (4 x N_sample)
        """
        if N_sample is None:
            self.sample_buf = quaternion.as_quat_array(self.sampler(self.A, 1)[0][0])
        else:
            self.sample_buf = quaternion.as_quat_array(self.sampler(self.A, N_sample)[0])
        return self.sample_buf

    def sample(self, N_sample=None):
        """Return quaternion samples on Bingham distribution(A),

        NOTE:
            this function uses cache buffer.
            Cache will be created on first call,
            also it will be updated
            when N_sample is different from previous call.
            If you need update cache with same N_buffer,
            please call update_sample.

        Args:
            N_sample (int): (optional) N of samples, default=None

        Returns:
            numpy.array: quaternion (4) when N_sample is None,
            else quaternions (4 x N_sample)

        """
        if self.sample_buf is None or self.sample_buf.shape[0] != N_sample:
            self.update_sample(N_sample=N_sample)
        return self.sample_buf

    def mode(self):
        """Mode of distribution"""
        return quaternion.as_quat_array(self.M[:, np.argmax(self.Z)])

    def conj(self):
        """Make conjugate distribution"""
        return BinghamDistribution(M=np.diag([1, -1, -1, -1]) @ self.M, Z=self.Z)

    @staticmethod
    def check_normalized(quats):
        """check all quaternions are normlized,
        if not, returns nomalized quaternions.
        """
        quats_array = quaternion.as_float_array(quats)
        quats_array = quats_array[None, :] if len(quats_array.shape) == 1 else quats_array

        quats_norms = np.linalg.norm(quats_array, axis=1)
        if not np.all(np.isclose(quats_norms, 1.0)):
            warnings.warn(
                "Given quat doesn't seem to be normalized." " quat will be normalized here.",
                UserWarning,
            )
        return quaternion.as_quat_array((quats_array.T / quats_norms).T)

    def __repr__(self):
        """repr"""
        repr_str = ""

        A = ["[{:>9.2e}, {:>9.2e}, {:>9.2e}, {:>9.2e}]".format(*a_row) for a_row in self.A]
        M = ["[{:>9.2e}, {:>9.2e}, {:>9.2e}, {:>9.2e}]".format(*m_row) for m_row in self.M]
        repr_str += "BinghamDistribution(\n"
        repr_str += "    A = [{},\n".format(A[0])
        repr_str += "         {},\n".format(A[1])
        repr_str += "         {},\n".format(A[2])
        repr_str += "         {}]\n".format(A[3])
        repr_str += "    M = [{},\n".format(M[0])
        repr_str += "         {},\n".format(M[1])
        repr_str += "         {},\n".format(M[2])
        repr_str += "         {}]\n".format(M[3])
        repr_str += "    Z = [{:>9.2e}, {:>9.2e}, {:>9.2e}, {:>9.2e}]\n".format(*self.Z)
        repr_str += " mode = [{:>9.2e}, {:>9.2e}, {:>9.2e}, {:>9.2e}])".format(
            *quaternion.as_float_array(self.mode())
        )
        return repr_str

    def __matmul__(self, other):
        """Left multiplication.
        If other's type is BinghamDistribution,
        then compute composition of them.
        If other's type is np.quaternion,
        then rotate the distribution.
        """
        if type(other) == BinghamDistribution:
            M_1, Z_1 = self.M, self.Z
            M_2, Z_2 = other.M, other.Z
            M_12, Z_12 = self.multiplyer(M_1, Z_1, M_2, Z_2)
            return BinghamDistribution(M=M_12, Z=Z_12)
        elif type(other) == np.quaternion:
            A_12 = self.rotator(self.A, right_quat=other)
            return BinghamDistribution(A=A_12)
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """Right multiplication.
        If other's type is np.quaternion, then rotate the distribution.
        """
        if type(other) == np.quaternion:
            A_12 = self.rotator(self.A, left_quat=other)
            return BinghamDistribution(A=A_12)
        else:
            raise ValueError("Cannot compute matmul with type '{}'".format(type(other)))

    def __mul__(self, other):
        """Product of pdf of Bingham distribution"""
        M_1, Z_1 = self.M, self.Z
        M_2, Z_2 = other.M, other.Z
        M_12, Z_12 = self.multiplyer(M_1, Z_1, M_2, Z_2)
        return BinghamDistribution(M=M_12, Z=Z_12)


class MultiplyBingham:
    """Multiplyer for Bingham distribution

    Examples:
        >>> multiplyer = MultiplyBingham()
        >>> nM, nZ = multiplyer(M_1, Z_1, M_2, Z_2)

    Args:
        M_1 (numpy.array): 4x4 eigenvectors matrix
        Z_1 (numpy.array): 4x1 eigenvalues
        M_2 (numpy.array): 4x4 eigenvectors matrix
        Z_2 (numpy.array): 4x1 eigenvalues

    Returns:
        (numpy.array, numpy.array):
            (4x4 eigenvectors matrix, 4x1 eigenvalues)

    """

    def __init__(self):
        pass

    def __call__(self, M_1, Z_1, M_2, Z_2):
        A_1 = M_1 @ np.diag(Z_1) @ M_1.T
        A_2 = M_2 @ np.diag(Z_2) @ M_2.T
        Z_12, M_12 = np.linalg.eigh(A_1 + A_2)
        return M_12, Z_12 - Z_12.max()


class RotateBingham:
    """Rotator for Bingham distribution

    Examples:
        >>> rotator = RotateBingham()
        >>> nA = rotator(A, left_quat=q)
        >>> nA = rotator(A, right_quat=q)
        >>> nA = rotator(A, left_quat=q, right_quat=q)

    NOTE:
        Left and right quaternions are commutable.

    Args:
        A (numpy.array): 4x4 Bingham distribution matrix
        left_quat (numpy.quaternion): quaternion from left side
        right_quat (numpy.quaternion): quaternion from right side

    Returns:
        numpy.array: 4x4 Bingham distribution matrix
    """

    def __init__(self):
        pass

    def __call__(self, A, left_quat=None, right_quat=None):
        assert (left_quat is not None) or (right_quat is not None)
        if left_quat is None:
            left_quat = np.quaternion(1, 0, 0, 0)
        if right_quat is None:
            right_quat = np.quaternion(1, 0, 0, 0)

        def Lmat(left_quat):
            """Create left rotation matrix from quaternion"""
            assert type(left_quat) == np.quaternion
            if not np.isclose(left_quat.norm(), 1.0):
                warnings.warn(
                    "left_quat doesn't seem to be normalized. " "This will be normalized here.",
                    UserWarning,
                )
                left_quat = left_quat.normalized()
            a = left_quat.w
            b = left_quat.x
            c = left_quat.y
            d = left_quat.z
            return np.array([[a, -b, -c, -d], [b, a, -d, c], [c, d, a, -b], [d, -c, b, a]])

        def Rmat(right_quat):
            """Create right rotation matrix from quaternion"""
            assert type(right_quat) == np.quaternion
            if not np.isclose(right_quat.norm(), 1.0):
                warnings.warn(
                    "right_quat doesn't seem to be normalized. " "This will be normalized here.",
                    UserWarning,
                )
                right_quat = right_quat.normalized()
            w = right_quat.w
            x = right_quat.x
            y = right_quat.y
            z = right_quat.z
            return np.array([[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]])

        # Calc K. Note that L_matrix and R_matrix are commutable.
        K = Lmat(left_quat) @ Rmat(right_quat)
        return K @ A @ K.T
