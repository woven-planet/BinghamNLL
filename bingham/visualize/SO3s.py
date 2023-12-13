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
Visualization functions for SO(3) using matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
import quaternion


def figure_to_image(fig):
    """Convert matplotlib figure to raster image as numpy array

    Args:
        fig (figure): figure object plotted Bingham Distribution

    Returns:
        numpy.ndarray: Numpy array of a IMAGE_WIDTH x IMAGE_HEIGHT GRB image.
    """
    IMAGE_WIDTH, IMAGE_HEIGHT = 500, 500

    # Set image resolution.
    dpi = fig.get_dpi()
    fig.set_size_inches(IMAGE_WIDTH / float(dpi), IMAGE_HEIGHT / float(dpi))
    fig.tight_layout()

    # Save the figure to a numpy array.
    fig.canvas.draw()
    fig_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_np = fig_np.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    return fig_np


def draw_bingham_3d(bdistr, quat_gt, num_samples=500, probability=0.7, **kwargs_to_drawSO3):
    """Draw samples from a Bingham distribution
    and return an image object.

    Args:
        bdistr (BinghamDistribution): Object of BinghamDistribution
        quat_gt (numpy.array): Ground truth rotation
        num_samples (int): The number of samples to draw.
        probability (float): Probability of Bingham distribution

    Returns:
        numpy.ndarray: Numpy array of a IMAGE_WIDTH x IMAGE_HEIGHT GRB image.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    draw_bingham_distribution(ax, bdistr, quat_gt, num_samples, probability, **kwargs_to_drawSO3)

    fig_np = figure_to_image(fig)
    plt.close(fig)

    return fig_np


def draw_bingham_distribution(ax, bdistr, quat_gt, num_samples=500, probability=0.7, **kwargs_to_drawSO3):
    """Draw samples from a Bingham distribution

    Args:
        ax (axes): Object of axes
        bdistr (BinghamDistribution): Object of BinghamDistribution
        quat_gt (numpy.array): Ground truth rotation
        num_samples (int): The number of samples to draw.
        probability (float): Probability of Bingham distribution
    """
    M, zs = bdistr.M, bdistr.Z
    if (M is not None) and (zs is not None):
        quaternions = bdistr.sample(num_samples)
        rotations = np.zeros([num_samples, 3, 3])

        # Convert quaternion to rotation matrix.
        for idx, quat in enumerate(quaternions):
            rotation = quaternion.as_rotation_matrix(quat)
            rotations[idx, :, :] = rotation

        axes = quaternion.as_rotation_matrix(quaternion.as_quat_array(np.array(M[:, -1])))
    else:
        rotations = None
        axes = None

    if quat_gt is not None:
        axes_gt = quaternion.as_rotation_matrix(quaternion.as_quat_array(quat_gt))
    else:
        axes_gt = None
    draw_so3s(ax, rotations, axes, axes_gt, **kwargs_to_drawSO3)


def make_sphere():
    """
    Make a unit sphere pointcloud.

    Returns:
        tuple (numpy.array, numpy.array, numpy.array):
        a tuple of three arrays where
        each array contains x, y, z components of the sphere surface
        positions.
    """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def draw_so3s(ax, rotations, axes=None, axes_gt=None,
              distance=9, point_of_view=np.array([0., 0., -1.]), xaxis_direction='right', **kwargs):
    """Draw 3D points for the tips of x, y, z axes over the given rotations.
    Optionally, a coordinate system can be drew by providing @p axes.

    Args:
        ax (axes): Object of axes
        rotations (numpy.array):
            An np.ndarray whose shape is (N, 3, 3)
            where N denotes the number of
            rotation matrices, which are stored in the last two dimensions.
        axes (numpy.array):
            An orthonormal np.ndarray whose shape is (3, 3) and each column
            represents x, y, z axis of a 3D coordinate system respectively.
        axes_gt (numpy.array):
            Ground truth rotation, as same type as axes
        distance (float):
            A distance value from the origin (0,0,0) to the camera.
            This value is used in the 3D plot.
        point_of_view (numpy.array):
            A vector of the view point coordinates.
            The viewer camera is placed at the coordinates
            specified by `point_of_view` and looks at the origin.
        xaxis_direction (str):
            An x-axis direction in the screen
            (chosen from ['left', 'right', 'up', 'down']).
            The x-axis may not be uniquely directed depending on
            the `point_of_view`, in which case the x-axis is oriented
            in the direction specified by `xaxis_direction`.
    """
    ax.grid(False)
    ax.set_box_aspect((1, 1, 1))

    front_alpha = 0.5
    back_alpha = 0.5

    # red, green, blue
    front_colors = ["#ff0000", "#008000", "#0000ff"]
    back_colors = ["#ff8080", "#80c080", "#8080ff"]

    if rotations is not None:
        xs = rotations[:, :, 0]
        ys = rotations[:, :, 1]
        zs = rotations[:, :, 2]

    def calc_viewangle(point_of_view, xaxis_direction):
        """
        xaxis_direction is used if azim_rad becomes undefined (both p[0] and p[1] are zero.)
        """
        if not xaxis_direction in ['left', 'right', 'up', 'down']:
            raise ValueError("Invalid xaxis_direction \"{}\" is given.".format(xaxis_direction))
        
        p = point_of_view / np.linalg.norm(point_of_view)
        
        if np.isclose(p[2], +1.0):
            azim_rad = {'down': 0., 'left': 90., 'up': 180., 'right': -90.}[xaxis_direction] * np.pi / 180
        elif np.isclose(p[2], -1.0):
            azim_rad = {'up': 0., 'left': 90., 'down': 180., 'right': -90.}[xaxis_direction] * np.pi / 180
        else:
            azim_rad = np.arctan2(p[1], p[0])
        elev_rad = np.arcsin(p[2])

        return {'azim': azim_rad * 180 / np.pi, 'elev': elev_rad * 180 / np.pi}

    def plot_scatter(data, color, label, alpha=0.3):
        ax.scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            color=color,
            alpha=alpha,
            marker=".",
            label=label,
        )

    def arrow3d(ax, R, along="z", length=1.0, width=0.03, head=0.33, headwidth=4, **kw):
        # TODO: need to review OR rewrite
        arrow_pts = [
            [0, 0],
            [width, 0],
            [width, (1 - head) * length],
            [headwidth * width, (1 - head) * length],
            [0, length],
        ]
        arrow_pts = np.array(arrow_pts)

        r, theta = np.meshgrid(arrow_pts[:, 0], np.linspace(0, 2 * np.pi, 30))
        z = np.tile(arrow_pts[:, 1], r.shape[0]).reshape(r.shape)
        x = r * np.sin(theta)
        y = r * np.cos(theta)

        if along == "x":
            R_swap = np.eye(3)[[2, 1, 0]]
        if along == "y":
            R_swap = np.eye(3)[[0, 2, 1]]
        if along == "z":
            R_swap = np.eye(3)[[0, 1, 2]]

        b1 = np.dot(R_swap, np.c_[x.flatten(), y.flatten(), z.flatten()].T)
        b2 = np.dot(R, b1).T
        x = b2[:, 0].reshape(r.shape)
        y = b2[:, 1].reshape(r.shape)
        z = b2[:, 2].reshape(r.shape)
        ax.plot_surface(x, y, z, **kw)

    if rotations is not None:
        xs_isback = (xs * point_of_view).sum(axis=1) < 0
        ys_isback = (ys * point_of_view).sum(axis=1) < 0
        zs_isback = (zs * point_of_view).sum(axis=1) < 0

        plot_scatter(xs[xs_isback], back_colors[0], label=None, alpha=back_alpha)
        plot_scatter(ys[ys_isback], back_colors[1], label=None, alpha=back_alpha)
        plot_scatter(zs[zs_isback], back_colors[2], label=None, alpha=back_alpha)

    x, y, z = make_sphere()
    ax.plot_wireframe(x, y, z, rstride=5, cstride=5, color="gray", linewidth=0.2)

    def draw_axes(axes, **kw):
        """
        The axes should be drawn from the back to the front,
        i.e., in the order of how far the tip of the axis is
        from the `point_of_view` \in R^3.
        """
        dist_from_axistip_to_viewpoint = np.linalg.norm(axes - point_of_view.reshape(-1, 1), axis=0)

        # The farthest should comes first, the nearest last.
        draw_order = np.argsort(-dist_from_axistip_to_viewpoint)
        for i in draw_order:
            arrow3d(ax, axes, along=["x", "y", "z"][i], color=["red", "green", "blue"][i], **kw)

    if axes is not None:
        draw_axes(axes)

    if axes_gt is not None:
        draw_axes(axes_gt, alpha=0.2)

    if rotations is not None:
        plot_scatter(
            xs[np.logical_not(xs_isback)], front_colors[0], label="x-axis", alpha=front_alpha
        )
        plot_scatter(
            ys[np.logical_not(ys_isback)], front_colors[1], label="y-axis", alpha=front_alpha
        )
        plot_scatter(
            zs[np.logical_not(zs_isback)], front_colors[2], label="z-axis", alpha=front_alpha
        )

    # calc azim and elev from point_of_view
    viewangles = calc_viewangle(point_of_view, xaxis_direction)
    ax.dist = distance
    ax.azim = viewangles['azim']
    ax.elev = viewangles['elev']

    ax.legend(loc="upper left")
    for set_ticks in (ax.set_xticks, ax.set_yticks, ax.set_zticks):
        set_ticks([-1.0, 0.0, 1.0])
