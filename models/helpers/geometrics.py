import torch


def homogenize(m):
    """Adds homogeneous coordinates to a [..., N,N] matrix, returning [..., N+1, N+1]."""
    assert m.shape[-1] == m.shape[-2]  # Must be square
    n = m.shape[-1]
    eye_n_plus_1 = torch.eye(n + 1).cuda().expand(list(m.shape[:-2]) + [-1, -1])
    extra_col = eye_n_plus_1[..., :-1, -1:]
    extra_row = eye_n_plus_1[..., -1:, :]
    including_col = torch.cat([m, extra_col], dim=-1)
    return torch.cat([including_col, extra_row], dim=-2)


def compute_world2local_dist(dists, radii, rotations):
    """Computes a transformation to the local element frames for encoding."""
    # We assume the center is an XYZ position for this transformation:
    # TODO(kgenova) Update this transformation to account for rotation.
    # assert len(dists.shape) == 3
    # batch_size, element_count = dists.shape[:2]

    # eye_3x3 = torch.eye(3).cuda().expand([batch_size, element_count, -1, -1])
    # eye_4x4 = torch.eye(4).cuda().expand([batch_size, element_count, -1, -1])

    # Centering transform
    # ones = torch.ones([batch_size, element_count, 1, 1])
    dists = dists[..., None]
    # tx = torch.cat([eye_3x3, -dists], dim=-1)
    # tx = torch.cat([tx, eye_4x4[..., 3:4, :]], dim=-2)  # Append last row

    # Compute the inverse rotation:
    rotation = roll_pitch_yaw_to_rotation_matrices(rotations) # torch.inverse(roll_pitch_yaw_to_rotation_matrices(rotations))
    # print("rotation", rotation[0,0])
    assert rotation.shape[-2:] == (3, 3)

    # Compute a scale transformation:
    diag = 1.0 / (radii + 1e-8)
    scale = torch.diag_embed(diag)

    # Apply both transformations and return the transformed points.
    tx3x3 = torch.matmul(scale, rotation)
    return torch.matmul(tx3x3, dists) #, torch.matmul(homogenize(tx3x3), tx)


def roll_pitch_yaw_to_rotation_matrices(roll_pitch_yaw):
    """Converts roll-pitch-yaw angles to rotation matrices.
    Args:
      roll_pitch_yaw: Tensor with shape [..., 3]. The last dimension contains
        the roll, pitch, and yaw angles in radians.  The resulting matrix
        rotates points by first applying roll around the x-axis, then pitch
        around the y-axis, then yaw around the z-axis.
    Returns:
      Tensor with shape [..., 3, 3]. The 3x3 rotation matrices corresponding to
      the input roll-pitch-yaw angles.
    """

    cosines = torch.cos(roll_pitch_yaw)
    sines = torch.sin(roll_pitch_yaw)
    cx, cy, cz = torch.unbind(cosines, dim=-1)
    sx, sy, sz = torch.unbind(sines, dim=-1)
    # pyformat: disable
    rotation = torch.stack(
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
        sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
        -sy, cy * sx, cy * cx], dim=-1)
    # pyformat: enable
    #shape = torch.cat([roll_pitch_yaw.shape[:-1], [3, 3]], axis=0)
    shape = list(roll_pitch_yaw.shape[:-1]) + [3, 3]
    rotation = torch.reshape(rotation, shape)
    return rotation


def get_vector_rotation_matrices(n_ori, n_new):
    """
    Get rotation matrices to rotate vector n_ori to n_new
    Note: the accuracy of the current impl is not high enough, but efficient
    :param n_ori: [B, 3], normalized
    :param n_new: [B, 3], normalized
    :return: [B, 3, 3]
    """
    v = torch.cross(n_ori, n_new) # B x 3
    s = torch.norm(v, dim=-1, keepdim=True) # B x 1
    c = torch.sum(n_ori * n_new, dim=-1, keepdim=True) # B x 1
    vx = torch.stack([torch.zeros_like(s), -v[:, 2:3], v[:, 1:2]], dim=2)
    vy = torch.stack([v[:, 2:3], torch.zeros_like(s), -v[:, 0:1]], dim=2)
    vz = torch.stack([-v[:, 1:2], v[:, 0:1], torch.zeros_like(s)], dim=2)
    v = torch.cat([vx, vy, vz], dim=1)
    s = s[..., None]
    c = c[..., None]
    v = v * (c < 0.8) # for numerical stability 
    eye = torch.eye(3, device=v.device, dtype=v.dtype).unsqueeze(0).expand(list(n_ori.shape[:-1]) + [-1, -1])
    R = eye + v * s + torch.matmul(v, v) * (1 - c) / (s ** 2)
    return R