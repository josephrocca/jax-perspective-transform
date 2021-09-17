# Mostly a copy-paste from: https://github.com/kornia/kornia/blob/6d839c8290b67dabfad9c1ffdc7e428c6030a499/kornia/geometry/transform/imgwarp.py#L34
# Probably buggy. 90% of the code here is either kinda useless (error checking where the parent function already checked) or docstrings which aren't
# correct since they're based on the Pytorch Kornia code ¯\_(ツ)_/¯

import jax
import jax.numpy as np
import jax.scipy as scipy


def warp_perspective(
    src: np.ndarray,
    M: np.ndarray,
    dsize,
    interp_mode = 1,
    padding_mode = 'constant',
    cval = (0, 0, 0),
) -> np.ndarray:
    r"""Apply a perspective transformation to an image.
    If using `jax.numpy`, only interp_mode=0 (nearest neighbor) and interp_mode=1 (linear interpolation) and padding_mode='constant' or 'nearest' or 'wrap' are currently supported (as of writing).
    See these docs for latest feature support: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.ndimage.map_coordinates.html
    Args:
        src: input image with shape `(C, H, W)`.
        M: transformation matrix with shape `(3, 3)`.
        dsize: size of the output image (height, width).
        interp_mode: nearest neighbor (interp_mode=0), linear interpolation (interp_mode=1)
        padding_mode: padding mode for outside grid values 'constant', 'nearest' or 'wrap'. If 'constant' then specify 'cval' as an rgb tuple ot use as padding.
    Returns:
        the warped input image `(C, H, W)`.
    """
    if not isinstance(src, np.ndarray):
        raise TypeError(f"Input src type is not a np.ndarray. Got {type(src)}")
        
    if src.dtype == np.uint8:
        raise TypeError(f"Input should not be uint8, use `src.astype('float32')`")

    if not isinstance(M, np.ndarray):
        raise TypeError(f"Input M type is not a np.ndarray. Got {type(M)}")

    if not len(src.shape) == 3:
        raise ValueError(f"Input src must be a CxHxW tensor. Got {src.shape}")
        
    if not (len(M.shape) == 2 and M.shape[-2:] == (3, 3)):
        raise ValueError(f"Input M must be a 3x3 tensor. Got {M.shape}")
        
    src = np.array([src])
        
    B, _, H, W = src.shape  
    h_out, w_out = dsize
    
    # we normalize the 3x3 transformation matrix and convert to 3x4
    dst_norm_trans_src_norm = normalize_homography(M, (H, W), (h_out, w_out))  # Bx3x3
    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)  # Bx3x3
    
    # this piece of code substitutes F.affine_grid since it does not support 3x3
    grid = create_meshgrid(h_out, w_out, normalized_coordinates=True).astype(src.dtype)
    grid = np.tile(grid, (B, 1, 1, 1))
    grid = transform_points(src_norm_trans_dst_norm[:, None, None], grid)
    
    # values in range [-1,1], based on center of src image (larger-magnitude values are outside image):
    xs = np.squeeze(grid[:,:,:,1], axis=0) 
    ys = np.squeeze(grid[:,:,:,0], axis=0)
    
    # this could all probably be done in 2 lines if i weren't a python/numpy newbie :)
    src_height = src.shape[2]
    src_width = src.shape[3]
    xs = (xs + 1) / 2  #[0,1]
    ys = (ys + 1) / 2  #[0,1]
    xs = xs * src_height # i think xs/ys are misnamed? hence the product of xs and height
    ys = ys * src_width
    coords = np.array([xs.flatten(), ys.flatten()]);
    inputs = src[0]
    r = scipy.ndimage.map_coordinates(inputs[0], coords, order=interp_mode, mode=padding_mode, cval=cval[0])
    g = scipy.ndimage.map_coordinates(inputs[1], coords, order=interp_mode, mode=padding_mode, cval=cval[1])
    b = scipy.ndimage.map_coordinates(inputs[2], coords, order=interp_mode, mode=padding_mode, cval=cval[2])
    out = np.array([r.reshape(xs.shape), g.reshape(xs.shape), b.reshape(xs.shape)])
    out = np.array([out]) # add batch dim back
    return out[0]


def get_perspective_transform(src, dst):
    r"""Calculate a perspective transform from four pairs of the corresponding
    points.
    The function calculates the matrix of a perspective transform so that:
    .. math ::
        \begin{bmatrix}
        t_{i}x_{i}^{'} \\
        t_{i}y_{i}^{'} \\
        t_{i} \\
        \end{bmatrix}
        =
        \textbf{map_matrix} \cdot
        \begin{bmatrix}
        x_{i} \\
        y_{i} \\
        1 \\
        \end{bmatrix}
    where
    .. math ::
        dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3
    Args:
        src: coordinates of quadrangle vertices in the source image with shape :math:`(4, 2)`.
        dst: coordinates of the corresponding quadrangle vertices in
            the destination image with shape :math:`(4, 2)`.
    Returns:
        the perspective transformation with shape :math:`(3, 3)`.
    .. note::
        This function is often used in conjunction with :func:`warp_perspective`.
    """
    if not isinstance(src, np.ndarray):
        raise TypeError(f"Input type is not a np.ndarray. Got {type(src)}")

    if not isinstance(dst, np.ndarray):
        raise TypeError(f"Input type is not a np.ndarray. Got {type(dst)}")

    if not src.shape[-2:] == (4, 2):
        raise ValueError(f"Inputs must be a Bx4x2 tensor. Got {src.shape}")

    if not src.shape == dst.shape:
        raise ValueError(f"Inputs must have the same shape. Got {dst.shape}")
        
    src = np.array([src]).astype('float32')
    dst = np.array([dst]).astype('float32')
        
    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    p = []
    for i in [0, 1, 2, 3]:
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'x'))
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'y'))

    # A is Bx8x8
    A = np.stack(p, axis=1)

    # b is a Bx8x1
    b = np.stack(
        [
            dst[:, 0:1, 0],
            dst[:, 0:1, 1],
            dst[:, 1:2, 0],
            dst[:, 1:2, 1],
            dst[:, 2:3, 0],
            dst[:, 2:3, 1],
            dst[:, 3:4, 0],
            dst[:, 3:4, 1],
        ],
        axis=1,
    )

    # solve the system Ax = b
    X, _ = _torch_solve_cast(b, A)

    # create variable to return
    batch_size = src.shape[0]
    M = np.ones((batch_size, 9), dtype=src.dtype)
    
    if hasattr(np, 'DeviceArray'): # detect jax
        M = jax.ops.index_update(M, jax.ops.index[..., :8], np.squeeze(X, axis=-1))
    else:
        M[..., :8] = np.squeeze(X, axis=-1)

    return M.reshape((-1, 3, 3))[0]  # 3x3

def _build_perspective_param(p: np.ndarray, q: np.ndarray, axis: str) -> np.ndarray:
    ones = np.ones_like(p)[..., 0:1]
    zeros = np.zeros_like(p)[..., 0:1]
    if axis == 'x':
        return np.concatenate(
            [p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros, -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]], axis=1
        )

    if axis == 'y':
        return np.concatenate(
            [zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones, -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]], axis=1
        )

    raise NotImplementedError(f"perspective params for axis `{axis}` is not implemented.")



def _torch_inverse_cast(input: np.ndarray) -> np.ndarray:
    """Helper function to make torch.inverse work with other than fp32/64.
    The function torch.inverse is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    if not isinstance(input, np.ndarray):
        raise AssertionError(f"Input must be np.ndarray. Got: {type(input)}.")
    dtype: np.dtype = input.dtype
    if dtype not in (np.float32, np.float64):
        dtype = np.float32
    return np.linalg.inv(input.astype(dtype)).astype(input.dtype)

def _torch_solve_cast(input: np.ndarray, A: np.ndarray):
    """Helper function to make torch.solve work with other than fp32/64.
    The function torch.solve is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.svd, and cast back to the input dtype.
    """
    if not isinstance(input, np.ndarray):
        raise AssertionError(f"Input must be np.ndarray. Got: {type(input)}.")
    dtype: np.dtype = input.dtype
    if dtype not in (np.float32, np.float64):
        dtype = np.float32
    
#     out = solve(A.to(dtype), input.to(dtype))
    out = np.linalg.solve(A.astype(dtype), input.astype(dtype))

    return (out.astype(input.dtype), out)

def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Generate a coordinate grid for an image.
    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.
    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.
    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.
    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])
        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    xs = np.linspace(0, width - 1, width, dtype=dtype)
    ys = np.linspace(0, height - 1, height, dtype=dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid = np.meshgrid(xs, ys)
    base_grid = np.stack(base_grid)
#     base_grid = base_grid.transpose(1, 2)  # 2xHxW
    return np.expand_dims(base_grid, axis=0).transpose(0, 2, 3, 1)  # 1xHxWx2



def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    dtype = None,
) -> np.ndarray:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].
    Args:
        height image height.
        width: image width.
        eps: epsilon to prevent divide-by-zero errors
    Returns:
        normalized transform with shape :math:`(1, 3, 3)`.
    """
    tr_mat = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], dtype=dtype)  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0
    
    if hasattr(np, 'DeviceArray'): # detect jax
        tr_mat = jax.ops.index_update(tr_mat, jax.ops.index[0, 0], tr_mat[0, 0] * 2.0 / width_denom)
        tr_mat = jax.ops.index_update(tr_mat, jax.ops.index[1, 1], tr_mat[1, 1] * 2.0 / height_denom)
    else:
        tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
        tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return np.expand_dims(tr_mat, axis=0)  # 1x3x3

def normalize_homography(
    dst_pix_trans_src_pix: np.ndarray, dsize_src, dsize_dst
) -> np.ndarray:
    r"""Normalize a given homography in pixels to [-1, 1].
    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          normalized. :math:`(B, 3, 3)`
        dsize_src: size of the source image (height, width).
        dsize_dst: size of the destination image (height, width).
    Returns:
        the normalized homography of shape :math:`(B, 3, 3)`.
    """
    if not isinstance(dst_pix_trans_src_pix, np.ndarray):
        raise AssertionError(f"dst_pix_trans_src_pix must be np.ndarray. Got: {type(dst_pix_trans_src_pix)}.")

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {dst_pix_trans_src_pix.shape}")

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: np.ndarray = normal_transform_pixel(src_h, src_w).astype(dst_pix_trans_src_pix.dtype)

    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: np.ndarray = normal_transform_pixel(dst_h, dst_w).astype(dst_pix_trans_src_pix.dtype)

    # compute chain transformations
    dst_norm_trans_src_norm: np.ndarray = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm

def convert_points_to_homogeneous(points: np.ndarray) -> np.ndarray:
    r"""Function that converts points from Euclidean to homogeneous space.
    Args:
        points: the points to be transformed.
    Returns:
        the points in homogeneous coordinates.
    Examples:
        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = convert_points_to_homogeneous(input)  # BxNx4
    """
    if not isinstance(points, np.ndarray):
        raise TypeError(f"Input type is not a np.ndarray. Got {type(points)}")
    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    return np.pad(points, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1.0)

def convert_points_from_homogeneous(points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    r"""Function that converts points from homogeneous to Euclidean space.
    Args:
        points: the points to be transformed.
        eps: to avoid division by zero.
    Returns:
        the points in Euclidean space.
    Examples:
        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = convert_points_from_homogeneous(input)  # BxNx2
    """
    if not isinstance(points, np.ndarray):
        raise TypeError(f"Input type is not a np.ndarray. Got {type(points)}")

    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    # we check for points at max_val
    z_vec = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask = np.absolute(z_vec) > eps
    scale = np.where(mask, 1.0 / (z_vec + eps), np.ones_like(z_vec))

    return scale * points[..., :-1]


def transform_points(trans_01: np.ndarray, points_1: np.ndarray) -> np.ndarray:
    r"""Function that applies transformations to a set of points.
    Args:
        trans_01 (torch.Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (torch.Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        torch.Tensor: tensor of N-dimensional points.
    Shape:
        - Output: :math:`(B, N, D)`
    Examples:
        >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
        >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
        >>> points_0 = transform_points(trans_01, points_1)  # BxNx3
    """
    if not isinstance(trans_01, np.ndarray):
        raise TypeError(f"Input type is not a np.ndarray. Got {type(points)}")
    if not isinstance(points_1, np.ndarray):
        raise TypeError(f"Input type is not a np.ndarray. Got {type(points)}")
    if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
        raise ValueError("Input batch size must be the same for both tensors or 1")
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError("Last input dimensions must differ by one unit")

    # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
#     shape_inp = list(points_1.shape)
    points_1 = points_1.reshape((-1, points_1.shape[-2], points_1.shape[-1]))
    trans_01 = trans_01.reshape((-1, trans_01.shape[-2], trans_01.shape[-1]))
    # We expand trans_01 to match the dimensions needed for bmm
    trans_01 = np.repeat(trans_01, repeats=points_1.shape[0] // trans_01.shape[0], axis=0)
    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1
    # transform coordinates
    points_0_h = points_1_h @ trans_01.transpose(0, 2, 1)
    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
    # reshape to the input shape
#     if hasattr(np, 'DeviceArray'): # detect jax
#         shape_inp = jax.ops.index_update(shape_inp, jax.ops.index[-2], points_0.shape[-2])
#         shape_inp = jax.ops.index_update(shape_inp, jax.ops.index[-1], points_0.shape[-1])
#     else:
#         shape_inp[-2] = points_0.shape[-2]
#         shape_inp[-1] = points_0.shape[-1]
#     points_0 = points_0.reshape(shape_inp)
    points_0 = np.array([points_0])
    return points_0
