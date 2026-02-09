
import torch
def normalize_position_to_range(xy, map_size, map_center, target_range=(0, 2)):
    """
    Normalize position to a specific target range (e.g., [0, 2]) for RBF centers.

    Args:
        xy: Tensor of shape (..., 2)
        map_size: Tuple (map_width, map_height)
        map_center: Tuple (x_center, y_center)
        target_range: Tuple (min_val, max_val)

    Returns:
        Normalized tensor of shape (..., 2)
    """
    # Compute world min/max
    min_x = map_center[0] - map_size[0] / 2
    min_y = map_center[1] - map_size[1] / 2
    width, height = map_size

    norm_x = (xy[..., 0] - min_x) / width   # scale to [0, 1]
    norm_y = (xy[..., 1] - min_y) / height  # scale to [0, 1]

    scaled_x = norm_x * (target_range[1] - target_range[0]) + target_range[0]
    scaled_y = norm_y * (target_range[1] - target_range[0]) + target_range[0]

    return torch.stack([scaled_x, scaled_y], dim=-1)

def normalize_rgb(image):
    """
    Normalize RGB image (c,h,w)
    """
    return image.float() / 255.0

def normalize_position_to_range(pos, map_size, map_center):
    """
    Normalize position to a rectified coordinate space while preserving aspect ratio.

    Args:
        pos (Tensor): (N, 2) position tensor in world coords
        map_size (tuple): (width, height)
        map_center (tuple): (x_center, y_center)

    Returns:
        Tensor: normalized position in shape (N, 2)
    """
    if not isinstance(pos, torch.Tensor):
        pos = torch.tensor(pos, dtype=torch.float32)

    map_size = torch.tensor(map_size, dtype=torch.float32, device=pos.device)
    map_center = torch.tensor(map_center, dtype=torch.float32, device=pos.device)

    # shift to center
    shifted = pos - map_center  
    normalized = shifted / map_size  

    # scale to [0, 2] (x) and [0, 2 * height/width] (y)
    scale_x = 2.0
    scale_y = 2.0 * (map_size[1] / map_size[0])  # height/width

    norm_x = (normalized[:, 0] + 0.5) * scale_x
    norm_y = (normalized[:, 1] + 0.5) * scale_y

    scaled = torch.stack([norm_x, norm_y], dim=-1)

    return scaled


def sincos_encode_angle(theta):
    """
    Encode angle as (sin, cos)
    :param theta: Tensor of shape (...,)
    :return: Tensor of shape (..., 2)
    """
    return torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)

def sincos_encode_velocity(angular_velocity):
    """
    Convert angular velocity scalar to sin/cos encoding.
    :param angular_velocity: Tensor of shape (...,)
    :return: Tensor of shape (..., 2)
    """
    return torch.stack([torch.sin(angular_velocity), torch.cos(angular_velocity)], dim=-1)
