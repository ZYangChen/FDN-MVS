import torch

def inverse(map):
    inverse_map = 1.0 / (map + 1e-6)
    return inverse_map

def depth_normalization(depth, inverse_depth_min, inverse_depth_max):
    '''convert depth map to the index in inverse range'''
    inverse_depth = 1.0 / (depth + 1e-6)
    normalized_depth = (inverse_depth - inverse_depth_max) / (inverse_depth_min - inverse_depth_max + 1e-6)
    return normalized_depth

def depth_unnormalization(normalized_depth, inverse_depth_min, inverse_depth_max):
    '''convert the index in inverse range to depth map'''
    inverse_depth = inverse_depth_max + normalized_depth * (inverse_depth_min - inverse_depth_max) # [B,1,H,W]
    depth = 1.0 / (inverse_depth + 1e-6)
    return depth
'''
def depth_norm_fun(depth, depth_min, depth_width):
    depth_min = torch.min(depth).item()
    normalized_depth = (depth - depth_min) / (depth_width)
    return normalized_depth

def depth_unnorm_fun(depth, depth_min, depth_width):
    normalized_depth = depth * (depth_width) + depth_min
    return normalized_depth
'''
def depth_norm_fun(depth, depth_width):
    '''normalization'''
    depth_min = torch.min(depth).item()
    normalized_depth = (depth - depth_min) / (depth_width)
    return normalized_depth

def depth_unnorm_fun(depth, depth_width):
    '''normalization'''
    depth_min = torch.min(depth).item()
    normalized_depth = depth * (depth_width) + depth_min
    return normalized_depth
