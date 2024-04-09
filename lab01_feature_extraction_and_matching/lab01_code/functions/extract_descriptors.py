import numpy as np

def filter_keypoints(img, keypoints, patch_size = 9):
    '''
    Inputs:
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    Returns:
    - keypoints:    (q', 2) numpy array of keypoint locations [x, y] that are far enough from edges
    '''
    # 1 -  img dimensions
    height, width = img.shape[:2]
    # 2 - minimum distance from the edge based on half of the patch size
    min_distance = patch_size // 2
    # 3 - Calculate the distance of each keypoint from the edges
    x_distances = np.minimum(keypoints[:, 0], width - keypoints[:, 0])
    y_distances = np.minimum(keypoints[:, 1], height - keypoints[:, 1])
    # 4 - Create a mask to filter keypoints based on distance
    mask = (x_distances >= min_distance) & (y_distances >= min_distance)
    # 5 - Use the mask to select the filtered keypoints
    filtered_keypoints = keypoints[mask]
    return filtered_keypoints


# The implementation of the patch extraction is already provided here
def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(img, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc = desc.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return desc

