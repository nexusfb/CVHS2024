import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    # 1 - Check that the dimensions of the descriptors match
    assert desc1.shape[1] == desc2.shape[1]
    # 2 - Compute the squared differences
    diff = desc1[:, np.newaxis, :] - desc2
    distances = np.sum(diff ** 2, axis=2)
    return distances

def match_descriptors(desc1, desc2, method="one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    # 1 - Check that the dimensions of the descriptors match
    assert desc1.shape[1] == desc2.shape[1]
    # 2 - Compute the squared distances between descriptors using SSD
    distances = ssd(desc1, desc2)
    # 3 - Get the number of descriptors in each set
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    # 4 - method: one way (query the nearest neighbor for each keypoint in image 1)
    if method == "one_way":
        # 4.1 - Find the index of the nearest neighbor in image 2 for each keypoint in image 1
        match_indices = np.argmin(distances, axis=1)
        # 4.2 - Create matches array
        matches = np.column_stack((np.arange(q1), match_indices))
    # 5 - method: mutual (query the nearest neighbor for each keypoint in image 1 and check for mutual agreement)
    elif method == "mutual":
        # 5.1 - Create array for keypoint indices 
        indices = np.arange(q1)
        # 5.2 - Find closest keypoint index for each keypoint.
        match_idxs = np.argmin(distances, axis=1)
        # 5.3 - Check if the matched keypoints also agree that they are the closest keypoints
        mutual_match_idxs = np.argmin(distances[:, match_idxs], axis=0)
        # 5.4 - Check if they correspond
        mask = mutual_match_idxs == indices
        # 5.5 - save the matched keypoint indices
        matches = np.column_stack((indices[mask], match_idxs[mask]))
    # 6 - method: ratio (check if ratio is below the threshold for the distance ratios between the best and second-best matches)
    elif method == "ratio":
        # 6.1 - Sort the indices of descriptors in image 2 by distances to descriptors in image 1
        sorted_indices = np.argsort(distances, axis=1)
        # 6.2 - Get indices of the two closest keypoints for all rows
        best_match_indices = sorted_indices[:, 0]
        second_best_match_indices = sorted_indices[:, 1]
        # 6.3 - Compute ratio of distances between the best and second-best matches
        ratios = distances[np.arange(q1), best_match_indices] / distances[np.arange(q1), second_best_match_indices]
        # 6.4 - Find the indices of valid matches where the distance ratio is below the specified threshold
        matching_indices = np.where(ratios < ratio_thresh)[0]
        # 6.5 - Create matches array
        matches = np.column_stack((np.arange(q1)[matching_indices], best_match_indices[matching_indices]))
    else:
        raise ValueError("Invalid matching method")

    return matches
