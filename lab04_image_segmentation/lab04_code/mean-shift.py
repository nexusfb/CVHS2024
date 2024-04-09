import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    return np.linalg.norm(X - x, axis=1)

def gaussian(dist, bandwidth):
    return  np.exp(-0.5 * (dist / bandwidth)**2) / (bandwidth * np.sqrt(2 * np.pi))

def update_point(weight, X):
    return np.sum(weight[:, np.newaxis] * X, axis=0) / np.sum(weight)

def meanshift_step(X, bandwidth=2.5):
    y = np.copy(X)
    for i in range(X.shape[0]):
        dist = distance(X[i], X)
        weight = gaussian(dist, bandwidth)
        y[i] = update_point(weight, X)
    return y

def meanshift(X):
    for i in range(20):
        X = meanshift_step(X)
    return X

def reduce_labels(labels):
    # Find the 24 most frequent labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    most_frequent_labels = unique_labels[np.argsort(counts)[-24:]]

    # Create a mapping from less frequent labels to the nearest frequent label
    label_mapping = {}
    for i, label in enumerate(most_frequent_labels):
        label_mapping[label] = i

    # Apply the mapping to reduce labels and map to the range 0-23
    reduced_labels = np.array([label_mapping.get(label, len(most_frequent_labels)) for label in labels])
    reduced_labels = reduced_labels % 24  # Map to the range 0-23

    return reduced_labels
scale = 0.5    # downscale the image to run faster


# Load image and convert it to CIELAB space
image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(image_lab)
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))


# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0
centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

# if 
if np.max(labels) > 23:
    # if kernel is too narrow
    labels = reduce_labels(labels)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
