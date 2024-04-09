import numpy as np

def estimate(particles, particles_w):
    mean_state = np.dot(particles.T, particles_w)
    return mean_state.T