import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost



def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist_target, sigma_observe):
    height_frame, width_frame, _ = frame.shape
    n_particles = len(particles)
    particles_w = np.zeros(n_particles)

    # color histograms for all particles
    hist_particles = np.zeros((n_particles, hist_bin * 3))
    for i in range(n_particles):
        # bounding box around the particle center
        xmin = max(0, int(np.floor(particles[i, 0] - 0.5 * bbox_width)))
        ymin = max(0, int(np.floor(particles[i, 1] - 0.5 * bbox_height)))
        xmax = min(width_frame, int(np.floor(particles[i, 0] + 0.5 * bbox_width)))
        ymax = min(height_frame, int(np.floor(particles[i, 1] + 0.5 * bbox_height)))
        
        # color histogram for the particle within the bounding box
        hist_particles[i, :] = color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin)
        
        # chi2 distance between the particle histogram and the target histogram
        chi2_distance = chi2_cost(hist_target, hist_particles[i, :])

        # update the weight of the particle using a Gaussian distribution
        particles_w[i] = (1 / (np.sqrt(2 * np.pi) * sigma_observe)) * \
                          np.exp(-0.5 * chi2_distance**2 / sigma_observe**2)

    # normalize particle weights
    particles_w = particles_w / np.sum(particles_w)

    return particles_w