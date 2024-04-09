import numpy as np

def propagate(particles, frame_height, frame_width, params):
    n_particles, dim = particles.shape

    if params['model'] == 0:  # no motion
        A = np.eye(dim)
        # update particle positions using the no motion model and add random noise
        particles = A @ particles.T + np.random.randn(dim, n_particles) * params['sigma_position']
        particles = particles.T
    else:  # constant velocity
        # system matrix for constant velocity motion model
        A = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        # random noise for position and velocity
        noise_position = np.random.randn(2, n_particles) * params['sigma_position']
        noise_velocity = np.random.randn(2, n_particles) * params['sigma_velocity']
        # update particle positions using the constant velocity motion model and noise
        particles = A @ particles.T + np.vstack((noise_position, noise_velocity))
        particles = particles.T

    # check that the particles lie inside the frame
    particles[:, 0] = np.clip(particles[:, 0], 0, frame_width)
    particles[:, 1] = np.clip(particles[:, 1], 0, frame_height)

    return particles