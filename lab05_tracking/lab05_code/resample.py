import numpy as np

def resample(particles, particles_w):
    n_particles = len(particles)

    # random number in the range [0, 1) for resampling
    r = 1 / n_particles * np.random.rand()
    
    c = particles_w[0]
    i = 0
    new_particles = []
    new_particles_w = []

    # resampling loop
    for m in range(1, n_particles + 1):
        # threshold for selecting the next particle
        U = r + 1 / n_particles * (m - 1)
        
        # update the index until the cumulative weight exceeds the threshold
        while U > c:
            i += 1
            c += particles_w[i]
        new_particles.append(particles[i, :])
        new_particles_w.append(particles_w[i])

    # normalize the new particle weights
    new_particles_w = np.array(new_particles_w) / np.sum(new_particles_w)

    return np.array(new_particles), new_particles_w
