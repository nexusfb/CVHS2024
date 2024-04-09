import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    # the region of interest from the frame
    roi = frame[ymin:ymax, xmin:xmax, :]

    # split into RGB channels
    r = roi[:, :, 0]
    g = roi[:, :, 1]
    b = roi[:, :, 2]

    # histogram values for each channel
    r_hist, r_edges = np.histogram(r.flatten(), bins=hist_bin, range=[0, 256])
    g_hist, g_edges = np.histogram(g.flatten(), bins=hist_bin, range=[0, 256])
    b_hist, b_edges = np.histogram(b.flatten(), bins=hist_bin, range=[0, 256])

    # concatenate all histograms
    hist = np.concatenate((r_hist, g_hist, b_hist))

    # normalize the histogram
    hist = hist / np.sum(hist)

    return hist