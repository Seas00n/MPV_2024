import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import math
import numpy as np
from scipy.signal import buttap, lp2hp_zpk, bilinear_zpk, zpk2tf, butter


def RANSAC(X, y, th):
    def is_data_valid(X_subset, y_subset):
        x = X_subset
        y = y_subset

        if abs(x[1] - x[0]) < 0.05:
            return False
        else:
            k = (y[1] - y[0]) / (x[1] - x[0])

        theta = math.atan(k)

        if abs(theta) < th:
            r = True
        else:
            r = False

        return r

    ransac = linear_model.RANSACRegressor(min_samples=2, residual_threshold=0.025,
                                          is_data_valid=is_data_valid,
                                          max_trials=400)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    # Predict calibrate of estimated models
    line_X = np.arange(X.min(), X.max(), 0.01)[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    theta_line = math.atan((line_y_ransac[-1] - line_y_ransac[0]) / (line_X[-1] - line_X[0]))
    return inlier_mask, outlier_mask, line_y_ransac, line_X, theta_line


def tons_lowpass_filter(cutoff_freq, sample_time, x0, x1, y1, yd1):
    # Compute coefficients of the state equation
    a = (2 * np.pi * cutoff_freq) ** 2
    b = np.sqrt(2) * 2 * np.pi * cutoff_freq

    # Integrate the filter state equation using the midpoint Euler method with step h
    h = sample_time
    denom = 4 + 2 * h * b + h ** 2 * a

    A = (4 + 2 * h * b - h ** 2 * a) / denom
    B = 4 * h / denom
    C = -4 * h * a / denom
    D = (4 - 2 * h * b - h ** 2 * a) / denom
    E = 2 * h ** 2 * a / denom
    F = 4 * h * a / denom

    y = A * y1 + B * yd1 + E * (x0 + x1) / 2
    yd = C * y1 + D * yd1 + F * (x0 + x1) / 2

    return y, yd


def jason_rewrite_of_tons_lowpass_filter(cutoff_freq, sample_time, x0, x1, x2, y1, y2):
    # Compute coefficients of the state equation
    a = (2 * np.pi * cutoff_freq) ** 2
    b = np.sqrt(2) * 2 * np.pi * cutoff_freq

    # Integrate the filter state equation using the midpoint Euler method with step h
    h = sample_time
    denom = 4 + 2 * h * b + h ** 2 * a

    A = (4 + 2 * h * b - h ** 2 * a) / denom
    B = 4 * h / denom
    C = -4 * h * a / denom
    D = (4 - 2 * h * b - h ** 2 * a) / denom
    E = 2 * h ** 2 * a / denom
    F = 4 * h * a / denom

    y = (A + B * D / h) * y1 + (B * C - B * D / h) * y2 + E / 2 * x0 + (B * F / 2 + E / 2) * x1 + B * F / 2 * x2

    return y


def jasons_highpass_filter(cutoff_freq, sample_time, x0, x1, x2, y1, y2):
    cutoff_freq = 1 / 2 / sample_time - cutoff_freq  # covert high pass freq to equivalent lowpass freq

    # Compute coefficients of the state equation
    a = (2 * np.pi * cutoff_freq) ** 2
    b = np.sqrt(2) * 2 * np.pi * cutoff_freq

    # Integrate the filter state equation using the midpoint Euler method with step h
    h = sample_time
    denom = 4 + 2 * h * b + h ** 2 * a

    A = (4 + 2 * h * b - h ** 2 * a) / denom
    B = 4 * h / denom
    C = -4 * h * a / denom
    D = (4 - 2 * h * b - h ** 2 * a) / denom
    E = 2 * h ** 2 * a / denom
    F = 4 * h * a / denom

    y = -(A + B * D / h) * y1 + (B * C - B * D / h) * y2 + E / 2 * x0 - (B * F / 2 + E / 2) * x1 + B * F / 2 * x2

    return y


if __name__ == "__main__":
    low_freq = 3
    high_freq = 10
    time = np.linspace(0, 10, num=500)
    sample_rate = 1 / np.mean(np.diff(time))
    freq = np.linspace(1, 15, 10)
    sig = np.zeros_like(time)
    for i in range(np.shape(freq)[0]):
        sig = sig + np.sin(2*np.pi*freq[i]*time)
    # tons_lowpass_filt_sig = np.zeros_like(sig)
    # tons_lowpass_filt_sigd = np.zeros_like(sig)
    jasons_lowpass_filt_sig = np.zeros_like(sig)
    jasons_highpass_filt_sig = np.zeros_like(sig)
    jasons_lowpass_highpass_filt_sig = np.zeros_like(sig)
    for i in range(2, len(time)):
        jasons_lowpass_filt_sig[i] = \
            jason_rewrite_of_tons_lowpass_filter(cutoff_freq=low_freq,
                                                 sample_time=time[i] - time[i - 1],
                                                 x0=sig[i], x1=sig[i - 1], x2=sig[i - 2],
                                                 y1=jasons_lowpass_filt_sig[i - 1],
                                                 y2=jasons_lowpass_filt_sig[i - 2])
        jasons_highpass_filt_sig[i] = \
            jasons_highpass_filter(cutoff_freq=high_freq,
                                   sample_time=time[i] - time[i - 1],
                                   x0=sig[i], x1=sig[i - 1], x2=sig[i - 2],
                                   y1=jasons_highpass_filt_sig[i - 1],
                                   y2=jasons_highpass_filt_sig[i - 2])
        jasons_lowpass_highpass_filt_sig[i] = \
            jasons_highpass_filter(cutoff_freq=high_freq,
                                   sample_time=time[i] - time[i - 1],
                                   x0=jasons_lowpass_filt_sig[i],
                                   x1=jasons_lowpass_filt_sig[i - 1],
                                   x2=jasons_lowpass_filt_sig[i - 2],
                                   y1=jasons_highpass_filt_sig[i - 1],
                                   y2=jasons_highpass_filt_sig[i - 2])

    fig, axes = plt.subplots(4, 1, sharex=True)
    axes[0].plot(time, sig)
    axes[1].plot(time, sig)
    axes[1].plot(time, jasons_lowpass_filt_sig, '-.', linewidth=4, label='Jasons')
    axes[2].plot(time, sig)
    axes[2].plot(time, jasons_highpass_filt_sig, '-.', linewidth=4, label='Jasons')
    axes[3].plot(time, sig)
    axes[3].plot(time, jasons_lowpass_highpass_filt_sig, '-.', linewidth=1, label='Jasons')
    for ax in axes:
        ax.set_xlim((0, 2))
    plt.show()
