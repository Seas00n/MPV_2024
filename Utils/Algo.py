from sklearn import linear_model, datasets
import math
import numpy as np
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

    ransac = linear_model.RANSACRegressor(min_samples=2, residual_threshold=0.025, is_data_valid=is_data_valid,
                                          max_trials=400)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max(), 0.01)[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)

    return inlier_mask, outlier_mask, line_y_ransac, line_X