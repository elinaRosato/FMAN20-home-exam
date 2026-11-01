import os
import numpy as np
import cv2
from scipy.spatial import cKDTree

SQUARE = 42          # image 1 is cut into squares of 42 x 42 pixels for the test set and the folds
N_FOLDS = 5


def save_image(out_dir, filename, img):
    """Save an image (BGR, or single-channel) at full resolution into out_dir."""
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, img)
    print(f"saved {path}")


def gb_features(g, b):
    """One row [G, B] per pixel."""
    return np.stack([g.ravel(), b.ravel()], axis=1)


# ---------------------------------------------------------------------------------------------------------------
# The two methods. Each one learns from (X_train, y_train) = ([G, B], R) and predicts the red value of the rows of X_query.
# ---------------------------------------------------------------------------------------------------------------
def linear_regression(X_train, y_train, X_query):
    """Least squares fit of R = a*G + b*B + c."""
    A = np.column_stack([X_train, np.ones(len(X_train))])
    w, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    return np.column_stack([X_query, np.ones(len(X_query))]) @ w


def knn_regression(X_train, y_train, X_query, k=100, step=3):
    """k nearest neighbours: the prediction for a pixel is a weighted average of the red values of the k training pixels
    with the most similar (G, B). The weights are 1 / distance^2, so the closest matches count most. Only every
    `step`th training pixel is used (faster)."""
    tree = cKDTree(X_train[::step])
    values = y_train[::step]
    result = np.zeros(len(X_query), np.float32)
    for start in range(0, len(X_query), 20000):                             # in chunks, so that a large k fits in memory
        distances, indices = tree.query(X_query[start:start + 20000], k=list(range(1, k + 1)))
        weights = 1.0 / (distances ** 2 + 1e-6)
        weights /= weights.sum(axis=1, keepdims=True)
        result[start:start + 20000] = (weights * values[indices]).sum(axis=1)
    return result


def knn(k):
    return lambda X_train, y_train, X_query: knn_regression(X_train, y_train, X_query, k)


def anisotropic_diffusion(image, iterations=2, kappa=100.0, step=0.2):
    """Noise reduction that keeps the edges: Perona-Malik anisotropic diffusion. In every iteration each pixel moves
    towards its four neighbours, u += step * sum(c(d) * d), where d is the difference to a neighbour. The conductance
    c(d) = exp(-(d / kappa)^2) is close to 1 for small differences (noise, which is smoothed) and close to 0 for large
    differences (edges, which stay). kappa sets how large a difference has to be to count as an edge."""
    u = image.astype(np.float32).copy()
    for _ in range(iterations):
        padded = np.pad(u, 1, mode='edge')
        differences = [padded[:-2, 1:-1] - u, padded[2:, 1:-1] - u, padded[1:-1, :-2] - u, padded[1:-1, 2:] - u]
        u += step * sum(np.exp(-(d / kappa) ** 2) * d for d in differences)
    return np.clip(u, 0, 255)


def gaussian_filter(image, sigma):
    return cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma)


def rmse(red, r1, where):
    """RMSE of the red channel over the pixels `where`, in grey levels."""
    return np.sqrt(np.mean((red - r1)[where] ** 2))


def to_bgr(b, g, r):
    return cv2.merge([np.clip(c, 0, 255).astype(np.uint8) for c in (b, g, r)])


# ---------------------------------------------------------------------------------------------------------------
# Test set and cross-validation on image 1
# ---------------------------------------------------------------------------------------------------------------
def make_splits(shape):
    """Cut image 1 into squares and divide them into a TEST set and N_FOLDS FOLDS with a regular pattern: square (x, y)
    goes to group (x + 2*y) mod (N_FOLDS + 1). Every group has the same number of squares, they are evenly spread over
    the image, and every row of squares contains all the groups. Group 0 is the test set and groups 1 ... N_FOLDS are
    the folds. Returns an image with -1 for the test squares and 0 ... N_FOLDS-1 for the folds."""
    ny, nx = shape[0] // SQUARE, shape[1] // SQUARE
    iy, ix = np.mgrid[0:ny, 0:nx]
    group = (ix + 2 * iy) % (N_FOLDS + 1)
    return np.kron(group - 1, np.ones((SQUARE, SQUARE), int))


def cross_validated_red(method, X, r1, split):
    """Red channel of every pixel of image 1, predicted by a model that has not seen the pixel. The pixels of each fold
    are predicted by a model that learned from the other folds. The test pixels are predicted by a model that learned
    from all the folds (the test pixels are never used for learning, and not for choosing any setting)."""
    y, labels = r1.ravel(), split.ravel()
    red = np.zeros(len(y), np.float32)
    for group in list(range(N_FOLDS)) + [-1]:
        train = (labels >= 0) & (labels != group)
        query = labels == group
        red[query] = np.clip(method(X[train], y[train], X[query]), 0, 255)
    return red.reshape(r1.shape)


def save_splits(out_dir, img1, split):
    """First picture: the test set (pink). Then one picture for each round of the cross-validation: the test squares
    are black (they are not used at all), the squares of the fold that is held out in this round are pink (validation),
    and the other squares are left as they are (they are used for learning)."""
    def panel(name, pink, black=None):
        tile = img1.copy()
        if black is not None:
            tile[black] = 0
        tile[pink] = (0.5 * tile[pink] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        cv2.putText(tile, name, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (255, 255, 255), 7)
        return tile
    tiles = [panel("test set", split == -1)]
    tiles += [panel(f"round {f + 1}", split == f, black=split == -1) for f in range(N_FOLDS)]
    save_image(out_dir, "splits.png", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))


def choose_k(X, r1, split, ks=(5, 10, 25, 50, 100, 200, 400, 800)):
    """The k with the lowest cross-validation RMSE, and the cross-validated red channel for that k."""
    development = split >= 0
    print("k nearest neighbours, 5-fold cross-validation (no noise reduction):")
    by_k = {k: cross_validated_red(knn(k), X, r1, split) for k in ks}
    for k in ks:
        print(f"  k = {k:3d}: RMSE {rmse(by_k[k], r1, development):5.2f}")
    best_k = min(ks, key=lambda k: rmse(by_k[k], r1, development))
    print(f"  -> k = {best_k}")
    return best_k, by_k[best_k]


def choose_noise_reduction(base, r1, split, sigmas=(0.3, 0.5, 0.7, 1, 1.5, 2), iterations=(1, 2, 3, 5),
                           kappas=(60, 100, 150, 200, 300)):
    """The strength of the Gaussian filter and of the anisotropic diffusion with the lowest cross-validation RMSE, when
    they are applied to the cross-validated k-NN result `base`. Returns the two settings and the two filtered images."""
    development = split >= 0
    gaussian = {sigma: gaussian_filter(base, sigma) for sigma in sigmas}
    anisotropic = {(it, kappa): anisotropic_diffusion(base, it, kappa) for kappa in kappas for it in iterations}
    for sigma, red in gaussian.items():
        print(f"  Gaussian sigma = {sigma:3}: RMSE {rmse(red, r1, development):5.2f}")
    for (it, kappa), red in anisotropic.items():
        print(f"  anisotropic, {it} iterations, kappa = {kappa:3d}: RMSE {rmse(red, r1, development):5.2f}")
    best_sigma = min(gaussian, key=lambda key: rmse(gaussian[key], r1, development))
    best_aniso = min(anisotropic, key=lambda key: rmse(anisotropic[key], r1, development))
    print(f"  -> Gaussian sigma = {best_sigma}, anisotropic diffusion with {best_aniso[0]} iterations and kappa = {best_aniso[1]}")
    return best_sigma, best_aniso, gaussian[best_sigma], anisotropic[best_aniso]


def evaluate_on_test(results, img1, split, out_dir):
    """RMSE of every method on the test squares, and a picture of each (the test squares predicted, the rest as in image 1)."""
    b1, g1, r1 = [c.astype(np.float32) for c in cv2.split(img1)]
    test = split == -1
    print("The methods on the test set (never used for learning or for choosing settings):")
    for name, red in results.items():
        print(f"  {name:16s}: RMSE {rmse(red, r1, test):5.2f}")
        image = to_bgr(b1, g1, np.where(test, red, r1))
        for y in range(0, g1.shape[0], SQUARE):
            for x in range(0, g1.shape[1], SQUARE):
                if test[y, x]:
                    cv2.rectangle(image, (x, y), (x + SQUARE - 1, y + SQUARE - 1), (255, 255, 255), 1)
        save_image(out_dir, f"test_{name}.png", image)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, 'output', 'task5')
    os.makedirs(out_dir, exist_ok=True)

    # load the two images. OpenCV gives the channels in the order B, G, R
    img1 = cv2.imread(os.path.join(here, 'data', 'cars_sm_1.png'))      # complete image
    img2 = cv2.imread(os.path.join(here, 'data', 'cars_sm_2.png'))      # red channel is all zero
    b1, g1, r1 = [c.astype(np.float32) for c in cv2.split(img1)]
    b2, g2, _ = [c.astype(np.float32) for c in cv2.split(img2)]
    save_image(out_dir, "image1.png", img1)
    save_image(out_dir, "image2_given.png", img2)
    X1, X2 = gb_features(g1, b1), gb_features(g2, b2)

    # Choose k and the noise reduction on image 1, where the red channel is known
    split = make_splits(r1.shape)
    save_splits(out_dir, img1, split)
    development, test = split >= 0, split == -1
    print(f"Test set: {test.mean() * 100:.0f} % of image 1, folds: {[f'{(split == f).mean() * 100:.0f} %' for f in range(N_FOLDS)]}")
    print("Average red per fold:", [round(float(r1[split == f].mean()), 1) for f in range(N_FOLDS)],
          "| test set:", round(float(r1[test].mean()), 1))

    linear = cross_validated_red(linear_regression, X1, r1, split)
    print(f"Linear regression (nothing to choose), cross-validation: RMSE {rmse(linear, r1, development):.2f}")
    best_k, knn_cv = choose_k(X1, r1, split)
    print(f"Noise reduction of the k = {best_k} result (cross-validation):")
    best_sigma, (iterations, kappa), gaussian_cv, anisotropic_cv = choose_noise_reduction(knn_cv, r1, split)

    evaluate_on_test({"linear": linear, "knn": knn_cv, "knn_gaussian": gaussian_cv, "knn_anisotropic": anisotropic_cv,
                      "average": np.mean([linear, anisotropic_cv], axis=0)}, img1, split, out_dir)

    # Predict
    reds = {"linear": np.clip(linear_regression(X1, r1.ravel(), X2), 0, 255).reshape(g2.shape),
            "knn": np.clip(knn_regression(X1, r1.ravel(), X2, k=best_k), 0, 255).reshape(g2.shape)}
    # The red channel of the k-NN is noisy, so it gets an extra noise reduction step. Two filters are compared.
    reds["knn_anisotropic"] = anisotropic_diffusion(reds["knn"], iterations, kappa)
    reds["knn_gaussian"] = gaussian_filter(reds["knn"], best_sigma)
    # Average of the red channels of linear regression and the denoised k-NN (anisotropic)
    reds["average"] = np.mean([reds["linear"], reds["knn_anisotropic"]], axis=0)
    for name, red in reds.items():
        save_image(out_dir, f"image2_R_{name}.png", red.astype(np.uint8))
        save_image(out_dir, f"image2_corrected_{name}.png", to_bgr(b2, g2, red))


if __name__ == "__main__":
    main()
