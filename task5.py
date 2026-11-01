import os
import numpy as np
import cv2
from scipy.spatial import cKDTree


def save_image(out_dir, filename, img):
    """Save an image (BGR, or single-channel) at full resolution into out_dir."""
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, img)
    print(f"saved {path}")


def linear_features(g, b):
    """[G, B, 1] per pixel, so that R = a*G + b*B + c is a matrix product."""
    return np.stack([g.ravel(), b.ravel(), np.ones(g.size)], axis=1)


def gb_features(g, b):
    return np.stack([g.ravel(), b.ravel()], axis=1)


# ---------------------------------------------------------------------------------------------------------------
# The two methods. Each learns from (features, red) of image 1 and returns a function that predicts red.
# ---------------------------------------------------------------------------------------------------------------
def fit_linear(X, y):
    """Least squares fit of R = a*G + b*B + c."""
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return lambda Xq: Xq @ w


def fit_knn(X, y, k=50, step=3):
    """k nearest neighbours: the prediction for a pixel is a weighted average of y over the k training pixels with the
    most similar features. The weights are 1 / distance^2 (inverse-distance weighting), so the closest matches count
    most. Only every `step`th pixel is used as training data (faster)."""
    tree = cKDTree(X[::step])
    values = y[::step]

    def predict(Xq):
        result = np.zeros(len(Xq), np.float32)
        for start in range(0, len(Xq), 20000):                              # in chunks, so that a large k fits in memory
            distances, indices = tree.query(Xq[start:start + 20000], k=list(range(1, k + 1)))
            weights = 1.0 / (distances ** 2 + 1e-6)
            weights /= weights.sum(axis=1, keepdims=True)
            result[start:start + 20000] = (weights * values[indices]).sum(axis=1)
        return result
    return predict


# method = (name, features, fit)
LINEAR = ("linear", linear_features, fit_linear)


def knn_method(k):
    return ("knn", gb_features, lambda X, y: fit_knn(X, y, k=k))


def predict_red(method, train, new, train_mask=None, query_mask=None):
    """Learn from the image `train` = (b, g, r) and predict the red channel of `new` = (b, g). If `train_mask` is
    given, only the pixels where it is True are used for learning. If `query_mask` is given, the red channel is only
    predicted where it is True (faster), and is 0 elsewhere."""
    name, features, fit = method
    b1, g1, r1 = train
    b2, g2 = new
    X, y = features(g1, b1), r1.ravel()
    if train_mask is not None:
        X, y = X[train_mask.ravel()], y[train_mask.ravel()]
    predict = fit(X, y)
    query = features(g2, b2)
    if query_mask is None:
        prediction = predict(query)
    else:
        prediction = np.zeros(g2.size, np.float32)
        prediction[query_mask.ravel()] = predict(query[query_mask.ravel()])
    return np.clip(prediction, 0, 255).reshape(g2.shape)


def make_splits(shape, size=42, n_folds=5):
    """Cut image 1 into squares of size x size pixels and divide them into a TEST set and n_folds FOLDS with a regular
    pattern: square (x, y) goes to group (x + 2*y) mod (n_folds + 1). Every group has the same number of squares,
    they are evenly spread over the image, and every row of squares contains all the groups. Group 0 is the test set
    and groups 1 ... n_folds are the folds. Returns an image with -1 for the test squares and 0 ... n_folds-1 for
    the folds."""
    ny, nx = shape[0] // size, shape[1] // size
    iy, ix = np.mgrid[0:ny, 0:nx]
    group = (ix + 2 * iy) % (n_folds + 1)
    return np.kron(group - 1, np.ones((size, size), int))


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


def to_bgr(b, g, r):
    return cv2.merge([c.astype(np.uint8) for c in (b, g, r)])


def rmse(red, r1, where):
    """RMSE of the red channel over the pixels `where`, in grey levels."""
    return np.sqrt(np.mean((red - r1)[where] ** 2))


def fold_predictions(method, b1, g1, r1, split):
    """Red channel of every pixel of image 1, predicted by a model that has not seen the pixel.
    The pixels of the folds are predicted one fold at a time by a model that learned from the other folds. The
    test pixels are predicted by a model that learned from all the folds. (The test pixels are never used for
    learning, and are not used to choose any setting.)"""
    development = split >= 0
    red = np.zeros(g1.shape, np.float32)
    for fold in range(split.max() + 1):
        predicted = predict_red(method, (b1, g1, r1), (b1, g1), train_mask=development & (split != fold),
                                query_mask=split == fold)
        red[split == fold] = predicted[split == fold]
    predicted = predict_red(method, (b1, g1, r1), (b1, g1), train_mask=development, query_mask=split == -1)
    red[split == -1] = predicted[split == -1]
    return red


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
    tiles += [panel(f"round {f + 1}", split == f, black=split == -1) for f in range(split.max() + 1)]
    save_image(out_dir, "splits.png", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))


def choose_settings(img1, out_dir, ks=(5, 10, 25, 50, 100, 200, 400, 800),
                    sigmas=(0.3, 0.5, 0.7, 1, 1.5, 2),
                    iterations=(1, 2, 3, 5), kappas=(60, 100, 150, 200, 300)):
    """Choose k and the strength of the noise reduction with 5-fold cross-validation on the development part of image 1
    (everything except the test set), and then measure the methods with the chosen settings on the test set."""
    b1, g1, r1 = [c.astype(np.float32) for c in cv2.split(img1)]
    split = make_splits(r1.shape)
    save_splits(out_dir, img1, split)
    development, test = split >= 0, split == -1
    print(f"Test set: {test.mean() * 100:.0f} % of image 1, folds: {[f'{(split == f).mean() * 100:.0f} %' for f in range(5)]}")
    print("Average red per fold:", [round(float(r1[split == f].mean()), 1) for f in range(5)],
          "| test set:", round(float(r1[test].mean()), 1))

    linear = fold_predictions(LINEAR, b1, g1, r1, split)
    print(f"Linear regression (nothing to choose), cross-validation: RMSE {rmse(linear, r1, development):.2f}")
    print("k nearest neighbours, 5-fold cross-validation (no noise reduction):")
    by_k = {k: fold_predictions(knn_method(k), b1, g1, r1, split) for k in ks}
    for k in ks:
        print(f"  k = {k:3d}: RMSE {rmse(by_k[k], r1, development):5.2f}")
    best_k = min(ks, key=lambda k: rmse(by_k[k], r1, development))
    print(f"  -> k = {best_k}")

    print(f"Noise reduction of the k = {best_k} result (cross-validation):")
    base = by_k[best_k]
    gaussian = {sigma: cv2.GaussianBlur(base, (0, 0), sigma) for sigma in sigmas}
    anisotropic = {(it, kappa): anisotropic_diffusion(base, it, kappa) for kappa in kappas for it in iterations}
    for sigma, red in gaussian.items():
        print(f"  Gaussian sigma = {sigma:3}: RMSE {rmse(red, r1, development):5.2f}")
    for (it, kappa), red in anisotropic.items():
        print(f"  anisotropic, {it} iterations, kappa = {kappa:3d}: RMSE {rmse(red, r1, development):5.2f}")
    best_sigma = min(gaussian, key=lambda key: rmse(gaussian[key], r1, development))
    best_aniso = min(anisotropic, key=lambda key: rmse(anisotropic[key], r1, development))
    print(f"  -> Gaussian sigma = {best_sigma}, anisotropic diffusion with {best_aniso[0]} iterations and kappa = {best_aniso[1]}")

    # The methods with the chosen settings, on the test set
    tested = {"linear": linear,
              "knn": base,
              "knn_gaussian": gaussian[best_sigma],
              "knn_anisotropic": anisotropic[best_aniso]}
    tested["average"] = np.mean([tested["linear"], tested["knn_anisotropic"]], axis=0)
    print("The methods on the test set (never used for learning or for choosing settings):")
    for name, red in tested.items():
        print(f"  {name:16s}: RMSE {rmse(red, r1, test):5.2f}")
        image = to_bgr(b1, g1, np.where(test, red, r1))               # test squares predicted, the rest as in image 1
        for y in range(0, g1.shape[0], 42):
            for x in range(0, g1.shape[1], 42):
                if test[y, x]:
                    cv2.rectangle(image, (x, y), (x + 41, y + 41), (255, 255, 255), 1)
        save_image(out_dir, f"test_{name}.png", image)
    return best_k, best_sigma, best_aniso


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

    # Choose k and the noise reduction on image 1, where the red channel is known
    best_k, best_sigma, (iterations, kappa) = choose_settings(img1, out_dir)

    # The colour of a car does not depend on where it is in the image, so what is learned from image 1 can be used
    # on image 2 without aligning the images. Here the methods learn from all of image 1.
    reds = {}
    for method in (LINEAR, knn_method(best_k)):
        reds[method[0]] = predict_red(method, (b1, g1, r1), (b2, g2))
        save_image(out_dir, f"image2_R_{method[0]}.png", reds[method[0]].astype(np.uint8))
        save_image(out_dir, f"image2_corrected_{method[0]}.png", to_bgr(b2, g2, reds[method[0]]))

    # The red channel of the k-NN is noisy, so it gets an extra noise reduction step. Two filters are compared.
    denoisers = {"knn_anisotropic": lambda red: anisotropic_diffusion(red, iterations, kappa),
                 "knn_gaussian": lambda red: cv2.GaussianBlur(red.astype(np.float32), (0, 0), best_sigma)}
    for name, denoise in denoisers.items():
        reds[name] = denoise(reds["knn"])
        save_image(out_dir, f"image2_R_{name}.png", reds[name].astype(np.uint8))
        save_image(out_dir, f"image2_corrected_{name}.png", to_bgr(b2, g2, reds[name]))

    # Average of the red channels of the denoised k-NN (anisotropic) and linear regression
    reds["average"] = np.mean([reds["linear"], reds["knn_anisotropic"]], axis=0)
    save_image(out_dir, "image2_R_average.png", reds["average"].astype(np.uint8))
    save_image(out_dir, "image2_corrected_average.png", to_bgr(b2, g2, reds["average"]))


if __name__ == "__main__":
    main()
