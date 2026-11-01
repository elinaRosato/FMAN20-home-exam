import os
import numpy as np
import cv2


def save_image(out_dir, filename, img):
    """Save an image (BGR, or single-channel mask) at full resolution into out_dir."""
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, img)
    print(f"saved {path}")


def hsv_threshold_regions(img_bgr, img_hsv, lower_threshold, upper_threshold, min_area, name, out_dir):
    """Mask of the large regions whose HSV colour is inside [lower, upper] (255 = region, 0 = rest)."""
    mask = cv2.inRange(img_hsv, lower_threshold, upper_threshold)
    save_image(out_dir, f"{name}_1_mask.png", mask)

    # Connected components of the mask
    num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Color each connected component differently (label 0 stays black)
    rng = np.random.default_rng(0)
    colors = rng.integers(50, 256, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = 0
    save_image(out_dir, f"{name}_2_components.png", colors[cc_labels])

    # Label 0 is the part outside the mask, we skip it
    # Keep every component whose area is above a threshold
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = np.where(areas >= min_area)[0]
    keep = keep[keep != 0]
    regions = np.isin(cc_labels, keep).astype(np.uint8) * 255
    save_image(out_dir, f"{name}_3_regions.png", regions)
    save_image(out_dir, f"{name}_4_regions_on_image.png", img_bgr * (regions[:, :, None] > 0))

    return regions


def blob_to_square(mask, size=256):
    """Crop a blob to its bounding box, scale the longest side to `size` 
    and pad with zeros to a size x size square. """
    ys, xs = np.where(mask > 0)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = size / max(h, w)
    crop = cv2.resize(crop, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_NEAREST)
    square = np.zeros((size, size), np.uint8)
    y0 = (size - crop.shape[0]) // 2
    x0 = (size - crop.shape[1]) // 2
    square[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return square > 0


def distance_to(mask):
    """For every pixel, the distance to the nearest pixel of the (boolean) mask."""
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def mismatch_penalty(man_square, shadow_square):
    """How badly the two shapes disagree. Every pixel that belongs to only one of the shapes costs the SQUARE of
    its distance to the other shape. A pixel just outside the other shape (a thin fringe from a small misalignment)
    costs almost nothing, a pixel far away costs a lot. Divided by the area of the man. Lower = better match."""
    shadow_extra = shadow_square & ~man_square        # shadow pixels outside the man
    man_extra = man_square & ~shadow_square           # man pixels outside the shadow
    cost = (distance_to(man_square)[shadow_extra] ** 2).sum() + (distance_to(shadow_square)[man_extra] ** 2).sum()
    return cost / man_square.sum()


def find_matching_shadow(man_mask, shadow_cc, shadow_blobs, out_dir):
    """Mismatch penalty of each shadow against the man's silhouette. The lowest penalty is the matching shadow."""
    man_square = blob_to_square(man_mask)
    save_image(out_dir, "man_square.png", man_square.astype(np.uint8) * 255)

    penalties = []
    for i, label in enumerate(shadow_blobs):
        shadow_square = blob_to_square((shadow_cc == label).astype(np.uint8) * 255)
        penalties.append(mismatch_penalty(man_square, shadow_square))
        # Overlay: man in red, shadow in green -> yellow where they agree
        overlay = np.zeros((*man_square.shape, 3), np.uint8)
        overlay[..., 2] = man_square * 255
        overlay[..., 1] = shadow_square * 255
        save_image(out_dir, f"shadow_{i}_overlay.png", overlay)

    return penalties


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    image_name = 'icecream_exam.jpg'      # 'icecream_exam_2.jpg' was also tried, see the report
    out_dir = os.path.join(here, 'output', 'task3', os.path.splitext(image_name)[0])
    os.makedirs(out_dir, exist_ok=True)

    # load image
    img_bgr = cv2.imread(os.path.join(here, 'data', image_name))
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    save_image(out_dir, "00_original.png", img_bgr)

    # HSV threshold to isolate the purple shadows
    shadows_mask = cv2.inRange(img_hsv, np.array([105, 40, 50]), np.array([150, 150, 200]))
    save_image(out_dir, "shadows_1_mask.png", shadows_mask)

    # Find individual shadow blobs
    num_shadow_labels, shadow_cc, shadow_stats, _ = cv2.connectedComponentsWithStats(shadows_mask, connectivity=8)

    # Sort blobs by area descending, keep top 6
    all_blobs = [(i, shadow_stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_shadow_labels)]
    all_blobs.sort(key=lambda x: x[1], reverse=True)
    shadow_blobs = [label for label, area in all_blobs[:6]]

    print("Keeping top 6 blobs:")
    for label, area in all_blobs[:6]:
        print(f"  Label {label}: area = {area}")

    # Shadows: only the kept blobs, everything else is ignored
    shadows = np.isin(shadow_cc, shadow_blobs).astype(np.uint8) * 255
    save_image(out_dir, "shadows.png", shadows)

    # The man
    background_mask = hsv_threshold_regions(img_bgr, img_hsv, np.array([15, 10, 140]), np.array([75, 95, 230]), 100000, "background", out_dir)

    # Everything that is not background: the man, the silhouettes, the letters, the title, the corner icons
    not_background = cv2.bitwise_not(background_mask)
    save_image(out_dir, "man_1_not_background.png", not_background)

    # The man is the largest remaining blob
    num_man_labels, man_cc, man_stats, _ = cv2.connectedComponentsWithStats(not_background, connectivity=8)
    man_label = 1 + np.argmax(man_stats[1:, cv2.CC_STAT_AREA])
    man_mask = (man_cc == man_label).astype(np.uint8) * 255
    save_image(out_dir, "man_3_silhouette.png", man_mask)
    save_image(out_dir, "man_without_bg.png", img_bgr * (man_mask[:, :, None] > 0))

    # Find the matching shadow
    penalties = find_matching_shadow(man_mask, shadow_cc, shadow_blobs, out_dir)

    print("Mismatch penalty of each shadow against the man (lower = better match):")
    for i in np.argsort(penalties):
        label = shadow_blobs[i]
        cx, cy = shadow_stats[label, cv2.CC_STAT_LEFT] + shadow_stats[label, cv2.CC_STAT_WIDTH] / 2, \
                 shadow_stats[label, cv2.CC_STAT_TOP] + shadow_stats[label, cv2.CC_STAT_HEIGHT] / 2
        print(f"  shadow {i} ({cx / img_bgr.shape[1] * 100:.0f}% across, {cy / img_bgr.shape[0] * 100:.0f}% down): "
              f"{penalties[i]:.3f}")
    print(f"Matching shadow = shadow {int(np.argmin(penalties))}")


if __name__ == "__main__":
    main()
