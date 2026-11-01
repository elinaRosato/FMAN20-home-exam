import os
import numpy as np
import cv2

from task3 import save_image, blob_to_square, mismatch_penalty


def main():
    """Find the man's shadow WITHOUT using the man: the true silhouette should be the one closest to the average
    of the six silhouettes (the wrong ones are variations of it, each different in a different place)."""
    here = os.path.dirname(os.path.abspath(__file__))
    image_name = 'icecream_exam_2.jpg'      # or 'icecream_exam.jpg'
    out_dir = os.path.join(here, 'output', 'task3_average', os.path.splitext(image_name)[0])
    os.makedirs(out_dir, exist_ok=True)

    # load image
    img_bgr = cv2.imread(os.path.join(here, 'data', image_name))
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    save_image(out_dir, "00_original.png", img_bgr)

    # HSV threshold to isolate the purple shadows
    shadows_mask = cv2.inRange(img_hsv, np.array([105, 40, 50]), np.array([150, 150, 200]))
    save_image(out_dir, "shadows_1_mask.png", shadows_mask)

    # Find individual shadow blobs and keep the 6 largest
    num_labels, shadow_cc, shadow_stats, _ = cv2.connectedComponentsWithStats(shadows_mask, connectivity=8)
    all_blobs = [(i, shadow_stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    all_blobs.sort(key=lambda x: x[1], reverse=True)
    shadow_blobs = [label for label, area in all_blobs[:6]]
    shadows = np.isin(shadow_cc, shadow_blobs).astype(np.uint8) * 255
    save_image(out_dir, "shadows_2_kept.png", shadows)

    # Scale every shadow to the same 256 x 256 square (same function as in task3.py)
    squares = [blob_to_square((shadow_cc == label).astype(np.uint8) * 255) for label in shadow_blobs]

    # Average of the six shadows: 0 where none of them is filled, 1 where all of them are
    average = np.mean(squares, axis=0)
    save_image(out_dir, "average_1_gray.png", (average * 255).astype(np.uint8))

    # Typical shape: the pixels that are filled in at least half of the shadows
    typical = average >= 0.5
    save_image(out_dir, "average_2_typical_shape.png", typical.astype(np.uint8) * 255)

    # Compare every shadow with the typical shape (same mismatch score as in task3.py, lower = better)
    penalties = []
    for i, shadow_square in enumerate(squares):
        penalties.append(mismatch_penalty(typical, shadow_square))
        # Overlay: typical shape in red, shadow in green -> yellow where they agree
        overlay = np.zeros((*typical.shape, 3), np.uint8)
        overlay[..., 2] = typical * 255
        overlay[..., 1] = shadow_square * 255
        save_image(out_dir, f"shadow_{i}_overlay.png", overlay)

    print("Mismatch penalty of each shadow against the average shape (lower = better match):")
    for i in np.argsort(penalties):
        label = shadow_blobs[i]
        cx = shadow_stats[label, cv2.CC_STAT_LEFT] + shadow_stats[label, cv2.CC_STAT_WIDTH] / 2
        cy = shadow_stats[label, cv2.CC_STAT_TOP] + shadow_stats[label, cv2.CC_STAT_HEIGHT] / 2
        print(f"  shadow {i} ({cx / img_bgr.shape[1] * 100:.0f}% across, {cy / img_bgr.shape[0] * 100:.0f}% down): "
              f"{penalties[i]:.3f}")
    print(f"Matching shadow = shadow {int(np.argmin(penalties))}")


if __name__ == "__main__":
    main()
