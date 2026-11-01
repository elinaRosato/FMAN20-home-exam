import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np

def save_image(out_dir, filename, img):
    """Save an image (BGR, or single-channel mask) at full resolution into out_dir."""
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, img)
    print(f"saved {path}")


def hsv_isolate_bg(img_bgr, img_hsv, lower_threshold, upper_threshold, min_area, name, out_dir):
    # HSV threshold to isolate background
    mask = cv2.inRange(img_hsv, lower_threshold, upper_threshold)
    save_image(out_dir, f"{name}_1_mask.png", mask)

    # Clean up small holes with morphological closing
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    save_image(out_dir, f"{name}_2_mask_closed.png", mask_closed)

    # Connected components of the closed mask
    num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)

    # Color each connected component differently (label 0 stays black)
    rng = np.random.default_rng(0)
    colors = rng.integers(50, 256, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = 0
    cc_color = colors[cc_labels]
    save_image(out_dir, f"{name}_3_components.png", cc_color)

    # Label 0 is non-background areas, we skip it
    # Keep every component whose area is above a threshold
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = np.where(areas >= min_area)[0]
    keep = keep[keep != 0]
    background_mask = np.isin(cc_labels, keep).astype(np.uint8) * 255
    save_image(out_dir, f"{name}_4_background_mask.png", background_mask)

    bird_mask = cv2.bitwise_not(background_mask)
    save_image(out_dir, f"{name}_5_bird_mask.png", bird_mask)

    birds_only = img_bgr * (bird_mask[:, :, None] > 0)
    save_image(out_dir, f"{name}_6_birds_only.png", birds_only)

    # Overview figure (saved and shown)
    # fig = plt.figure(figsize=(24, 5))
    # plt.subplot(1, 6, 1); plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); plt.title("Original")
    # plt.subplot(1, 6, 2); plt.imshow(mask, cmap='gray'); plt.title("Mask (before closing)")
    # plt.subplot(1, 6, 3); plt.imshow(mask_closed, cmap='gray'); plt.title("Mask (after closing)")
    # plt.subplot(1, 6, 4); plt.imshow(cc_color); plt.title(f"Connected components ({num_labels - 1})")
    # plt.subplot(1, 6, 5); plt.imshow(bird_mask, cmap='gray'); plt.title("Bird mask")
    # plt.subplot(1, 6, 6); plt.imshow(cv2.cvtColor(birds_only, cv2.COLOR_BGR2RGB)); plt.title("Birds only")
    # fig.savefig(os.path.join(out_dir, f"{name}_overview.png"), dpi=150, bbox_inches='tight')
    # plt.show()

    return bird_mask


def crop_bird(img_bgr, bird_cc, bird_stats, blob_label, max_side=800):
    """Crop one bird (background set to black) and shrink it so its longest side is max_side."""
    x = bird_stats[blob_label, cv2.CC_STAT_LEFT]
    y = bird_stats[blob_label, cv2.CC_STAT_TOP]
    w = bird_stats[blob_label, cv2.CC_STAT_WIDTH]
    h = bird_stats[blob_label, cv2.CC_STAT_HEIGHT]
    mask = (bird_cc[y:y+h, x:x+w] == blob_label).astype(np.uint8) * 255
    crop = img_bgr[y:y+h, x:x+w] * (mask[:, :, None] > 0)
    s = max_side / max(w, h)
    size = (int(w * s), int(h * s))
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA), cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def find_odd_bird(img_bgr, bird_cc, bird_stats, bird_blobs, out_dir):
    """Odd bird without alignment: SIFT descriptors are scale/rotation invariant, so just match them directly."""
    crops = [crop_bird(img_bgr, bird_cc, bird_stats, b) for b in bird_blobs]
    N = len(crops)

    sift = cv2.SIFT_create()
    all_des = []
    for i, (crop, mask) in enumerate(crops):
        # Shrink the mask so no keypoints sit on the artificial cut-out outline
        inner_mask = cv2.erode(mask, np.ones((31, 31), np.uint8))
        save_image(out_dir, f"bird_{i}_1_eroded.png", crop * (inner_mask[:, :, None] > 0))

        # SIFT works on grey values, so this is the image it actually sees
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        save_image(out_dir, f"bird_{i}_2_gray.png", gray * (inner_mask > 0))

        kp, des = sift.detectAndCompute(gray, inner_mask)
        all_des.append(des)
        save_image(out_dir, f"bird_{i}_3_keypoints.png",
                   cv2.drawKeypoints(crop, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

    matcher = cv2.BFMatcher()
    scores = []
    for i in range(N):
        # Match bird i's descriptors against all the other birds pooled together
        other_des = np.vstack([all_des[j] for j in range(N) if j != i])
        distances = sorted(m.distance for m in matcher.match(all_des[i], other_des))
        # Worst 3% of matches: the unique local features (like the half-closed eye) that the others lack
        k = max(1, int(0.03 * len(distances)))
        scores.append(np.mean(distances[-k:]))

    print("descriptor distance to the other birds (higher = more different):")
    for i in np.argsort(scores)[::-1]:
        label = bird_blobs[i]
        cx = (bird_stats[label, cv2.CC_STAT_LEFT] + bird_stats[label, cv2.CC_STAT_WIDTH] / 2) / img_bgr.shape[1]
        cy = (bird_stats[label, cv2.CC_STAT_TOP] + bird_stats[label, cv2.CC_STAT_HEIGHT] / 2) / img_bgr.shape[0]
        print(f"  bird {i} ({cx*100:.0f}% across, {cy*100:.0f}% down): {scores[i]:.1f}")
    print(f"Odd bird = bird {int(np.argmax(scores))}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, 'output', 'task1')
    os.makedirs(out_dir, exist_ok=True)

    # load image
    img_bgr = cv2.imread(os.path.join(here, 'data', 'birds_exam.jpg'))
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) 

    # HSV threshold to isolate green background
    bird_mask_green = hsv_isolate_bg(img_bgr, img_hsv, np.array([35, 105, 90]), np.array([43, 210, 255]), 6000, "green", out_dir)
    # HSV threshold to isolate light blue sky
    bird_mask_blue = hsv_isolate_bg(img_bgr, img_hsv, np.array([85, 25, 150]), np.array([115, 100, 255]), 1000, "blue", out_dir)

    # Combine: a pixel is a bird only if neither the green nor the blue step removed it
    bird_mask = cv2.bitwise_and(bird_mask_green, bird_mask_blue)
    save_image(out_dir, "final_bird_mask.png", bird_mask)

    # Remove thin leaf outlines that join birds together
    open_kernel = np.ones((15, 15), np.uint8)
    bird_mask_opened = cv2.morphologyEx(bird_mask, cv2.MORPH_OPEN, open_kernel)
    save_image(out_dir, "final_bird_mask_opened.png", bird_mask_opened)

    # Find individual bird blobs
    num_bird_labels, bird_cc, bird_stats, _ = cv2.connectedComponentsWithStats(bird_mask_opened, connectivity=8)

    # Sort blobs by area descending, keep top 8
    all_blobs = [(i, bird_stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_bird_labels)]
    all_blobs.sort(key=lambda x: x[1], reverse=True)
    bird_blobs = [label for label, area in all_blobs[:8]]

    print(f"Keeping top 8 blobs:")
    for label, area in all_blobs[:8]:
        print(f"  Label {label}: area = {area}")

    # Final mask: only the kept blobs, everything else is ignored
    final_mask = np.isin(bird_cc, bird_blobs).astype(np.uint8) * 255
    save_image(out_dir, "final_mask_kept_blobs.png", final_mask)

    # Apply the final mask to the original image
    final_birds = img_bgr * (final_mask[:, :, None] > 0)
    save_image(out_dir, "final_birds_masked.png", final_birds)

    # Visualize (only kept blobs colored, rest black)
    rng2 = np.random.default_rng(42)
    bird_colors = rng2.integers(50, 256, size=(num_bird_labels, 3), dtype=np.uint8)
    bird_colors[0] = 0
    bird_cc_color = bird_colors[bird_cc] * (final_mask[:, :, None] > 0)

    save_image(out_dir, "final_bird_blobs.png", cv2.cvtColor(bird_cc_color, cv2.COLOR_RGB2BGR))
    plt.imshow(bird_cc_color)
    plt.title(f"Bird blobs after area filter ({len(bird_blobs)} kept)")
    plt.show()

    #  Find the odd bird: align every bird to a reference with SIFT features, then compare
    odd_dir = os.path.join(out_dir, 'odd_bird')
    os.makedirs(odd_dir, exist_ok=True)
    find_odd_bird(img_bgr, bird_cc, bird_stats, bird_blobs, odd_dir)


if __name__ == "__main__":
    main()
