import os
import numpy as np
import cv2
from scipy import ndimage

SCALE = 0.25            # the image is 5691 x 6847 pixels, all the work is done on a 4x smaller image


def save_image(out_dir, filename, img):
    """Save an image (BGR, or single-channel) into out_dir."""
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, img)
    print(f"saved {path}")


def ellipse_outline(ellipse, step=4):
    """Points on the outline of an ellipse ((cx, cy), (width, height), angle), one every `step` degrees."""
    (cx, cy), (d1, d2), angle = ellipse
    return cv2.ellipse2Poly((int(round(cx)), int(round(cy))), (int(round(d1 / 2)), int(round(d2 / 2))), int(round(angle)),
                            0, 360, step).astype(np.float32)


def is_crater_shape(ellipse):
    """What a crater looks like: a flat oval, three to seven times wider than high, lying almost horizontally, and of
    a certain size (the numbers are for the image at 1/4 size). Ellipses that do not look like this are not tried."""
    (cx, cy), (d1, d2), angle = ellipse
    major, minor = max(d1, d2), min(d1, d2)
    tilt = angle % 180 if d1 >= d2 else (angle + 90) % 180             # angle of the long axis, 0 = horizontal
    tilt = min(tilt, 180 - tilt)
    return 120 <= major <= 240 and 3 <= major / max(minor, 1) <= 7 and tilt <= 25


def ransac_ellipse(points, ink_distance, iterations=3000, tolerance=3.0, min_covered=0.6, seed=0):
    """RANSAC fit of an ellipse to the points of a dark ink blob, with the shape of a crater as a constraint.
    Five random points define a candidate ellipse (cv2.fitEllipse). Candidates that do not have the shape and size of a
    crater are thrown away. Of the rest, only the candidates where at least `min_covered` of the OUTLINE lies on ink
    (within `tolerance` pixels) are accepted, so an ellipse that is too long, and runs out along the line that leads to a
    stone, is rejected because the end of it has no ink. Among the accepted candidates, the one with the longest part of
    the outline on ink wins, which prefers the whole crater to a small oval inside it. `ink_distance` is the distance
    from every pixel to the nearest ink pixel. Returns (the covered part of the outline, between 0 and 1, ellipse)."""
    rng = np.random.default_rng(seed)
    best_count, best_fraction, best_ellipse = 0, 0.0, None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 5, replace=False)].astype(np.float32)
        try:
            ellipse = cv2.fitEllipse(sample)
        except cv2.error:
            continue
        if not is_crater_shape(ellipse):
            continue
        outline = np.round(ellipse_outline(ellipse)).astype(int)
        inside = (outline[:, 0] >= 0) & (outline[:, 0] < ink_distance.shape[1]) & (outline[:, 1] >= 0) & (outline[:, 1] < ink_distance.shape[0])
        if not inside.all():
            continue
        covered = ink_distance[outline[:, 1], outline[:, 0]] < tolerance
        if covered.mean() >= min_covered and covered.sum() > best_count:
            best_count, best_fraction, best_ellipse = int(covered.sum()), float(covered.mean()), ellipse
    if best_ellipse is not None:                          # refit to all the ink points that are close to the best ellipse
        outline = ellipse_outline(best_ellipse, step=2)
        near = np.min(np.linalg.norm(points[:, None, :] - outline[None, :, :], axis=2), axis=1) < tolerance
        if near.sum() >= 20:
            refined = cv2.fitEllipse(points[near].astype(np.float32))
            if is_crater_shape(refined):
                best_ellipse = refined
    return best_fraction, best_ellipse


def find_craters(img_bgr, out_dir):
    """Step 1: the craters. Every crater is an oval with a dark ink outline, so the craters are found by fitting ellipses
    to the dark ink. Returns a list of ellipses and a mask (255 = crater)."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ground = np.zeros((h, w), bool)
    ground[int(0.26 * h):] = True                         # the sky and the rockets are above this line
    ground[:, :int(0.03 * w)] = False                     # the dark edge of the page on the left

    # 1. The dark ink: the outlines of the craters (and of the arrows, the boot prints and the stones)
    ink = ((hsv[..., 2] < 130) & ground).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    save_image(out_dir, "01_dark_ink.png", ink * 255)

    # 2. The ink blobs that are big enough to be a crater outline. The arrows, the stones and the boot prints are small.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    candidates = [i for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] >= 60 and stats[i, cv2.CC_STAT_HEIGHT] >= 20
                  and stats[i, cv2.CC_STAT_AREA] < 8000]
    blobs = np.isin(labels, candidates).astype(np.uint8) * 255
    save_image(out_dir, "02_big_ink_blobs.png", blobs)

    # 3. Fit an ellipse to every blob with RANSAC
    fitted = img_bgr.copy()
    ellipses = []
    for i in candidates:
        ys, xs = np.nonzero(labels == i)
        points = np.stack([xs, ys], axis=1)
        if len(points) > 1500:                            # a random subset is enough, and faster
            points = points[np.random.default_rng(i).choice(len(points), 1500, replace=False)]
        blob = (labels == i).astype(np.uint8)
        ink_distance = cv2.distanceTransform(1 - blob, cv2.DIST_L2, 3)
        score, ellipse = ransac_ellipse(points, ink_distance)
        if ellipse is not None and score >= 0.5:
            ellipses.append(ellipse)
            cv2.ellipse(fitted, ellipse, (255, 0, 255), 3)
    save_image(out_dir, "03_fitted_ellipses.png", fitted)

    # 4. A margin, so that the whole crater with its rim is covered
    craters = np.zeros((h, w), np.uint8)
    overlay = img_bgr.copy()
    for (centre, (d1, d2), angle) in ellipses:
        enlarged = (centre, (d1 * 1.1 + 8, d2 * 1.1 + 8), angle)
        cv2.ellipse(craters, enlarged, 255, -1)
        cv2.ellipse(overlay, enlarged, (255, 0, 255), 3)
    save_image(out_dir, "04_craters.png", craters)
    save_image(out_dir, "05_craters_on_image.png", overlay)
    print(f"Found {len(ellipses)} craters")
    return ellipses, craters


def find_arrows(img_bgr, out_dir):
    """Step 2: the arrows. They are dark (grey-blue with a dark outline) on the yellow ground. Every arrow is a separate
    blob of dark pixels, and its direction comes from the blob: the long axis (PCA of the pixel positions) gives the
    line the arrow lies on, and the end with the arrowhead, which has more dark pixels than the other end, tells which
    way it points.
    Returns a list of arrows, each (tail, head, ratio), where the tail and the head are (x, y) points and the ratio is
    how many times more dark pixels the head end has than the tail end (close to 1 = the direction is uncertain)."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ground = np.zeros((h, w), bool)
    ground[int(0.26 * h):] = True
    ground[:, :int(0.03 * w)] = False

    # 1. The dark ink, with a threshold on the gray level. The closing fills the small holes, and the opening removes
    # the thin unwanted lines
    ink = ((gray < 130) & ground).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    save_image(out_dir, "06_dark_ink_mask.png", ink * 255)

    # 2. One blob per arrow. Blobs that are too small (specks) or too big (the text) are not arrows, and an arrow lies on
    # the yellow ground, so the ring just around a blob must be mostly yellow (this removes the aliens in the craters,
    # the rocket and the outlines of the boot prints and stones, which are surrounded by other colours)
    yellow = cv2.inRange(hsv, (25, 90, 200), (36, 255, 255)) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    arrows, blobs, kept_labels = [], img_bgr.copy(), []
    rng = np.random.default_rng(1)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not 100 <= area <= 1800:
            continue
        blob = (labels == i).astype(np.uint8)
        ring = (cv2.dilate(blob, np.ones((15, 15), np.uint8)) > 0) & (cv2.dilate(blob, np.ones((5, 5), np.uint8)) == 0)
        if yellow[ring].mean() < 0.7:
            continue
        ys, xs = np.nonzero(blob)
        points = np.stack([xs, ys], axis=1).astype(np.float64)
        centre = points.mean(axis=0)
        # long axis of the blob: the eigenvector of the covariance matrix with the biggest eigenvalue
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov((points - centre).T))
        axis = eigenvectors[:, 1]
        along = (points - centre) @ axis                  # position along the long axis
        length = along.max() - along.min()
        if length < 30 or np.sqrt(eigenvalues[1] / max(eigenvalues[0], 1e-6)) < 3:       # not long and thin
            continue
        # the arrowhead is the end with more pixels: count the pixels in the two end quarters of the blob
        tail_pixels = (along < along.min() + 0.25 * length).sum()
        head_pixels = (along > along.max() - 0.25 * length).sum()
        direction = axis if head_pixels > tail_pixels else -axis
        ratio = max(head_pixels, tail_pixels) / max(min(head_pixels, tail_pixels), 1)
        head = centre + direction * (length / 2)
        tail = centre - direction * (length / 2)
        arrows.append((tail, head, ratio))
        kept_labels.append(i)
        blobs[labels == i] = rng.integers(0, 255, 3)
    save_image(out_dir, "07_arrow_blobs_on_image.png", blobs)
    mask = np.isin(labels, kept_labels).astype(np.uint8) * 255
    save_image(out_dir, "08_arrow_mask.png", mask)

    # 3. The arrows drawn with their direction: a line from the tail (blue dot) to the head (red dot). The arrows where
    # the two ends are nearly equal (ratio below 1.15) are drawn in orange.
    drawn = img_bgr.copy()
    for tail, head, ratio in arrows:
        cv2.line(drawn, tuple(int(v) for v in tail), tuple(int(v) for v in head), (0, 160, 0) if ratio >= 1.15 else (0, 140, 255), 2)
        cv2.circle(drawn, tuple(int(v) for v in tail), 5, (255, 0, 0), -1)
        cv2.circle(drawn, tuple(int(v) for v in head), 5, (0, 0, 255), -1)
    save_image(out_dir, "09_arrow_directions.png", drawn)
    print(f"Found {len(arrows)} arrows, {sum(r < 1.15 for _, _, r in arrows)} of them with an uncertain direction (orange in the picture)")
    return arrows


def find_footprints(img_bgr, out_dir):
    """Step 3: the boot prints. A boot print is an orange sole and a smaller orange heel, with a line of ink between
    them and an ink outline round both. They are redder than the craters, so the orange pixels with a hue below 19 are
    taken. Then the sole and the heel are joined with a closing (which also fills the ink line between them), the holes
    are filled, and only the blobs with the size and shape of a boot print are kept (about 2.5 times longer than wide).
    Returns a mask (255 = boot print) and the list of boot prints as ((cx, cy), (width, height), angle)."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ground = np.zeros((h, w), bool)
    ground[int(0.26 * h):] = True
    ground[:, :int(0.03 * w)] = False
    yellow = cv2.inRange(hsv, (25, 90, 200), (36, 255, 255)) > 0
    orange = (cv2.inRange(hsv, (3, 85, 150), (18, 255, 255)) > 0) & ground
    save_image(out_dir, "10_orange_pixels.png", orange.astype(np.uint8) * 255)
    joined = cv2.morphologyEx(orange.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    joined = ndimage.binary_fill_holes(joined > 0).astype(np.uint8)
    joined = cv2.morphologyEx(joined, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    footprints = np.zeros((h, w), np.uint8)
    boots = []
    for i in range(1, n):
        if not 800 <= stats[i, cv2.CC_STAT_AREA] <= 6000:
            continue
        if np.median(hsv[..., 0][(labels == i) & orange]) > 16:              # the craters are yellower than the boot prints
            continue
        blob = (labels == i).astype(np.uint8)
        ring = (cv2.dilate(blob, np.ones((15, 15), np.uint8)) > 0) & (cv2.dilate(blob, np.ones((5, 5), np.uint8)) == 0)
        if yellow[ring].mean() < 0.5:                                         # a boot print lies on the yellow ground (not the rocket door)
            continue
        contours, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        major, minor = max(rect[1]), max(min(rect[1]), 1)
        if 48 <= major <= 110 and 1.5 <= major / minor <= 4:
            footprints[labels == i] = 255
            boots.append(rect)
    save_image(out_dir, "11_footprints.png", footprints)
    overlay = img_bgr.copy()
    for rect in boots:
        cv2.drawContours(overlay, [np.int32(cv2.boxPoints(rect))], -1, (0, 0, 255), 3)
    save_image(out_dir, "12_footprints_on_image.png", overlay)
    print(f"Found {len(boots)} boot prints")
    return footprints


def find_start_and_goal(img_bgr, out_dir):
    """Step 5: the START and the goal (MÅL) icons. Both are small, tall, red marks on the ground. The other red things
    (the rocket, the pink alien) are much bigger or wider than tall. The lower one is START and the other is MÅL.
    Returns the two (x, y) centres."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] >= 165) | (hsv[..., 0] <= 3)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 90)).astype(np.uint8)
    red[:int(0.26 * h)] = 0
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    save_image(out_dir, "13_red_pixels.png", red * 255)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(red, connectivity=8)
    icons = [i for i in range(1, n) if 600 <= stats[i, cv2.CC_STAT_AREA] <= 1500
             and stats[i, cv2.CC_STAT_HEIGHT] > 1.5 * stats[i, cv2.CC_STAT_WIDTH]]
    icons.sort(key=lambda i: -centroids[i][1])             # the lowest first
    assert len(icons) == 2, f"expected two icons, found {len(icons)}"
    start, goal = centroids[icons[0]], centroids[icons[1]]
    overlay = img_bgr.copy()
    for point, name in ((start, "START"), (goal, "MAL")):
        cv2.circle(overlay, (int(point[0]), int(point[1])), 40, (255, 0, 255), 4)
        cv2.putText(overlay, name, (int(point[0]) + 45, int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    save_image(out_dir, "14_start_and_goal.png", overlay)
    print(f"START at {start.astype(int)}, goal at {goal.astype(int)}")
    return start, goal


def find_nodes(img_bgr, arrows, footprints, craters, start, goal, out_dir, end_radius=28, boot_growth=30):
    """Step 6: the nodes of the maze. The arrows do not touch the boot prints or each other: there is a gap of 25 to 45
    pixels. So every boot print is grown (dilated) by `boot_growth` pixels, and a disc of radius `end_radius` is drawn
    round every arrow end (head or tail). Areas that touch belong together, so the candidates for nodes are the connected
    components of the union of all the grown areas.
    A candidate is a node only if all of this is true:
      - it has at least one arrow head and at least one arrow tail (a place where arrows arrive AND where arrows leave; a
        candidate with only heads, only tails, only heads and a boot print or only tails and a boot print is not a node),
      - it does not touch a crater, unless it contains a boot print (a grown area round a boot print can reach the margin of a
        crater next to it, and the boot print itself is not in the crater).
    The exceptions are the START node and the goal node (the candidates next to the START and the goal icons), which are
    always nodes. The other candidates are not removed from the picture, they are just not counted as nodes.
    Returns the label image with only the nodes in it (0 = not a node), the START node and the goal node."""
    h, w = img_bgr.shape[:2]
    grown_boots = cv2.dilate(footprints, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * boot_growth + 1, 2 * boot_growth + 1)))
    tail_discs = np.zeros((h, w), np.uint8)
    head_discs = np.zeros((h, w), np.uint8)
    for tail, head, _ in arrows:
        cv2.circle(tail_discs, (int(round(tail[0])), int(round(tail[1]))), end_radius, 255, -1)
        cv2.circle(head_discs, (int(round(head[0])), int(round(head[1]))), end_radius, 255, -1)

    overlay = img_bgr.copy()
    for mask, colour in ((grown_boots, (0, 0, 255)), (tail_discs, (255, 0, 0)), (head_discs, (0, 200, 0))):
        overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array(colour)).astype(np.uint8)
    for tail, head, _ in arrows:
        cv2.line(overlay, tuple(int(round(v)) for v in tail), tuple(int(round(v)) for v in head), (0, 0, 0), 2)
    for point in (start, goal):
        cv2.circle(overlay, (int(point[0]), int(point[1])), 40, (255, 0, 255), 4)
    save_image(out_dir, "15_grown_areas.png", overlay)

    union = cv2.bitwise_or(cv2.bitwise_or(grown_boots, tail_discs), head_discs)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(union, connectivity=8)
    at = lambda p, i: labels[min(max(int(round(p[1])), 0), h - 1), min(max(int(round(p[0])), 0), w - 1)] == i

    # START and goal nodes: the candidates nearest to the icons
    distance, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    nearest = lambda p: int(labels[iy[int(p[1]), int(p[0])], ix[int(p[1]), int(p[0])]])
    start_candidate, goal_candidate = nearest(start), nearest(goal)

    valid, reasons = {}, {}
    for i in range(1, n):
        heads = sum(at(hd, i) for _, hd, _ in arrows)
        tails = sum(at(t, i) for t, _, _ in arrows)
        has_boot = bool((footprints[labels == i] > 0).any())
        crater = bool((craters[labels == i] > 0).any()) and not has_boot
        ok = heads > 0 and tails > 0 and not crater
        if i in (start_candidate, goal_candidate):
            ok = True
        valid[i] = ok
        reasons[i] = "node" if ok else ("touches a crater" if crater else f"only heads ({heads})" if tails == 0 and heads > 0
                                         else f"only tails ({tails})" if heads == 0 else "?")
    node_ids = [i for i in range(1, n) if valid[i]]
    numbering = {old: new for new, old in enumerate(node_ids, start=1)}
    nodes = np.zeros((h, w), np.int32)
    for old, new in numbering.items():
        nodes[labels == old] = new

    # nodes: all the same green; candidates that are not nodes: red diagonal stripes
    picture = img_bgr.copy()
    yy, xx = np.mgrid[0:h, 0:w]
    stripes = ((xx + yy) // 8) % 2 == 0
    for i in range(1, n):
        area = labels == i
        if valid[i]:
            picture[area] = (0.5 * picture[area] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
        else:
            area = area & stripes
            picture[area] = (0.4 * picture[area] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        outline = (0, 120, 0) if valid[i] else (0, 0, 200)
        contours, _ = cv2.findContours(((labels == i) * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(picture, contours, -1, outline, 2)
    for i in node_ids:
        cv2.putText(picture, str(numbering[i]), (int(centroids[i][0]) - 10, int(centroids[i][1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    save_image(out_dir, "16_nodes.png", picture)

    counts = {}
    for i in range(1, n):
        counts[reasons[i]] = counts.get(reasons[i], 0) + 1
    print(f"{n - 1} candidate areas, {len(node_ids)} of them are nodes (end radius {end_radius}, boot growth {boot_growth})")
    print("  not nodes:", {k: v for k, v in counts.items() if k != 'node' and not k.startswith('only')},
          "| only heads:", sum(v for k, v in counts.items() if k.startswith('only heads')),
          "| only tails:", sum(v for k, v in counts.items() if k.startswith('only tails')))
    return nodes, numbering[start_candidate], numbering[goal_candidate]


def solve_maze(img_bgr, arrows, nodes, start_node, goal_node, start, goal, out_dir):
    """Step 7: the graph and the path. Every arrow is a directed edge from the node of its tail to the node of its head. An
    arrow with an end that is not in a node (see find_nodes) is not an edge. The shortest path from the START node to the
    goal node is found with breadth-first search."""
    h, w = img_bgr.shape[:2]
    node_of = lambda p: int(nodes[min(max(int(round(p[1])), 0), h - 1), min(max(int(round(p[0])), 0), w - 1)])
    edges = {}
    for k, (tail, head, _) in enumerate(arrows):
        a, b = node_of(tail), node_of(head)
        if a > 0 and b > 0 and a != b:
            edges.setdefault((a, b), k)
    centres = {}
    for i in range(1, int(nodes.max()) + 1):
        ys, xs = np.nonzero(nodes == i)
        centres[i] = (int(xs.mean()), int(ys.mean()))

    graph_picture = img_bgr.copy()
    for (a, b) in edges:
        cv2.arrowedLine(graph_picture, centres[a], centres[b], (0, 0, 0), 2, tipLength=0.15)
    for i, c in centres.items():
        cv2.circle(graph_picture, c, 9, (255, 0, 255) if i in (start_node, goal_node) else (0, 160, 0), -1)
        cv2.putText(graph_picture, str(i), (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    save_image(out_dir, "17_graph.png", graph_picture)
    print(f"The graph has {len(centres)} nodes and {len(edges)} edges")

    graph = {}
    for (a, b) in edges:
        graph.setdefault(a, []).append(b)
    previous = {start_node: None}
    queue = [start_node]
    for node in queue:
        for nxt in graph.get(node, []):
            if nxt not in previous:
                previous[nxt] = node
                queue.append(nxt)
    if goal_node not in previous:
        print(f"No path from START (node {start_node}) to the goal (node {goal_node}); {len(previous)} nodes can be reached from START")
        return None, edges
    path = [goal_node]
    while previous[path[-1]] is not None:
        path.append(previous[path[-1]])
    path.reverse()
    print("Path (node numbers):", path, f"({len(path) - 1} steps)")
    picture = img_bgr.copy()
    for a, b in zip(path[:-1], path[1:]):
        cv2.arrowedLine(picture, centres[a], centres[b], (0, 0, 255), 10, tipLength=0.12)
    for point in (start, goal):
        cv2.circle(picture, (int(point[0]), int(point[1])), 40, (255, 0, 255), 6)
    save_image(out_dir, "18_path.png", picture)
    return path, edges


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, 'output', 'task6')
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(os.path.join(here, 'data', 'moon_exam.jpg'))
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    save_image(out_dir, "00_image.png", img)

    ellipses, craters = find_craters(img, out_dir)
    arrows = find_arrows(img, out_dir)
    footprints = find_footprints(img, out_dir)
    start, goal = find_start_and_goal(img, out_dir)
    nodes, start_node, goal_node = find_nodes(img, arrows, footprints, craters, start, goal, out_dir)
    path, edges = solve_maze(img, arrows, nodes, start_node, goal_node, start, goal, out_dir)


if __name__ == "__main__":
    main()
