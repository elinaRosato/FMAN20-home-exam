# FMAN20 Image Analysis – Home exam

Elina Rosato. The full solution, with figures and explanations, is in **`report/report.pdf`** (source: `report/report.tex`).

## Tasks

| Task | What it asks | Method | Where to find it |
|------|--------------|--------|------------------|
| 1 | Find the one toucan (of 8) with a different eyelid in `birds_exam.jpg` | Remove the background with HSV thresholds and morphology, find the bird blobs, compare SIFT descriptors between the birds. The bird with the worst match is the odd one (bird F). | `task1.py`, report section 1 |
| 2 | Histogram equalization: threshold level and transformations for given distributions | Solved by hand with the cumulative distribution (threshold $2^{-1/3}\approx 0.794$ for part a). | Report section 2 only (no code) |
| 3 | Find the shadow that belongs to the man in `icecream_exam.jpg` | Threshold the purple shadows and the background, scale each silhouette to a square and compare with a distance-based mismatch score (silhouette E). | `task3.py`, report section 3 |
| 3 (extra) | Same, but without using the man | Compare each shadow with the average of the six shadows. | `task3_average.py`, report section 3 |
| 4 | Parameters and receptive field of a small CNN | Counted by hand: 7365 parameters, receptive field 18x18. | Report section 4 only (no code) |
| 5 | Reconstruct the missing red channel of `cars_sm_2.png` from `cars_sm_1.png` | Regression from (G, B) to R: linear regression, RANSAC and k-NN (with Gaussian or anisotropic-diffusion denoising), compared with k-fold cross-validation and a test set; the final result averages linear regression and the denoised k-NN. | `task5.py`, report section 5 |
| 6 | Find the way from START to the rocket in `moon_exam.jpg`, following the arrows and avoiding the craters | Craters by RANSAC ellipse fitting, arrows by blobs and PCA, footprints by colour, nodes from the grown footprint and arrow-end areas, then a directed graph and breadth-first search (25 steps). | `task6.py`, report section 6 |

## Running the code

The scripts read the images from `data/` and save every intermediate step as an image in `output/taskN/` (this folder is not in the repository, it is created when the scripts are run). They need Python 3 and the packages in `requirements.txt` (`pip install -r requirements.txt`).

```
python3 task1.py
python3 task3.py
python3 task3_average.py
python3 task5.py
python3 task6.py
```

The figures used in the report are copies of the images from `output/` and are stored in `report/figures/`.
