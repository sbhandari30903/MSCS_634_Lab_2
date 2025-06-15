# MSCS_634_Lab_2
# Classification Using KNN and RNN Algorithms
# Wine Classification Lab

## Purpose

This lab investigates the performance of two proximity-based classifiers—K-Nearest Neighbors (KNN) and Radius Neighbors (RNN)—on the Wine Dataset from `sklearn`. By systematically varying the number of neighbors (`k`) and the search radius, the goal is to understand how these parameters influence classification accuracy and to identify optimal settings for each method.

## Key Insights

### KNN Performance
- Accuracy improved from **~0.778** (k=1) to **~0.806** for k ≥ 5 and remained stable up to k=21.
- A very small `k` leads to overfitting, while a moderate value (around 5) provides the best trade-off between bias and variance.

### RNN Performance
- Highest accuracy (**~0.722**) occurred at the smallest tested radius (350).
- Accuracy declined steadily with increasing radius, reaching **~0.667** for radii ≥550.
- A larger radius includes more distant, less relevant points, which can degrade classification.

### Model Comparison
- Overall, KNN outperformed RNN on this dataset (peak **80.6%** vs. **72.2%**).
- KNN’s fixed-`k` approach is more robust when neighborhood density varies, while RNN requires careful radius tuning to avoid empty or overly broad neighborhoods.

## Challenges & Decisions

- **Parameter Selection:**
  - Choosing an appropriate range of radius values was non-trivial because feature scales span different magnitudes. Initial small radii risked empty neighbor sets, so larger values (350–600) ensured sufficient neighbors.
  - For KNN, we included both small and larger `k` values to illustrate the bias–variance trade-off.

- **Data Handling:**
  - No feature scaling was applied since all features share comparable scales; however, in other datasets, normalization would be essential.
  - The train/test split used `stratify=y` to maintain original class distribution.

- **Implementation Choices:**
  - Default Euclidean distance metric was used for both classifiers.
  - Matplotlib was chosen for visualization, with grid lines and markers added for readability.


