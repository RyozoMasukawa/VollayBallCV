"""
train_svm.py
------------
Builds a synthetic training dataset that mimics the joint angles observed
during a volleyball spike approach/arm-swing, then fits a Support Vector
Machine (SVC) classifier that labels a pose as:

    1 -> "Good Spike Form"   (elbow fully extended, knee loaded/bent)
    0 -> "Needs Adjustment"  (everything else: locked knee, dropped elbow, etc.)

The resulting model is serialized to 'app_model.joblib' with joblib so the
FastAPI server (main.py) can load it instantly at startup without needing
to retrain. This script is executed once during the Docker image build
(see Dockerfile), so the container always ships with a ready-to-use model.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Fix the random seed so the synthetic dataset (and therefore the trained
# model) is reproducible across builds.
RNG = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# 1. Generate a realistic synthetic dataset
# ---------------------------------------------------------------------------
# We generate two clusters of (elbow_angle, knee_angle) pairs:
#
#   GOOD FORM (label 1):
#       - Right elbow is nearly fully extended during ball contact
#         (~150-180 degrees).
#       - Right knee is loaded/bent for explosive push-off
#         (~100-135 degrees).
#
#   NEEDS ADJUSTMENT (label 0):
#       - Any other combination: a collapsed/bent elbow (poor arm swing),
#         a locked-straight knee (no load for the jump), or generally
#         awkward angle combinations that do not fall in the "good" range.
#
# We generate 100 samples per class (200 total) with Gaussian noise so the
# decision boundary is smooth and realistic rather than a hard rectangle.

N_PER_CLASS = 100

# --- Class 1: Good Spike Form ---
good_elbow = RNG.normal(loc=165, scale=7, size=N_PER_CLASS)   # centered ~165 deg
good_knee = RNG.normal(loc=117, scale=8, size=N_PER_CLASS)    # centered ~117 deg
good_elbow = np.clip(good_elbow, 145, 180)
good_knee = np.clip(good_knee, 95, 140)
good_labels = np.ones(N_PER_CLASS, dtype=int)

# --- Class 0: Needs Adjustment ---
# We mix three "bad" sub-patterns so the class 0 cloud is spread out and
# realistic, rather than a single unrealistic blob:
#   a) Dropped/bent elbow with a reasonable knee bend (poor arm extension)
#   b) Good elbow extension but a locked-straight knee (no jump power)
#   c) Both angles awkward / mid-range (generally sloppy form)
n_a = N_PER_CLASS // 3
n_b = N_PER_CLASS // 3
n_c = N_PER_CLASS - n_a - n_b

bad_elbow_a = RNG.normal(loc=100, scale=15, size=n_a)   # bent elbow
bad_knee_a = RNG.normal(loc=115, scale=10, size=n_a)

bad_elbow_b = RNG.normal(loc=168, scale=6, size=n_b)    # good elbow
bad_knee_b = RNG.normal(loc=165, scale=8, size=n_b)     # locked knee

bad_elbow_c = RNG.normal(loc=130, scale=12, size=n_c)   # awkward mid-range
bad_knee_c = RNG.normal(loc=145, scale=12, size=n_c)

bad_elbow = np.concatenate([bad_elbow_a, bad_elbow_b, bad_elbow_c])
bad_knee = np.concatenate([bad_knee_a, bad_knee_b, bad_knee_c])
bad_elbow = np.clip(bad_elbow, 20, 180)
bad_knee = np.clip(bad_knee, 20, 180)
bad_labels = np.zeros(N_PER_CLASS, dtype=int)

# --- Combine into a single dataset ---
X = np.column_stack([
    np.concatenate([good_elbow, bad_elbow]),
    np.concatenate([good_knee, bad_knee]),
])
y = np.concatenate([good_labels, bad_labels])

# Shuffle the dataset so the classes are interleaved (not strictly required
# for SVM training, but good practice before a train/test split).
shuffle_idx = RNG.permutation(len(y))
X = X[shuffle_idx]
y = y[shuffle_idx]

print(f"Generated synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")

# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Train the SVM classifier
# ---------------------------------------------------------------------------
# An RBF kernel handles the smooth, non-linear boundary between the "good"
# cluster and the surrounding "needs adjustment" cloud well. probability=True
# lets us optionally expose a confidence score from the API in the future.
model = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")

# ---------------------------------------------------------------------------
# 5. Serialize the trained model to disk
# ---------------------------------------------------------------------------
MODEL_PATH = "app_model.joblib"
joblib.dump(model, MODEL_PATH)
print(f"Model saved to '{MODEL_PATH}'")
