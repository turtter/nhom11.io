import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import joblib
import random

# ====================== Utility ======================
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def gini(y):
    counts = Counter(y)
    impurity = 1.0
    for lbl in counts:
        prob = counts[lbl] / len(y)
        impurity -= prob**2
    return impurity

# ====================== Decision Tree ======================
class DecisionTree:
    def __init__(self, max_depth=10, min_size=2, n_features=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.tree = None

    def fit(self, X, y):
        self.n_features = self.n_features or int(np.sqrt(X.shape[1]))
        self.tree = self._build_tree(X, y, 0)

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1:
            return y[0]
        if depth >= self.max_depth or len(y) <= self.min_size:
            return Counter(y).most_common(1)[0][0]

        feat_idxs = np.random.choice(X.shape[1], self.n_features, replace=False)
        best_feat, best_thresh, best_score, best_split = None, None, 1e9, None

        for feat in feat_idxs:
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_idx = X[:, feat] <= t
                right_idx = ~left_idx
                if sum(left_idx) == 0 or sum(right_idx) == 0:
                    continue
                gini_left = gini(y[left_idx])
                gini_right = gini(y[right_idx])
                score = (sum(left_idx)*gini_left + sum(right_idx)*gini_right) / len(y)
                if score < best_score:
                    best_feat, best_thresh, best_score = feat, t, score
                    best_split = (left_idx, right_idx)

        if best_feat is None:
            return Counter(y).most_common(1)[0][0]

        left = self._build_tree(X[best_split[0]], y[best_split[0]], depth+1)
        right = self._build_tree(X[best_split[1]], y[best_split[1]], depth+1)
        return (best_feat, best_thresh, left, right)

    def _predict_one(self, x, node):
        if not isinstance(node, tuple):
            return node
        feat, thresh, left, right = node
        return self._predict_one(x, left if x[feat] <= thresh else right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

# ====================== Random Forest ======================
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_size=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.trees.clear()
        for _ in range(self.n_trees):
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree = DecisionTree(self.max_depth, self.min_size, self.n_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        final_pred = [Counter(preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(final_pred)
    
    def predict_proba(self, X):
        # Predict each tree’s label for each sample
        preds = np.array([tree.predict(X) for tree in self.trees])
        probs = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            probs[:, i] = np.mean(preds == cls, axis=0)
        return probs