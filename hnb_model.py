import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

class HybridNeuralBoostingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb=None, lgbm=None, mlp=None, alpha=0.25):
        self.xgb = xgb
        self.lgbm = lgbm
        self.mlp = mlp
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self
        
    def optimize_alpha(self, X, y):
        best_alpha = self.alpha
        best_score = 0.0
        
        xgb_proba = self.xgb.predict_proba(X)
        lgbm_proba = self.lgbm.predict_proba(X)
        mlp_proba = self.mlp.predict_proba(X)
        tree_proba = (xgb_proba + lgbm_proba) / 2.0
        
        for alpha in np.linspace(0, 1.0, 21):
            final_proba = alpha * mlp_proba + (1.0 - alpha) * tree_proba
            preds = np.argmax(final_proba, axis=1)
            acc = accuracy_score(y, preds)
            if acc > best_score:
                best_score = acc
                best_alpha = alpha
                
        self.alpha = best_alpha
        print(f"Optimized alpha: {self.alpha:.2f} (Accuracy: {best_score:.4f})")
        return self.alpha

    def predict_proba(self, X):
        xgb_proba = self.xgb.predict_proba(X)
        lgbm_proba = self.lgbm.predict_proba(X)
        mlp_proba = self.mlp.predict_proba(X)
        
        tree_proba = (xgb_proba + lgbm_proba) / 2.0
        final_proba = self.alpha * mlp_proba + (1.0 - self.alpha) * tree_proba
        return final_proba

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
