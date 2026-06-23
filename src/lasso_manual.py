"""
lasso_manual.py  –  Implementación LASSO From Scratch (Coordinate Descent)
Estructura orientada a objetos, compatible con la interfaz de scikit-learn.
"""

import numpy as np


class LassoManual:
    """
    Regresión LASSO implementada con Coordinate Descent y Soft-Thresholding.

    Matemática central (actualización pura L1, sin denominador L2):
        β_j ← S(ρ_j, λ) / z_j

    donde:
        ρ_j  = X_j^T (y - Xβ + β_j * X_j)   (correlación parcial)
        z_j  = X_j^T X_j  = ‖X_j‖²          (normalización)
        S(ρ, λ) = sign(ρ) * max(|ρ| - λ, 0)  (Soft-Thresholding / umbral suave)

    Parámetros
    ----------
    alpha       : float  –  fuerza de regularización L1 (≥ 0)
    max_iter    : int    –  máximo de iteraciones de Coordinate Descent
    tol         : float  –  tolerancia de convergencia (cambio máximo en β)
    fit_intercept: bool  –  si True, calcula intercept sin penalizar
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000,
                 tol: float = 1e-4, fit_intercept: bool = True):
        self.alpha         = alpha
        self.max_iter      = max_iter
        self.tol           = tol
        self.fit_intercept = fit_intercept

        # Atributos aprendidos
        self.coef_      = None
        self.intercept_ = 0.0
        self.loss_history_ = []   # MSE por iteración (para curva de convergencia)
        self.n_iter_    = 0

    # ── Soft-Thresholding ─────────────────────────────────────────────────────
    @staticmethod
    def _soft_threshold(rho: float, lam: float) -> float:
        """S(ρ, λ) = sign(ρ) · max(|ρ| − λ, 0)"""
        if rho > lam:
            return rho - lam
        elif rho < -lam:
            return rho + lam
        else:
            return 0.0

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Entrena el modelo LASSO con Coordinate Descent.

        Parámetros
        ----------
        X : (n_samples, n_features)  –  datos estandarizados
        y : (n_samples,)             –  target
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n, p = X.shape

        # Inicialización de coeficientes en cero
        beta = np.zeros(p)

        # Intercept: se actualiza como media del residuo actual
        if self.fit_intercept:
            intercept = np.mean(y)
        else:
            intercept = 0.0

        self.loss_history_ = []

        for iteration in range(self.max_iter):
            beta_old = beta.copy()

            # Actualizar intercept (no penalizado)
            if self.fit_intercept:
                residual = y - X @ beta
                intercept = np.mean(residual)

            # Coordinate Descent: recorrer cada feature j
            for j in range(p):
                # Residuo parcial (excluye la contribución de β_j)
                r_j = y - intercept - X @ beta + beta[j] * X[:, j]

                # Correlación parcial
                rho_j = X[:, j] @ r_j          # X_j^T · r_j

                # Normalización ‖X_j‖²
                z_j = X[:, j] @ X[:, j]        # = n si los datos están estandarizados

                if z_j == 0:
                    beta[j] = 0.0
                else:
                    beta[j] = self._soft_threshold(rho_j, self.alpha * n) / z_j

            # Pérdida MSE de esta iteración
            y_pred = intercept + X @ beta
            mse    = np.mean((y - y_pred) ** 2)
            self.loss_history_.append(mse)

            # Criterio de convergencia
            if np.max(np.abs(beta - beta_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_      = beta
        self.intercept_ = intercept
        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    # ── Métricas ──────────────────────────────────────────────────────────────
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))

    def mae(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))

    # ── Repr ──────────────────────────────────────────────────────────────────
    def __repr__(self):
        return (f"LassoManual(alpha={self.alpha}, max_iter={self.max_iter}, "
                f"tol={self.tol}, fit_intercept={self.fit_intercept})")