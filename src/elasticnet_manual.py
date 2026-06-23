import numpy as np


def soft_threshold(z, gamma):
    """
    Operador de umbralización suave (soft-thresholding).
    Proviene de la penalización L1 de LASSO/Elastic Net.

    S(z, gamma) =  z - gamma   si z >  gamma
                   0            si |z| <= gamma
                   z + gamma   si z < -gamma
    """
    if z > gamma:
        return z - gamma
    elif z < -gamma:
        return z + gamma
    else:
        return 0.0


class ElasticNetFromScratch:
    """
    Elastic Net via Coordinate Descent implementado desde cero.

    Minimiza:
        (1/2n) * ||y - Xβ||² + λ * [α * ||β||₁ + (1-α)/2 * ||β||²]

     """

    def __init__(self, lambda_=0.01, alpha=0.5, max_iter=1000, tol=1e-6):
        self.lambda_   = lambda_
        self.alpha     = alpha
        self.max_iter  = max_iter
        self.tol       = tol
        self.coef_         = None
        self.intercept_    = None
        self.n_iter_       = 0
        self.loss_history_ = []

    def _loss(self, X, y, beta, intercept):
        """Función de pérdida completa de Elastic Net."""
        n          = len(y)
        residuals  = y - (X @ beta + intercept)
        mse        = np.sum(residuals**2) / (2 * n)
        l1_pen     = self.lambda_ * self.alpha * np.sum(np.abs(beta))
        l2_pen     = self.lambda_ * (1 - self.alpha) / 2 * np.sum(beta**2)
        return mse + l1_pen + l2_pen

    def fit(self, X, y):
        """
        Algoritmo:
        1. Inicializar todos los β en 0
        2. Para cada característica j, calcular el residuo parcial
           (residuo sin la contribución de β_j)
        3. Actualizar β_j con la fórmula de Elastic Net:
               β_j ← S(ρ_j, λα) / (1 + λ(1-α))
           donde ρ_j = (1/n) * Σ x_ij * (y_i - ŷ_i^(-j))
        4. Repetir hasta convergencia
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n, p = X.shape

        beta      = np.zeros(p)
        intercept = np.mean(y)

        for iteration in range(self.max_iter):
            beta_old  = beta.copy()
            intercept = np.mean(y - X @ beta)

            for j in range(p):
                residual_j = y - intercept - X @ beta + X[:, j] * beta[j]
                rho_j      = (1 / n) * np.dot(X[:, j], residual_j)
                numerador   = soft_threshold(rho_j, self.lambda_ * self.alpha)
                denominador = 1 + self.lambda_ * (1 - self.alpha)
                beta[j]     = numerador / denominador

            self.loss_history_.append(self._loss(X, y, beta, intercept))

            if np.max(np.abs(beta - beta_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_      = beta
        self.intercept_ = intercept
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return X @ self.coef_ + self.intercept_
