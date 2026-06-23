import numpy as np

def soft_threshold(z, gamma):
    if z > gamma:
        return z - gamma
    elif z < -gamma:
        return z + gamma
    else:
        return 0.0

class LassoManual:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self.loss_history_ = []
        self.n_iter_ = 0

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n, p = X.shape

        beta = np.zeros(p)
        intercept = np.mean(y) if self.fit_intercept else 0.0
        self.loss_history_ = []

        for iteration in range(self.max_iter):
            beta_old = beta.copy()

            if self.fit_intercept:
                intercept = np.mean(y - X @ beta)

            for j in range(p):
                r_j = y - intercept - X @ beta + beta[j] * X[:, j]
                rho_j = np.dot(X[:, j], r_j)
                z_j = np.dot(X[:, j], X[:, j])

                if z_j == 0:
                    beta[j] = 0.0
                else:
                    beta[j] = soft_threshold(rho_j, self.alpha * n) / z_j

            y_pred = intercept + X @ beta
            mse = np.mean((y - y_pred) ** 2)
            self.loss_history_.append(mse)

            if np.max(np.abs(beta - beta_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = beta
        self.intercept_ = intercept
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return self.intercept_ + X @ self.coef_