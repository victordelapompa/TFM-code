"""censored gaussian process. """

# Authors: Victor de la Pompa <vpp020294@gmail.com>
#
# License: 
# Implements the GPR Censored, based on the GPR inplementation of sklearn
import warnings
from operator import itemgetter

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.deprecation import deprecated

from expectation_propagation import ep_gpr_censored

class GaussianProcessRegressorCensored(BaseEstimator, RegressorMixin):
    """Gaussian process regression Censored (GPRC).
        Hyperparametrization is calculated with a normal GPR.
    """
    def __init__(self, gpr, max_iter = 10, tol = 1e-5, lb = 0, ub = 1):
        self.gpr = gpr
        self.max_iter = max_iter
        self.tol = tol
        self.lb = lb
        self.ub = ub

    def fit(self, X, y):
        """Fit Gaussian process regression censored model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        self.gpr.fit(X,y)
        params = self.gpr.kernel_.get_params()
        keys = np.array(list(params.keys()))
        idx = ['noise' in x for x in keys]
        key_noise = keys[idx][0]
        print("Ruido obtenido :", params[key_noise])
        self.sigma_noise = params[key_noise]
        K = self.gpr.kernel_(self.gpr.X_train_) - self.sigma_noise*np.eye(len(y))
        self.mu_tilda, self.sigma_tilda, self.z_tilda = ep_gpr_censored(K, y, self.sigma_noise, max_iter = self.max_iter, 
                                                                        tol = self.tol, lb = self.lb, ub = self.ub)

        K += np.diag(self.sigma_tilda)
        try:
            self.L_ = cholesky(K, lower=True)  
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Look at the function "
                        "ep_gpr_censored there might be something wrong there."
                        % self.gpr.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.mu_tilda) 


    def predict(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process Censored regression model

        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True) or covariance (return_cov=True).
        Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated

        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean

        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points

        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.

        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")
        K_trans = self.gpr.kernel_(X, self.gpr.X_train_)
        y_mean = K_trans.dot(self.alpha_)
        if return_cov:
            v = cho_solve((self.L_, True), K_trans.T)  # Line 5
            y_cov = self.gpr.kernel_(X) - K_trans.dot(v)  # Line 6
            return y_mean, y_cov
        elif return_std:
            # compute inverse K_inv of K based on its Cholesky
            # decomposition L and its inverse L_inv
            L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
            K_inv = L_inv.dot(L_inv.T)
            # Compute variance of predictive distribution
            y_var = self.gpr.kernel_.diag(X)
            y_var -= np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)

            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn("Predicted variances smaller than 0. "
                            "Setting those variances to 0.")
                y_var[y_var_negative] = 0.0
            return y_mean, np.sqrt(y_var)
        else:
            L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
            K_inv = L_inv.dot(L_inv.T)
            # Compute variance of predictive distribution
            y_var = self.gpr.kernel_.diag(X)
            y_var -= np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans) 
            y_mean2 = norm.cdf((y_mean-1.0)/np.sqrt(y_var + self.sigma_noise))
            y_mean2 += (y_var+self.sigma_noise)*(norm.pdf(0, y_mean, np.sqrt(y_var + self.sigma_noise)) - norm.pdf(1, y_mean, np.sqrt(y_var + self.sigma_noise)))
            y_mean2 += y_mean*(norm.cdf((1-y_mean)/np.sqrt(y_var + self.sigma_noise)) - norm.cdf(-y_mean/np.sqrt(y_var + self.sigma_noise)))
            
            y_median = np.array([self.calculate_ymedian(y_mean[i], y_var[i]) for i in range(len(y_mean))])

            return y_mean, y_mean2, y_median

    def calculate_ymedian(self, mu, sigma2):
        if mu <= 0:
            return 0
        elif mu >= 1:
            return 1
        val = 1 - norm.cdf(mu/np.sqrt(sigma2 + self.sigma_noise))

        return mu

