"""Expectation Propagation for censored gaussian process. """

# Authors: Victor de la Pompa <vpp020294@gmail.com>
#
# License: 
# Implements the algorithm of Expectation Propagation from the paper
# Gaussian Process Regression with Censored Data Using Expectation Propagation by Perry Groot, Peter Lucas.

import numpy as np
from copy import deepcopy
from scipy.stats import norm
from math import exp, sqrt, pi
""" Applies the EP algorithm to calculate the distribution of p(f | X , y).
    With the assumptions that p(f, X) is a N(0, K) and p(y | f) factorizes,
    i.e., p(y | f) = p(y1 | f1)p(y2 | f2) ... p(yN | fN) and the posterior
    is approximated using Gaussians.

    Parameters
    ----------
    K : array-like, shape = (n_samples_X, n_samples_X)
        The covariance matrix of f | X

    y : array-like, shape = (n_samples_X)
        The vector of targets.

    sigma_noise : int
        Value of sigma^2_noise

    max_iter : int, default: 10
        Maximum number of iterations for the EP algorithm

    tol : int, default : 1e-5
        Tolerance to stop the optimization

    lb = int, default : 0
        lower bound of the censored GP

    ub = int, default: 1
        upper bound of the censored GP

    Returns 
    -------
    mu_tilda : The mean of the approximated likelihood
    sigma_tilda :  The covariance of the approximated likelihood
    z_tilda : The normalization terms
    
""" 
def ep_gpr_censored(K, y, sigma_noise, max_iter = 10, tol = 1e-5, lb = 0, ub = 1):
    # Note all the sigmas are variances and not the stdev.
    N = len(y)
    mu_tilda = deepcopy(y)
    sigma_tilda = sigma_noise*np.ones(N)
    z_tilda = np.ones(N)


    mu_tilda_old = deepcopy(y)
    sigma_tilda_old = sigma_noise*np.ones(N)
    z_tilda_old = np.ones(N)

    j = 0

    flag = True
    while(j < max_iter and flag):
        j += 1
        mu_tilda_old[:] = mu_tilda[:]
        sigma_tilda_old[:] = sigma_tilda[:]
        z_tilda_old[:] = z_tilda[:]

        for i in range(N):
            # Calculate the matrix Sigma and the vector mu

            sigma = K - K @ np.linalg.inv(K + np.diag(sigma_tilda)) @ K
            mu = sigma @ (mu_tilda/sigma_tilda)

            # Calculate the cavity distribution q\i
            sigma_c = 1.0/(1.0/sigma[i,i] - 1.0/sigma_tilda[i])
            mu_c = sigma_c*(mu[i]/sigma[i,i] - mu_tilda[i]/sigma_tilda[i])

            # Calculate z, mu, sigma with hats by cases
            if y[i] == lb:
                z_lb = (mu_c - lb)/sqrt(sigma_noise + sigma_c)
                z_hat = 1 - norm.cdf(z_lb)
                mu_hat = mu_c - sigma_c*norm.pdf(z_lb)/(z_hat*sqrt(sigma_noise + sigma_c))
                sigma_hat = sigma_c - sigma_c*sigma_c*norm.pdf(z_lb)/((sigma_noise + sigma_c)*z_hat) * (-z_lb + norm.pdf(z_lb)/z_hat)


            elif y[i] == ub:
                z_ub = (mu_c - ub)/sqrt(sigma_noise + sigma_c)
                z_hat = norm.cdf(z_ub)
                mu_hat = mu_c + sigma_c*norm.pdf(z_ub)/(z_hat*sqrt(sigma_noise + sigma_c))
                sigma_hat = sigma_c - sigma_c*sigma_c*norm.pdf(z_ub)/((sigma_noise + sigma_c)*z_hat) * (z_ub + norm.pdf(z_ub)/z_hat)


            else: # lb < y[i] < ub
                z_y = (mu_c - y[i])/sqrt(sigma_noise + sigma_c)
                z_hat = exp(-(y[i]-mu_c)**2/(2*(sigma_noise + sigma_c)))/sqrt(2*pi*(sigma_noise + sigma_c))
                mu_hat = mu_c + sigma_c*(y[i]-mu_c)/(sigma_noise + sigma_c)
                sigma_hat = sigma_c - sigma_c * sigma_c/(sigma_noise + sigma_c)

            # Calculate the new mu, sigma, z tilda
            sigma_tilda[i] = 1.0/(1.0/sigma_hat - 1/sigma_c)
            if sigma_tilda[i] < 0:
                print("Error sigma negativa: ", sigma_tilda[i])
            mu_tilda[i] = sigma_tilda[i]*(mu_hat/sigma_hat - mu_c/sigma_c)
            z_tilda[i] = z_hat*sqrt(2*pi*(sigma_c + sigma_tilda[i]))*exp((mu_c - mu_tilda[i])**2/(2*(sigma_c + sigma_tilda[i])))


        flag = max(abs(mu_tilda_old - mu_tilda)) > tol or \
               max(abs(sigma_tilda_old - sigma_tilda)) > tol or \
               max(abs(z_tilda_old - z_tilda)) > tol

    return mu_tilda, sigma_tilda, z_tilda