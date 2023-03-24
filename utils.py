import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, root_scalar, minimize, fminbound
import math
import scipy
import pdb


def loss(x, y, u, v):
    return 1 / 2 * np.linalg.norm((y - x.dot(u.dot(v))), 'fro') ** 2


def loss_under(x, y, w):
    return 1 / 2 * np.linalg.norm((y - x.dot(w)), 'fro') ** 2


def d(u, v):
    return u.T.dot(u) - v.dot(v.T)


def e(x, y, u, v):
    return y - x.dot(u.dot(v))


def c(x):
    return (np.linalg.norm(x, ord=2) / np.linalg.norm(x, ord=-2)) ** 2


def p1(x, y, u, v, idx):
    if idx > np.min([y.shape[0], y.shape[1]]):
        return 0.0
    else:
        return np.max([np.linalg.norm(y, ord=-2) - np.linalg.norm(e(x, y, u, v), 'fro'), 0]) / np.linalg.norm(x, ord=2)


def p2(x, y, u, v):
    return (np.linalg.norm(y, 'fro') + np.linalg.norm(e(x, y, u, v), 'fro')) / np.linalg.norm(x, ord=-2)


def beta(x, y, u, v):
    imbalance = d(u, v)
    w, _ = np.linalg.eig(imbalance)
    w_neg, _ = np.linalg.eig(-imbalance)
    w = np.real(w)
    w = -np.sort(-w)
    w_neg = np.real(w_neg)
    w_neg = -np.sort(-w_neg)
    l1 = np.max([w[0], 0])
    l2 = np.max([w_neg[0], 0])
    p = p2(x, y, u, v)
    return (l1 + np.sqrt(l1 ** 2 + 4 * p ** 2)) / 2 + (l2 + np.sqrt(l2 ** 2 + 4 * p ** 2)) / 2


def alpha(x, y, u, v):
    r = np.linalg.matrix_rank(x)
    m = v.shape[1]
    pm = p1(x, y, u, v, m)
    pr = p1(x, y, u, v, r)
    imbalance = d(u, v)
    w = scipy.linalg.eigh(imbalance, eigvals_only=True)
    w = -np.sort(-w)

    w_neg = scipy.linalg.eigh(-imbalance, eigvals_only=True)
    w_neg = -np.sort(-w_neg)

    delta1 = np.max([w[0], 0]) - np.max([w[r-1], 0])
    delta2 = np.max([w_neg[0], 0]) - np.max([w_neg[m-1], 0])
    delta3 = np.max([w[r-1], 0]) + np.max([w_neg[m-1], 0])
    ret1 = -delta1+np.sqrt((delta1+delta3)**2+4*pm**2)
    ret2 = -delta2+np.sqrt((delta2+delta3)**2+4*pr**2)
    return (ret1 + ret2) / 2


def a1(x, alpha_value, c2):
    return 2 * alpha_value * c2 * np.linalg.norm(x, ord=-2) ** 2


def a2(x, y, u, v, c1, c2, alpha_value, beta_value, p2_value, version):
    if version == 'aistats':
        return np.sqrt(8) * p2_value * np.sqrt(loss(x, y, u, v)) * np.linalg.norm(x, ord=2) \
               * np.linalg.norm(x, ord=-2) ** 2 \
               + c(x) * beta_value ** 2 * c1 ** 2 * np.linalg.norm(x, ord=-2) ** 4
    else:
        K = np.linalg.norm(x, ord=2) ** 2
        mu = np.linalg.norm(x, ord=-2) ** 2
        return mu * (K * c1 * c2 * alpha_value * beta_value + c2 * alpha_value * np.sqrt(2 * K * loss(x, y, u, v)))


def a3(x, y, u, v, c1, c2, alpha_value, beta_value, p2_value, version):
    if version == 'aistats':
        return np.sqrt(8) * beta_value * p2_value * c1 * np.sqrt(loss(x, y, u, v)) * np.linalg.norm(x, ord=2) ** 3 \
               * np.linalg.norm(x, ord=-2) ** 2
    else:
        K = np.linalg.norm(x, ord=2) ** 2
        mu = np.linalg.norm(x, ord=-2) ** 2
        return 3 * mu * K * c1 * c2 * alpha_value * beta_value * np.sqrt(2 * K * loss(x, y, u, v))


def a4(x, y, u, v, c2, alpha0, p2_value, version):
    if version == "aistats":
        return 2 * c(x) ** 2 * p2_value ** 2 * loss(x, y, u, v) * np.linalg.norm(x, ord=-2) ** 6
    elif version == 'neurips':
        K = np.linalg.norm(x, ord=2) ** 2
        mu = np.linalg.norm(x, ord=-2) ** 2
        return 6 * mu * K ** 2 * p2_value * c2 * alpha0 * loss(x, y, u, v)


def target(eta, x, y, u, v, alpha_value, beta_value, p2_value, c1=1.0, c2=1.0):
    poly = 1.0 \
           - a1(x, alpha_value, c2) * eta \
           + a2(x, y, u, v, c1, beta_value, p2_value) * eta ** 2 \
           + a3(x, y, u, v, c1, beta_value, p2_value) * eta ** 3 \
           + a4(x, y, u, v, p2_value) * eta ** 4
    return poly


def find_lr(x, y, u, v, c1, c2, u_initial, v_initial, version, alpha_0=0.0, beta_0=0.0, p2_0=0.0, constraint=True):

    if version == 'rho_hat':
        alpha_t = alpha(x, y, u, v)
        beta_t = beta(x, y, u, v)
        p2_t = p2(x, y, u, v)
        roots = np.roots([4 * a4(x, y, u, v, p2_t),
                          3 * a3(x, y, u, v, 1.0, beta_t, p2_t),
                          2 * a2(x, y, u, v, 1.0, beta_t, p2_t),
                          -a1(x, alpha_t, 1.0)])

    elif version == 'rho':
        r = np.linalg.matrix_rank(x)
        m = v.shape[1]
        u1 = u.dot(u.T)
        v1 = v.T.dot(v)
        eigen_u1 = scipy.linalg.eigvals(u1)
        eigen_u1 = np.real(-np.sort(-eigen_u1))
        eigen_v1 = scipy.linalg.eigvals(v1)
        eigen_v1 = np.real(-np.sort(-eigen_v1))
        alpha_t = eigen_u1[r-1]+eigen_v1[m-1]
        beta_t = np.linalg.norm(u, ord=2) ** 2 + np.linalg.norm(v, ord=2)
        p2_t = np.linalg.norm(u.dot(v), ord=2)
        roots = np.roots([4 * a4(x, y, u, v, p2_t),
                          3 * a3(x, y, u, v, 1.0, beta_t, p2_t),
                          2 * a2(x, y, u, v, 1.0, beta_t, p2_t),
                          -a1(x, alpha_t, 1.0)])

    else:
        roots = np.roots([4 * a4(x, y, u, v, p2_0),
                          3 * a3(x, y, u, v, c1, beta_0, p2_0),
                          2 * a2(x, y, u, v, c1, beta_0, p2_0),
                          -a1(x, alpha_0, c2)])

    if constraint:
        roots_g1 = np.roots([a4(x, y, u_initial, v_initial, p2_0),
                             a3(x, y, u_initial, v_initial, c1, beta_0, p2_0),
                             a2(x, y, u_initial, v_initial, c1, beta_0, p2_0) + (
                                         4 * c1 * loss(x, y, u_initial, v_initial) * np.linalg.norm(x, ord=2) ** 2) / (
                                         c1 - 1),
                             -a1(x, alpha_0, c2)])
        roots_g2 = np.roots([a4(x, y, u_initial, v_initial, p2_0),
                             a3(x, y, u_initial, v_initial, c1, beta_0, p2_0),
                             a2(x, y, u_initial, v_initial, c1, beta_0, p2_0) + (
                                         8 * c1 * beta_0 * loss(x, y, u_initial, v_initial) * np.linalg.norm(x, ord=2) ** 2) / (
                                         (1 - c2) * alpha_0),
                             -a1(x, alpha_0, c2)])
        for i in roots_g1:
            if np.isreal(i) == False:
                pass
            else:
                r2 = np.real(i)

        for i in roots_g2:
            if np.isreal(i) == False:
                pass
            else:
                r3 = np.real(i)

    for i in roots:
        if np.isreal(i) == False:
            pass
        elif i < 0:
            pass
        else:
            r1 = np.real(i)
    if constraint:
        return np.min([r1, r2, r3])
    else:
        return r1



