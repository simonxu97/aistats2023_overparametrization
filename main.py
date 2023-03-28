import os.path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, root_scalar, minimize, fminbound
import math
from scipy import stats
import pdb
import pandas as pd
from utils import *
from plot import *
from plot import plot_compare_rate
# sns.set_theme(style="darkgrid")


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q


def train_over(sig, epochs=50000, repetition=3, random_seed=[1, 2, 3], eta_list=[], initial="normal"):

    for rep in range(repetition):

        N = 20
        n = 20
        m = 20
        h = 1000
        np.random.seed(random_seed[rep])
        temp = np.random.normal(0, 1, size=(N, n))
        x, _ = np.linalg.qr(temp)
        theta = np.random.normal(0, np.sqrt(1/n), size=(n, m))
        y = x.dot(theta)

        # # initialization
        u0 = np.random.normal(0, 1, size=(n, h))
        v0 = np.random.normal(0, 1, size=(h, m))

        if sig >= 0:
            ratio = np.sqrt((sig + np.sqrt(sig ** 2 + 4)) / 2.0)
            w1 = ratio * u0
            w2 = v0 / ratio
        else:
            temp = theta.T.dot(theta)
            u, s, vh = np.linalg.svd(temp, full_matrices=False)
            temp_pos = np.dot(u, np.dot(np.diag(s ** (1 / 4)), vh))
            temp_neg = np.dot(u, np.dot(np.diag(s ** (-1 / 4)), vh))
            Q = gram_schmidt_columns(theta)
            w1 = theta.dot(temp_neg).dot(Q.T)
            w2 = Q.T.dot(temp_pos)

        # print(alpha(x, y, w1, w2) / beta(x, y, w1, w2))
        c1 = 1.5
        c2 = 0.5
        eta_list = {}
        w1_list = {}
        w2_list = {}
        loss_monitor_list = {}
        new_tight_list = {}
        w1_initial = w1
        w2_initial = w2
        # print(loss(x, y, w1, w2)/N)
        # exit()
        alpha0 = alpha(x, y, w1_initial, w2_initial)
        beta0 = beta(x, y, w1_initial, w2_initial)
        p2_0 = p2(x, y, w1_initial, w2_initial)

        f = open(os.path.join(os.getcwd(), f"m={m}_train_over_rep={rep}_sig={sig}.txt"), 'a')
        f2 = open(os.path.join(os.getcwd(), f"m={m}_train_over_rep={rep}_sig={sig}_ratio.txt"), 'a')

        for index in range(4):
            '''
                0: fixed step size minimizing f
                1: adaptive step size minimizing f
                2: adaptive step size minimizing rho_hat
                3: adaptive step size minimizing rho
            '''
            w1_list[index] = w1
            w2_list[index] = w2
            loss_monitor_list[index] = []
            if index == 0:
                eta_list[index] = find_lr(x, y, w1_initial, w2_initial, c1, c2, w1_initial, w2_initial, 'fixed', alpha0, beta0, p2_0)
            elif index == 1:
                eta_list[index] = [find_lr(x, y, w1_initial, w2_initial, c1, c2, w1_initial, w2_initial, 'f', alpha0, beta0, p2_0)]
            elif index == 2:
                eta_list[index] = [find_lr(x, y, w1_initial, w2_initial, c1, c2, w1_initial, w2_initial, 'rho_hat', alpha0, beta0, p2_0)]
                print(eta_list[index])
            else:
                eta_list[index] = [find_lr(x, y, w1_initial, w2_initial, c1, c2, w1_initial, w2_initial, 'rho', alpha0, beta0, p2_0)]
                print(eta_list[index])

        rate0 = target(eta_list[0], x, y, w1_initial, w2_initial,
                       alpha0,
                       beta0,
                       p2(x, y, w1_initial, w2_initial),
                       c1, c2)

        for epoch in range(epochs):
            print(f"epoch={epoch}")

            for index in [3]:

                if index == 0:
                    eta = eta_list[index]
                else:
                    eta = eta_list[index][-1]

                loss_monitor_list[index].append(1 / N * loss(x, y, w1_list[index], w2_list[index]))
                f.write(f"epoch={epoch}, loss={1 / N * loss(x, y, w1_list[index], w2_list[index])}\n")
                # print(f"epoch={epoch}, loss={1 / N * loss(x, y, w1_list[index], w2_list[index])}")
                w1_new = w1_list[index] + eta * x.T.dot(e(x, y, w1_list[index], w2_list[index])).dot(w2_list[index].T)
                w2_new = w2_list[index] + eta * w1_list[index].T.dot(x.T.dot(e(x, y, w1_list[index], w2_list[index])))

                # update theoretical bound
                if epoch == 0:
                    new_tight_list[index] = [1/N*loss(x, y, w1_list[index], w2_list[index])]
                else:
                    if index == 0:
                        new_tight_list[index].append(new_tight_list[index][-1] * rate0)
                    elif index == 1:
                        new_tight_list[index].append(new_tight_list[index][-1]
                                                     * target(eta, x, y, w1_list[index], w2_list[index],
                                                              alpha0,
                                                              beta0,
                                                              p2(x, y, w1_initial, w2_initial), c1, c2))

                    elif index == 2:
                        alpha_t = alpha(x, y, w1_list[index], w2_list[index])
                        beta_t = beta(x, y, w1_list[index], w2_list[index])
                        new_tight_list[index].append(new_tight_list[index][-1]
                                                     * target(eta, x, y, w1_list[index], w2_list[index],
                                                              alpha_t,
                                                              beta_t,
                                                              p2(x, y, w1_list[index], w2_list[index])))
                        f2.write(f"epoch={epoch}, ratio={alpha_t/beta_t}\n")

                    else:
                        r = np.linalg.matrix_rank(x)
                        m = w2_list[index].shape[1]
                        u1 = w1_list[index].dot(w1_list[index].T)
                        v1 = w2_list[index].T.dot(w2_list[index])
                        eigen_u1 = scipy.linalg.eigvals(u1)
                        eigen_u1 = np.real(-np.sort(-eigen_u1))
                        eigen_v1 = scipy.linalg.eigvals(v1)
                        eigen_v1 = np.real(-np.sort(-eigen_v1))
                        alpha_t = eigen_u1[r - 1] + eigen_v1[m - 1]
                        beta_t = np.linalg.norm(w1_list[index], ord=2) ** 2 + np.linalg.norm(w2_list[index], ord=2) ** 2
                        p2_t = np.linalg.norm(w1_list[index].dot(w2_list[index]), ord=2)
                        new_tight_list[index].append(new_tight_list[index][-1]
                                                     * target(eta, x, y, w1_list[index], w2_list[index],
                                                              alpha_t,
                                                              beta_t,
                                                              p2_t))

                        f2.write(f"epoch={epoch}, ratio={alpha_t / beta_t}\n")
                        f2.write(f"epoch={epoch}, alpha={alpha_t}\n")
                        f2.write(f"epoch={epoch}, beta={beta_t}\n")
                        f2.write(f"epoch={epoch}, eta={eta}\n")


                f.write(f"epoch={epoch}, bound={new_tight_list[index][-1]}\n")
                # print(f"epoch={epoch}, bound={new_tight_list[index][-1]}")

                w1_list[index] = w1_new
                w2_list[index] = w2_new

                # find next step step size
                if index == 1:
                    eta_list[index].append(find_lr(x, y, w1_new, w2_new,
                                                   c1, c2, w1_initial, w2_initial, 'f', alpha0, beta0, p2_0))
                elif index == 2:
                    eta_list[index].append(find_lr(x, y, w1_new, w2_new,
                                                   c1, c2, w1_initial, w2_initial, 'rho_hat', alpha0, beta0, p2_0))
                elif index == 3:
                    eta_list[index].append(find_lr(x, y, w1_new, w2_new,
                                                   c1, c2, w1_initial, w2_initial, 'rho', alpha0, beta0, p2_0))
        f.close()
        f = open(os.path.join(os.getcwd(), f"m={m}_train_over_rep={rep}_sig={sig}_initial_ratio.txt"), 'a')
        f.write(str(alpha0/beta0))
        f.close()


def generate_matrix_cond(cond, nrow, ncol):
    x = np.random.normal(0, 1, size=(nrow, ncol))
    u, s, vh = np.linalg.svd(x, full_matrices=True)
    s = np.linspace(cond, 1, min(nrow, ncol))
    return np.dot(u[:, :s.shape[0]] * s, vh)


def compare_diff_lr(repetition, epochs=50, random_seed=[1, 2, 3]):
    N = 20
    n = 20
    m = 1
    h = 500
    std = 0.5

    for rep in range(repetition):

        np.random.seed(random_seed[rep])
        x = np.eye(N)

        # initialization
        theta = np.random.normal(0, std, size=(n, m))
        u, s, vh = np.linalg.svd(theta, full_matrices=False)
        d_min = np.min([m, n])
        s_zero_pad = np.zeros([h - d_min, ])
        u_zero_pad = np.zeros([n, h - d_min])
        v_zero_pad = np.zeros([h - d_min, m])
        s = np.concatenate([s, s_zero_pad])
        u = np.concatenate([u, u_zero_pad], axis=1)
        vh = np.concatenate([vh, v_zero_pad])
        w1 = u.dot(np.diag(np.sqrt(s)))
        w2 = np.diag(np.sqrt(s)).dot(vh)
        y = x.dot(theta) + 0.04 * np.random.normal(0, 1, size=(N, m))
        print(np.linalg.norm(y-x.dot(theta), ord='fro'))

        w1_initial = w1
        w2_initial = w2
        alpha0 = alpha(x, y, w1_initial, w2_initial)
        beta0 = beta(x, y, w1_initial, w2_initial)
        p2_0 = p2(x, y, w1_initial, w2_initial)
        c1 = 1.2
        c2 = 0.8
        # arora lr
        eta_arora = p1(x, y, w1_initial, w2_initial, 1) ** 3 / (10**5*np.linalg.norm(y, 'fro')**4)
        # du lr
        r = np.linalg.matrix_rank(y)
        eps = np.linalg.norm(y, 'fro') / 2
        eta_du = []
        for i in range(epochs):
            eta_du.append(np.sqrt(eps/r) / (100*(i+1) * np.linalg.norm(y, 'fro') ** (3/2)))

        loss_list = {}
        eta_list = {}
        w_list = {}
        for index in range(6):
            """0: our method with fixed lr
               1: our method with adaptive lr with f
               2: our method with adaptive lr with rho_hat
               3: our method with adaptive lr with rho
               4: arora lr
               5: du lr
            """
            loss_list[index] = []
            w_list[index] = (w1, w2)

            if index == 0:
                eta_list[index] = [find_lr(x, y, w1_initial, w2_initial, c1, c2, w1_initial, w2_initial, 'fixed', alpha0, beta0, p2_0)]
            elif index == 1:
                eta_list[index] = [find_lr(x, y, w1_initial, w2_initial, c1, c2, w1_initial, w2_initial, 'f', alpha0, beta0, p2_0)]
            elif index == 2:
                eta_list[index] = [find_lr(x, y, w1_initial, w2_initial, c1, c2, w1_initial, w2_initial, 'rho_hat', alpha0, beta0, p2_0)]
            elif index == 3:
                eta_list[index] = [find_lr(x, y, w1_initial, w2_initial, c1, c2, w1_initial, w2_initial, 'rho', alpha0, beta0, p2_0)]
            elif index == 4:
                eta_list[index] = [eta_arora]
            else:
                eta_list[index] = eta_du

        f1 = open(os.path.join(os.getcwd(), f"compare_rep={rep}.txt"), 'a')
        f2 = open(os.path.join(os.getcwd(), f"compare_rep={rep}_lr.txt"), 'a')
        # f = open(os.path.join(os.getcwd(), f"compare_small/step_size.txt"), 'a')
        for j in range(epochs):
            for i in range(6):
                w1, w2 = w_list[i]
                loss_list[i].append(1 / N * loss(x, y, w1, w2))
                if i == 5:
                    current_eta = eta_list[i][j]
                else:
                    current_eta = eta_list[i][-1]

                w1_new = w1 + current_eta * x.T.dot(e(x, y, w1, w2)).dot(w2.T)
                w2_new = w2 + current_eta * w1.T.dot(x.T.dot(e(x, y, w1, w2)))
                w_list[i] = (w1_new, w2_new)
                f1.write(f"epoch={j}, loss={1/N*loss(x, y, w1, w2)}\n")
                f2.write(f"epoch={j}, current_eta={current_eta}\n")
                print(f"epoch={j}, loss={1/N*loss(x, y, w1, w2)}\n")

                if i == 1:
                    eta_list[i].append(find_lr(x, y, w1_new, w2_new, c1, c2, w1_initial, w2_initial, 'f', alpha0, beta0, p2_0))
                elif i == 2:
                    eta_list[i].append(find_lr(x, y, w1_new, w2_new, c1, c2, w1_initial, w2_initial, 'rho_hat', alpha0, beta0, p2_0))
                elif i == 3:
                    eta_list[i].append(find_lr(x, y, w1_new, w2_new, c1, c2, w1_initial, w2_initial, 'rho', alpha0, beta0, p2_0))


if __name__ == "__main__":
    # code for plot training loss in section4.1
    train_over(1.0, 300, 50)
    train_over(5.0, 300, 50)
    train_over(100.0, 300, 50)
    # code for plot training loss in section4.2
    # compare_diff_lr(50, 100, [i for i in range(50)])
    # plot_compare_rate(50)
    # plot_train()
