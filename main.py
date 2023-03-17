import os.path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, root_scalar, minimize, fminbound
import math
from scipy import stats
import pdb
import pandas as pd
from utils import *
# sns.set_theme(style="darkgrid")


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q


def train_over(sig, epochs=50000, repetition=3, random_seed=[1, 2, 3], eta_list=[], initial="normal"):

    for rep in range(repetition):

        v = 20
        N = v
        n = v
        m = v
        h = 500
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

        f = open(os.path.join(os.getcwd(), f"m={m}_train_over_rep={rep}_sig={sig}.txt"), 'a')

        for index in range(2):
            w1_list[index] = w1
            w2_list[index] = w2
            loss_monitor_list[index] = []
            if index == 0:
                eta_list[index] = find_lr(x, y, w1_initial, w2_initial, c1, c2, w1_initial, w2_initial, 'f')
            else:
                eta_list[index] = [find_lr(x, y, w1_initial, w2_initial, c1, c2, w1_initial, w2_initial, 'rho_hat')]

        for epoch in range(epochs):

            for index in range(2):

                if index == 0:
                    eta = eta_list[index]
                else:
                    eta = eta_list[index][-1]

                loss_monitor_list[index].append(1 / N * loss(x, y, w1_list[index], w2_list[index]))
                f.write(f"epoch={epoch}, loss={1 / N * loss(x, y, w1_list[index], w2_list[index])}\n")
                print(f"epoch={epoch}, loss={1 / N * loss(x, y, w1_list[index], w2_list[index])}")
                w1_new = w1_list[index] + eta * x.T.dot(e(x, y, w1_list[index], w2_list[index])).dot(w2_list[index].T)
                w2_new = w2_list[index] + eta * w1_list[index].T.dot(x.T.dot(e(x, y, w1_list[index], w2_list[index])))

                # update theoretical bound
                if epoch == 0:
                    new_tight_list[index] = [1/N*loss(x, y, w1_list[index], w2_list[index])]
                else:
                    if index == 0:
                        new_tight_list[index].append(new_tight_list[index][-1]
                                                     * target(eta, x, y, w1_list[index], w2_list[index],
                                                              alpha(x, y, w1_list[index], w2_list[index]),
                                                              beta(x, y, w1_list[index], w2_list[index]),
                                                              p2(x, y, w1_list[index], w2_list[index]),
                                                              c1, c2))
                    else:
                        new_tight_list[index].append(new_tight_list[index][-1]
                                                     * target(eta, x, y, w1_list[index], w2_list[index],
                                                              alpha(x, y, w1_list[index], w2_list[index]),
                                                              beta(x, y, w1_list[index], w2_list[index]),
                                                              p2(x, y, w1_list[index], w2_list[index])))
                    # new_tight_list[index].append(new_tight_list[index][-1]
                    #                              * target(eta, x, y, w1_list[index], w2_list[index],
                    #                                       np.linalg.norm(w1_list[index], ord=-2)**2+np.linalg.norm(w2_list[index], ord=-2)**2,
                    #                                       np.linalg.norm(w1_list[index], ord=2)**2+np.linalg.norm(w2_list[index], ord=2)**2,
                    #                                       np.linalg.norm(w1_list[index].dot(w2_list[index]), ord=2)))
                f.write(f"epoch={epoch}, bound={new_tight_list[index][-1]}\n")
                print(f"epoch={epoch}, bound={new_tight_list[index][-1]}")

                w1_list[index] = w1_new
                w2_list[index] = w2_new

                # find next step step size
                if index == 1:
                    eta_list[index].append(find_lr(x, y, w1_new, w2_new, c1, c2, w1_initial, w2_initial, 'rho_hat'))

        f.close()
        f = open(os.path.join(os.getcwd(), f"m={m}_train_over_rep={rep}_sig={sig}_final_ratio.txt"), 'a')
        f.write(str(alpha(x, y, w1_list[1], w2_list[1])/beta(x, y, w1_list[1], w2_list[1])))
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

        w1_initial = w1
        w2_initial = w2
        c1 = 1.2
        c2 = 0.8
        # our lr
        lr_our_work = find_lr(x, y, w1, w2, c1, c2, w1_initial, w2_initial, 'f')
        # arora lr
        eta_arora = p1(x, y, w1, w2) ** 3 / (10**5*np.linalg.norm(y, 'fro')**4)
        # du lr
        r = np.linalg.matrix_rank(y)
        eps = np.linalg.norm(y, 'fro') / 2
        eta_du = []
        for i in range(epochs):
            eta_du.append(np.sqrt(eps/r) / (100*(i+1) * np.linalg.norm(y, 'fro') ** (3/2)))

        loss_list = {}
        eta_list = {}
        w_list = {}
        for i in range(4):
            """0: our method with fixed lr
               1: our method with adaptive lr
               2: arora lr
               3: du lr
            """
            loss_list[i] = []
            w_list[i] = (w1, w2)
            if i == 0 or i == 1:
                eta_list[i] = find_lr(x, y, w1, w2, c1, c2, w1_initial, w2_initial, 'rho_hat')
            if i == 1:
                eta_list[i] = [find_lr(x, y, w1, w2, c1, c2, w1_initial, w2_initial, 'rho_hat')]
            elif i == 2:
                eta_list[i] = eta_arora
            elif i == 3:
                eta_list[i] = eta_du

        f1 = open(os.path.join(os.getcwd(), f"compare_rep={rep}.txt"), 'a')
        f2 = open(os.path.join(os.getcwd(), f"compare_rep={rep}_lr.txt"), 'a')
        # f = open(os.path.join(os.getcwd(), f"compare_small/step_size.txt"), 'a')
        for j in range(epochs):
            for i in range(4):
                if i == 1:
                    eta_list[i].append(find_lr(x, y, w1, w2, c1, c2, w1, w2, 'rho_hat'))
                    current_eta = eta_list[i][-1]
                elif i == 3:
                    current_eta = eta_list[i][j]
                else:
                    current_eta = eta_list[i]

                w1, w2 = w_list[i]
                w1_new = w1 + current_eta * x.T.dot(e(x, y, w1, w2)).dot(w2.T)
                w2_new = w2 + current_eta * w1.T.dot(x.T.dot(e(x, y, w1, w2)))
                w_list[i] = (w1_new, w2_new)
                loss_list[i].append(1/N*loss(x, y, w1, w2))
                f1.write(f"epoch={j}, loss={1/N*loss(x, y, w1, w2)}\n")
                f2.write(f"epoch={j}, current_eta={current_eta}\n")
                print(f"epoch={j}, loss={1/N*loss(x, y, w1, w2)}\n")


if __name__ == "__main__":
    # code for plot training loss in section4.1
    train_over(0.1, 500, 3)
    train_over(2.0, 500, 3)
    train_over(10.0, 500, 3)
    # code for plot training loss in section4.2
    compare_diff_lr(3, 100, [0, 1, 3])