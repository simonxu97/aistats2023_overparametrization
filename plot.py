import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_file(path, idx_num, num=-1):
    output = {}
    for i in range(idx_num):
        output[i] = []
    f = open(path, 'r')
    lines = f.readlines()
    if num > 0:
        lines = lines[:idx_num*num]

    total_num = len(lines)
    epochs = total_num // idx_num

    for epoch in range(epochs):
        for idx in range(idx_num):
            loss = np.log10(float(lines[idx+epoch*idx_num].split('=')[-1]))
            output[idx].append(loss)
    return output


def plot_compare_rate(repetition):
    color_bar = ['b-', 'r-', 'g-', 'y-', 'm-', 'c-']
    color_bar = ['b', 'r', 'g', 'y', 'm', 'c']
    legend_bar = [r'Algorithm 1 with $h(\eta, t) = f(\eta, 0)$',
                  r'Algorithm 1 with $h(\eta, t) = f(\eta, t)$',
                  r'Algorithm 1 with $h(\eta, t) = \hat\rho(\eta, t)$',
                  r'Algorithm 1 with $h(\eta, t) = \rho(\eta, t)$',
                  'Step size by Arora et al.',
                  'Step size by Du et al.']
    fig, ax = plt.subplots(2, 1, figsize=(9, 15))
    loss_list = {}
    for i in range(repetition):
        loss_list[i] = read_file(f'compare_rep={i}.txt', 6)
    path4 = f'compare_rep=0_lr.txt'
    step_size_list = read_file(path4, 6)

    for k in range(6):
        temp = pd.DataFrame([loss_list[j][k] for j in range(repetition)])
        mean = temp.mean(axis=0).to_numpy()
        std = temp.std(axis=0).to_numpy()

        if k % 2 == 0:
            temp_marker = 'o'
            interval = 10
        else:
            temp_marker = 'v'
            interval = 6
        x_axis = [j for j in range(len(mean))][::interval]
        if x_axis[-1] < 99:
            x_axis.append(99)
        mean_current = mean[::interval]
        std_current = std[::interval]
        step_current = step_size_list[k][::interval]
        if len(std_current) < len(x_axis):
            std_current = np.append(std_current, std[-1])
            mean_current = np.append(mean_current, mean[-1])
            step_current = np.append(step_current, step_size_list[k][-1])

        ax[0].errorbar(x_axis, mean_current, std_current, color=color_bar[k], label=legend_bar[k], marker=temp_marker)
        ax[1].plot(x_axis, step_current, color_bar[k], label=legend_bar[k], marker='o')

        ax[0].set_xlabel('Iterations', fontsize=20)
        ax[1].set_xlabel('Iterations', fontsize=20)
        if k == 0:
            ax[0].set_ylabel(r'$log_{10}L(t)$', fontsize=20, labelpad=20)
            ax[1].set_ylabel(r'$log_{10}\eta$', fontsize=20, labelpad=20)
        ax[0].set_ylim((-30, 0))
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout(pad=5.0)
    plt.show()


def plot_train(repetition):

    color_bar = ['b', 'r', 'm', 'g']
    shade_color_bar = ['b', 'r', 'm', 'g']
    legend_bar = [r'Algorithm 1 with $h(\eta, t) = f(\eta, 0)$',
                  r'Algorithm 1 with $h(\eta, t) = f(\eta, t)$',
                  r'Algorithm 1 with $h(\eta, t) = \hat\rho(\eta, t)$',
                  r'Algorithm 1 with $h(\eta, t) = \rho(\eta, t)$']

    legend_bar_upper = [r'equation(35) with $h(\eta, t) = f(\eta, 0)$',
                        r'equation(35) with $h(\eta, t) = f(\eta, t)$',
                        r'equation(35) with $h(\eta, t) = \hat\rho(\eta, t)$',
                        r'equation(35) with $h(\eta, t) = \hat\rho(\eta, t)$']
    sigma = [1.0, 5.0, 100.0]

    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    for i in range(3):

        loss_list = {}
        for j in range(repetition):
            loss_list[j] = read_file(f'tightness_checkness_sig={sigma[i]}/m=20_train_over_rep={j}_sig={sigma[i]}.txt', 8, 200)

        for k in range(4):
            temp = pd.DataFrame([loss_list[j][2*k] for j in range(repetition)])
            temp_bound = pd.DataFrame([loss_list[j][2*k+1] for j in range(repetition)])
            mean = temp.mean(axis=0).to_numpy()
            mean_bound = temp_bound.mean(axis=0).to_numpy()
            std = temp.std(axis=0).to_numpy()

            if k % 2 == 0:
                temp_marker = 'o'
                interval = 10
            else:
                temp_marker = 'v'
                interval = 15

            x_axis = [j for j in range(len(mean))][::interval]
            if x_axis[-1] < 199:
                x_axis.append(199)
            mean_current = mean[::interval]
            std_current = std[::interval]
            mean_bound_current = mean_bound[::interval]
            if len(std_current) < len(x_axis):
                std_current = np.append(std_current, std[-1])
                mean_current = np.append(mean_current, mean[-1])
                mean_bound_current = np.append(mean_bound_current, mean_bound[-1])

            ax[i].errorbar(x_axis, mean_current, std_current, label=legend_bar[k], errorevery=2,
                           color=color_bar[k], marker='o')
            ax[i].plot(x_axis, mean_bound_current, label=legend_bar_upper[k], color=color_bar[k],
                       marker='v')

            ax[i].set_xlabel('Iterations', fontsize=20, labelpad=20)
            ax[i].set_ylabel(r'$log_{10}$ L(t)', fontsize=20, labelpad=20)
            ax[i].legend(loc='lower left')


        ax[i].legend(loc='lower left')
        if i == 0:
            ax[i].set_title(r"$\frac{\alpha_0}{\beta_0}$=0.505, $\sigma$=1.0", fontsize=20)
        elif i == 1:
            ax[i].set_title(r"$\frac{\alpha_0}{\beta_0}$=0.555, $\sigma$=5.0", fontsize=20)
            ax[i].set_ylim((-25, 6))
        else:
            ax[i].set_title(r"$\frac{\alpha_0}{\beta_0}$=0.600, $\sigma$=100.0", fontsize=20)
            ax[i].set_ylim((-25, 6))

    plt.tight_layout(pad=5.0)
    plt.show()


if __name__ == "__main__":
    # code for plot training loss in section4.1
    plot_train(50)

    # code for plot training loss in section4.2
    # plot_compare_rate(50)

