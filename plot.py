import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_file(path, num=-1, step_size=False):
    output = {}
    output[0] = []
    output[1] = []
    output[2] = []
    output[3] = []
    f = open(path, 'r')
    lines = f.readlines()
    if num > 0:
        lines = lines[:4*num]
    for idx, line in enumerate(lines):
        if step_size == False:
            loss = np.log10(float(line.split('=')[-1])) + np.log10(10)
        else:
            loss = np.log10(float(line.split('=')[-1]))
        # if idx >= 800:
        #     break
        if idx % 4 == 0:
            output[0].append(loss)
        elif idx % 4 == 1:
            output[1].append(loss)
        elif idx % 4 == 2:
            output[2].append(loss)
        else:
            output[3].append(loss)

    return output


def plot_compare_rate():
    color_bar = ['b-', 'r-', 'g-', 'y-']
    shade_color_bar = ['b', 'r', 'g', 'y']
    legend_bar = ['Fixed Step Size Thm3.2', r'Adaptive Step Size using $\hat\rho(\eta, t)$', 'Step Size by Arora et al',
                  'Step Size by Du et al']
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    path1 = f'compare_rep=0.txt'
    path2 = f'compare_rep=1.txt'
    path3 = f'compare_rep=2.txt'
    path4 = f'compare_rep=0_lr.txt'
    output1 = read_file(path1, 100)
    output2 = read_file(path2, 100)
    output3 = read_file(path3, 100)
    step_size_list = read_file(path4, 100, True)

    for k in range(4):
        temp = pd.DataFrame([output1[k], output2[k], output3[k]])
        mean = temp.mean(axis=0).to_numpy()
        std = temp.std(axis=0).to_numpy()

        ax[0].plot(mean, color_bar[k], label=legend_bar[k])
        ax[0].fill_between([j for j in range(len(mean))], mean - std, mean + std, alpha=0.2, color=shade_color_bar[k])
        ax[0].set_xlabel('Iterations', fontsize=20)
        ax[1].set_xlabel('Iterations', fontsize=20)
        if k == 0:
            ax[0].set_ylabel(r'$log_{10}L(t)$', fontsize=15, labelpad=15)
            ax[1].set_ylabel(r'$log_{10}\eta$', fontsize=15, labelpad=15)
        ax[1].plot(step_size_list[k], color_bar[k], label=legend_bar[k])
        ax[0].set_ylim((-30, 0))
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()


def plot_train():

    color_bar = ['b', 'r']
    shade_color_bar = ['b', 'r']
    legend_bar = ['Fixed Step Size Thm3.2', r'Adaptive Step Size using $\hat\rho(\eta, t)$']
    legend_bar_upper = ['Theoretical Upper Bound Thm3.2', 'Theoretical Upper Bound in equation(36)']
    sigma = [0.1, 2.0, 10.0]
    ratio = [0.345, 0.388, 0.457]
    final_ratio = [0.458, 0.477, 0.486]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):

        path1 = f'm=20_train_over_rep=0_sig={sigma[i]}.txt'
        path2 = f'm=20_train_over_rep=1_sig={sigma[i]}.txt'
        path3 = f'm=20_train_over_rep=2_sig={sigma[i]}.txt'
        output1 = read_file(path1, 500)
        output2 = read_file(path2, 500)
        output3 = read_file(path3, 500)

        for k in range(2):
            temp = pd.DataFrame([output1[2 * k], output2[2 * k], output3[2 * k]])
            temp_bound = pd.DataFrame([output1[2 * k + 1], output2[2 * k + 1], output3[2 * k + 1]])
            mean = temp.mean(axis=0).to_numpy()
            mean_bound = temp_bound.mean(axis=0).to_numpy()
            std = temp.std(axis=0).to_numpy()
            ax[i].plot(mean, color_bar[k], label=legend_bar[k])
            ax[i].fill_between([j for j in range(len(mean))], mean - std, mean + std, alpha=0.2, color=shade_color_bar[k])
            ax[i].plot(mean_bound, color_bar[k], label=legend_bar_upper[k], linestyle='dashed')
            ax[i].set_xlabel('Iterations', fontsize=15)
            ax[i].set_ylim((-25, 6))
            if i == 0 and k == 0:
                ax[i].set_ylabel(r'$log_{10}$ L(t)', fontsize=15, labelpad=20)
        ax[i].legend(loc='lower left')
        if i == 0:
            ax[i].set_title(r"$\frac{\alpha_0}{\beta_0}$=0.345, $\frac{\alpha_{final}}{\beta_{final}}=0.458$, $\sigma$=0.1", fontsize=15)
        elif i == 1:
            ax[i].set_title(r"$\frac{\alpha_0}{\beta_0}$=0.388, $\frac{\alpha_{final}}{\beta_{final}}=0.477$, $\sigma$=2.0", fontsize=15)
        else:
            ax[i].set_title(r"$\frac{\alpha_0}{\beta_0}$=0.457, $\frac{\alpha_{final}}{\beta_{final}}=0.486$, $\sigma$=10.0", fontsize=15)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # code for plot training loss in section4.1
    plot_train()

    # code for plot training loss in section4.2
    plot_compare_rate()