import os

from SALib.analyze import morris as analyze_morris
from SALib.sample.morris import sample as sample_morris
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.sample import finite_diff
from SALib.analyze import dgsm
from SALib.sample import fast_sampler
from SALib.analyze import fast
from SALib.sample import latin
from SALib.analyze import rbd_fast
from SALib.analyze import delta

import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import networkx as nx

import argparse

def nan_to_zero(si_result):
    for k in si_result.keys():
        si_result[k] = np.nan_to_num(si_result[k])
    return si_result

def S2_to_dict(matrix, problem):
    result = {}
    names = list(problem["names"])

    for i in range(problem["num_vars"]):
        for j in range(i + 1, problem["num_vars"]):
            if names[i] not in result:
                result[names[i]] = {}
            if names[j] not in result:
                result[names[j]] = {}
            result[names[i]][names[j]] = result[names[j]][names[i]] = float(matrix[i][j])
    return result

def sort_result(result, on='ST', n=30):
    sorted_result = {'S1': {}, 'S1_conf': {}, 'ST': {}, 'ST_conf': {}, 'S2': {}, 'S2_conf': {}}
    if on == 'S2':
        sorted_result = {}
    else:
        sort_keys = [i for i, _ in sorted(result[on].items(), key=lambda x: x[1], reverse=True)]
        topn = sort_keys[:n]
        for k in topn:
            sorted_result['S1'][k] = result['S1'][k]
            sorted_result['S1_conf'][k] = result['S1_conf'][k]
            sorted_result['ST'][k] = result['ST'][k]
            sorted_result['ST_conf'][k] = result['ST_conf'][k]
        for i in topn:
            sorted_result['S2'][i] = {}
            sorted_result['S2_conf'][i] = {}
            for j in topn:
                if i != j:
                    sorted_result['S2'][i][j] = result['S2'][i][j]
                    sorted_result['S2_conf'][i][j] = result['S2_conf'][i][j]
    return sorted_result


def sort_Si(Si, key, sortby):
    return np.array([Si[key][x] for x in np.argsort(Si[sortby])])

def morris_method(args, model, problem):
    # make morris samples
    X = sample_morris(problem, N=args.trajectory_number,
               num_levels=4,
               optimal_trajectories=None)

    # evaluate model outputs
    Y = model.predict(X)
    Y = np.nan_to_num(Y).flatten()

    # morris analysis
    si_morris = analyze_morris.analyze(problem, X, Y, conf_level=0.95,
                               print_to_console=False,
                               num_levels=4, num_resamples=args.sample_number)

    # sort morris outputs
    names_sorted = sort_Si(si_morris, 'names', sortby='mu_star')
    mu_star_sorted = sort_Si(si_morris, 'mu_star', sortby='mu_star')
    sigma_sorted = sort_Si(si_morris, 'sigma', sortby='mu_star')
    mu_star_conf_sorted = sort_Si(si_morris, 'mu_star_conf', sortby='mu_star')
    mu_sorted = sort_Si(si_morris, 'mu', sortby='mu_star')

    # find effective features
    mu_star = si_morris['mu_star']
    sigma = si_morris['sigma']
    names = si_morris['names']

    effective_par = 0
    indlist = []
    ind = -1
    for m, s in zip(mu_star, sigma):
        ind += 1
        if m >= s:
            effective_par += 1
            indlist.append(ind)
            print(m, s, ind)

    # covariance plot
    y = sigma
    x = mu_star

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(x, y, c=u'b', marker=u'o')
    ax.set_ylabel(r'$\sigma$')
    ax.set_xlim(0, )
    ax.set_ylim(0, )
    x_axis_bounds = np.array(ax.get_xlim())
    line1 = ax.plot(x_axis_bounds, x_axis_bounds, 'k-', label=r'$\sigma / \mu^{\star} = 1.0$')

    for i in range(len(indlist)):
        ax.annotate(names[i], (x[i], y[i]))
        ax.scatter(x[i], y[i], c=u'r', marker=u'o')
        print('mu_star: {}, sigma: {}'.format(x[i], y[i]))

    ys = sigma_sorted[-10:]
    xs = mu_star_sorted[-10:]
    namess = names_sorted[-10:]
    for i, txt in enumerate(namess):
        ax.annotate(txt, (xs[i], ys[i]))
        ax.scatter(xs[i], ys[i], c=u'g', marker=u'o')

    unit = ''
    ax.set_xlabel(r'$\mu^\star$ ' + unit)
    ax.set_ylim(0 - (0.01 * np.array(ax.get_ylim()[1])), )

    ax.set_title('Morris analysis')
    ax.legend(loc='upper left')

    model_name = args.model.split('/')[-1]
    file_name = os.path.join(args.data_dir, 'results')+'/morris_cov_plot_'+model_name[-3]+'_'+args.feature_type
    plt.savefig(file_name+'.png',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    plt.savefig(file_name+'.pdf',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    plt.savefig(file_name+'.svg',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='svg')

    # horizontal bar plot
    y_pos = np.arange(len(names_sorted))
    plot_names = names_sorted

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 60), sharey=True)

    ax1.barh(y_pos,
             mu_star_sorted,
             xerr=mu_star_conf_sorted,
             align='center',
             ecolor='black',
             color='blue')

    ax2.barh(y_pos,
             mu_sorted,
             align='center',
             ecolor='black',
             color='red')

    ax3.barh(y_pos,
             sigma_sorted,
             align='center',
             ecolor='black',
             color='green')

    # plot mu_star distribution
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(plot_names)
    ax1.set_xlabel(r'$\mu^\star$')
    ax1.set_ylim(min(y_pos) - 1, max(y_pos) + 1)
    ax1.set_title('$\mu^\star$ distribution')

    # plot mu distribution
    ax2.set_xlabel(r'$\mu$')
    ax2.set_ylim(min(y_pos) - 1, max(y_pos) + 1)
    ax2.set_title('$\mu$ distribution')

    # plot sigma distribution
    ax3.set_xlabel(r'$\sigma$')
    ax3.set_ylim(min(y_pos) - 1, max(y_pos) + 1)
    ax3.set_title('$\sigma$ distribution')

    file_name = os.path.join(args.data_dir, 'results')+'/morris_hbar_plot_'+model_name[-3]+'_'+args.feature_type
    plt.savefig(file_name+'.png',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    plt.savefig(file_name+'.pdf',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    plt.savefig(file_name+'.svg',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='svg')

    # make results dataframe and save it to the disk
    df = pd.DataFrame(si_morris)
    file_name = os.path.join(args.data_dir, 'results')+'/morris_table_'+model_name[-3]+'_'+args.feature_type
    df.to_csv(file_name+'.csv')

def sobol_method(args, model, problem):

    X = saltelli.sample(problem, 100, calc_second_order=True)
    Y = model.predict(X)
    Y = np.nan_to_num(Y).flatten()
    si_sobol = sobol.analyze(problem, Y, print_to_console=False)

    Si_key = {k: si_sobol[k] for k in ['S1', 'S1_conf', 'ST', 'ST_conf']}
    Si_df = pd.DataFrame(Si_key, index=problem['names'])

    Si_df['names'] = problem['names']

    sortcol = 'ST'
    names_sorted = sort_Si(Si_df, 'names', sortcol)
    val_sorted = sort_Si(Si_df, 'S1', sortcol)
    err_sorted = sort_Si(Si_df, 'S1_conf', sortcol)

    val2_sorted = sort_Si(Si_df, 'ST', sortcol)
    err2_sorted = sort_Si(Si_df, 'ST_conf', sortcol)

    fig, ax = plt.subplots(1, 2, figsize=(20, 150), sharey=True)
    # plt.xlim([-0.5,0.5])
    # fig.set_size_inches(2,150)

    y_pos = np.arange(len(val_sorted))
    plot_names = names_sorted

    ax[0].barh(y_pos,
               val_sorted,
               xerr=err_sorted,
               align='center',
               ecolor='black',
               color='green')
    ax[0].set_yticks(y_pos)
    ax[0].set_yticklabels(plot_names)
    ax[0].set_xlabel('S1')
    ax[0].set_ylim(min(y_pos) - 1, max(y_pos) + 1)

    y_pos = np.arange(len(val_sorted))

    ax[1].barh(y_pos,
               val2_sorted,
               xerr=err2_sorted,
               align='center',
               ecolor='black',
               color='blue')
    ax[1].set_yticks(y_pos)
    ax[1].set_yticklabels(plot_names)
    ax[1].set_xlabel('ST')
    ax[1].set_ylim(min(y_pos) - 1, max(y_pos) + 1)

    model_name = args.model.split('/')[-1]
    file_name = os.path.join(args.data_dir, 'results') + '/sobol_hbar_plot_' + model_name[-3] + '_' + args.feature_type
    plt.savefig(file_name + '.png',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    plt.savefig(file_name + '.pdf',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    plt.savefig(file_name + '.svg',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='svg')

    # convert nan values to zeros
    si_sobol = nan_to_zero(si_sobol)

    result = {}  # create dictionary to store new
    result['S1'] = {k: float(v) for k, v in zip(problem["names"], si_sobol["S1"])}
    result['S1_conf'] = {k: float(v) for k, v in zip(problem["names"], si_sobol["S1_conf"])}
    result['S2'] = S2_to_dict(si_sobol['S2'], problem)
    result['S2_conf'] = S2_to_dict(si_sobol['S2_conf'], problem)
    result['ST'] = {k: float(v) for k, v in zip(problem["names"], si_sobol["ST"])}
    result['ST_conf'] = {k: float(v) for k, v in zip(problem["names"], si_sobol["ST_conf"])}

    # plot cord graph of parameters with top 30 ST values
    results = sort_result(result, on='ST', n=20)

    # plot cord graph
    cordgraphs(results)

    # make results dataframe and save it to the disk
    df1 = pd.DataFrame(problem['names'])
    df1.columns = ['name']
    df1['S1'] = si_sobol['S1']
    df1['S1_conf'] = si_sobol['S1_conf']
    df1['ST'] = si_sobol['ST']
    df1['ST_conf'] = si_sobol['ST_conf']
    df2 = pd.DataFrame(si_sobol['S2'])
    df2.columns = problem['names']
    df2.index = problem['names']
    df3 = pd.DataFrame(si_sobol['S2_conf'])
    df3.columns = problem['names']
    df3.index = problem['names']

    file_name = os.path.join(args.data_dir, 'results') + '/sobol_S1_ST_table_' + model_name[-3] + '_' + args.feature_type
    df1.to_csv(file_name + '.csv')
    file_name = os.path.join(args.data_dir, 'results') + '/sobol_S2_table_' + model_name[-3] + '_' + args.feature_type
    df2.to_csv(file_name + '.csv')
    file_name = os.path.join(args.data_dir, 'results') + '/sobol_S2_conf_table_' + model_name[-3] + '_' + args.feature_type
    df3.to_csv(file_name + '.csv')

def dgsm_method(args, model, problem):
    param_values = finite_diff.sample(problem, 100, delta=0.001)
    Y = model.predict(param_values)
    Y = np.nan_to_num(Y).flatten()
    Si_dgsm = dgsm.analyze(problem, param_values, Y, num_resamples=1000,
                           conf_level=0.95, print_to_console=False)

    # ---dgsm plot
    Si_key = {k: Si_dgsm[k] for k in ['vi', 'vi_std', 'dgsm', 'dgsm_conf']}
    Si_df = pd.DataFrame(Si_key, index=problem['names'])

    Si_df['names'] = problem['names']

    sortcol = 'vi'
    names_sorted = sort_Si(Si_df, 'names', sortcol)
    val_sorted = sort_Si(Si_df, 'vi', sortcol)
    err_sorted = sort_Si(Si_df, 'vi_std', sortcol)

    # sortcol = 'dgsm'
    # names_sorted = sort_Si(Si_df, 'names',sortcol)
    val2_sorted = sort_Si(Si_df, 'dgsm', sortcol)
    err2_sorted = sort_Si(Si_df, 'dgsm_conf', sortcol)

    fig, ax = plt.subplots(1, 2, figsize=(20, 150), sharey=True)
    # plt.xlim([-0.5,0.5])
    # fig.set_size_inches(2,150)

    y_pos = np.arange(len(val_sorted))
    plot_names = names_sorted

    ax[0].barh(y_pos,
               val_sorted,
               xerr=err_sorted,
               align='center',
               ecolor='black',
               color='green')

    ax[0].set_yticks(y_pos)
    ax[0].set_yticklabels(plot_names)
    ax[0].set_xlabel('vi')
    ax[0].set_ylim(min(y_pos) - 1, max(y_pos) + 1)

    y_pos = np.arange(len(val_sorted))

    ax[1].barh(y_pos,
               val2_sorted,
               xerr=err2_sorted,
               align='center',
               ecolor='black',
               color='blue')

    ax[1].set_yticks(y_pos)
    ax[1].set_yticklabels(plot_names)
    ax[1].set_xlabel('dgsm')
    ax[1].set_ylim(min(y_pos) - 1, max(y_pos) + 1)

    model_name = args.model.split('/')[-1]
    file_name = os.path.join(args.data_dir, 'results') + '/dgsm_hbar_plot_' + model_name[-3] + '_' + args.feature_type
    plt.savefig(file_name + '.png',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    plt.savefig(file_name + '.pdf',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    plt.savefig(file_name + '.svg',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='svg')

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = Si_dgsm['names']
    dgsm_sorted = sort_Si(Si_dgsm, 'dgsm', 'dgsm')
    labels_sorted = sort_Si(Si_dgsm, 'names', 'dgsm')
    dgsm_percentages = dgsm_sorted[-20:] / sum(dgsm_sorted[-20:]) * 100

    # make explode list
    explode_list = [0 for _ in range(dgsm_percentages.shape[0])]
    explode_list[np.argmax(dgsm_percentages)] = 0.1  # only explode the biggest slice

    fig, ax = plt.subplots(2, 1, figsize=(20, 20), sharey=True)

    theme = plt.get_cmap('bwr')
    ax[0].set_prop_cycle("color", [theme(1. * i / 20) for i in range(20)])

    ax[0].pie(dgsm_percentages, explode=explode_list, labels=labels_sorted[-20:],
              autopct='%1.1f%%', shadow=True, startangle=90,
              textprops={'fontsize': 6.5})

    ax[0].axis('equal')
    ax[0].title.set_text('dgsm')
    plt.subplots_adjust(hspace=0.2)

    vi_sorted = sort_Si(Si_dgsm, 'vi', 'vi')
    dgsm_vi_percentages = vi_sorted[-20:] / sum(vi_sorted[-20:]) * 100
    labels_sorted = sort_Si(Si_dgsm, 'names', 'vi')

    # make explode list
    explode_list = [0 for _ in range(dgsm_vi_percentages.shape[0])]
    explode_list[np.argmax(dgsm_vi_percentages)] = 0.1  # only explode the biggest slice

    ax[1].set_prop_cycle("color", [theme(1. * i / 20) for i in range(20)])

    ax[1].pie(dgsm_percentages, explode=explode_list, labels=labels_sorted[-20:],
              autopct='%1.1f%%', shadow=True, startangle=90,
              textprops={'fontsize': 6.5})

    ax[1].pie(dgsm_vi_percentages, explode=explode_list, labels=labels_sorted[-20:],
              autopct='%1.1f%%', shadow=True, startangle=90,
              textprops={'fontsize': 6.5})
    ax[1].axis('equal')
    ax[1].title.set_text('vi')

    model_name = args.model.split('/')[-1]
    file_name = os.path.join(args.data_dir, 'results') + '/dgsm_piechart_plot_' + model_name[-3] + '_' + args.feature_type
    plt.savefig(file_name + '.png',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    plt.savefig(file_name + '.pdf',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    plt.savefig(file_name + '.svg',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='svg')

    # make results dataframe and save it to the disk
    df = pd.DataFrame(Si_dgsm)
    file_name = os.path.join(args.data_dir, 'results') + '/dgsm_table_' + model_name[-3] + '_' + args.feature_type
    df.to_csv(file_name + '.csv')

def fast_method(args, model, problem):
    param_values = fast_sampler.sample(problem, 1000)
    Y = model.predict(param_values)
    Y = np.nan_to_num(Y).flatten()
    Si_fast = fast.analyze(problem, Y, print_to_console=False)

    # ---fast plot
    Si_key = {k: Si_fast[k] for k in ['S1', 'ST']}
    Si_df = pd.DataFrame(Si_key, index=problem['names'])

    Si_df['names'] = problem['names']

    sortcol = 'ST'
    names_sorted = sort_Si(Si_df, 'names', sortcol)
    val_sorted = sort_Si(Si_df, 'S1', sortcol)

    val2_sorted = sort_Si(Si_df, 'ST', sortcol)

    fig, ax = plt.subplots(1, 2, figsize=(20, 150), sharey=True)
    # plt.xlim([-0.5,0.5])
    # fig.set_size_inches(2,150)

    y_pos = np.arange(len(val_sorted))
    plot_names = names_sorted

    ax[0].barh(y_pos,
               val_sorted,
               align='center',
               ecolor='black',
               color='green')
    ax[0].set_yticks(y_pos)
    ax[0].set_yticklabels(plot_names)
    ax[0].set_xlabel('S1')
    ax[0].set_ylim(min(y_pos) - 1, max(y_pos) + 1)

    y_pos = np.arange(len(val_sorted))

    ax[1].barh(y_pos,
               val2_sorted,
               align='center',
               ecolor='black',
               color='blue')
    ax[1].set_yticks(y_pos)
    ax[1].set_yticklabels(plot_names)
    ax[1].set_xlabel('ST')
    ax[1].set_ylim(min(y_pos) - 1, max(y_pos) + 1)

    model_name = args.model.split('/')[-1]
    file_name = os.path.join(args.data_dir, 'results') + '/fast_hbar_plot_' + model_name[-3] + '_' + args.feature_type
    plt.savefig(file_name + '.png',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    plt.savefig(file_name + '.pdf',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    plt.savefig(file_name + '.svg',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='svg')

    # make results dataframe and save it to the disk
    df = pd.DataFrame(Si_fast)
    file_name = os.path.join(args.data_dir, 'results') + '/fast_table_' + model_name[-3] + '_' + args.feature_type
    df.to_csv(file_name + '.csv')

def rbd_fast_method(args, model, problem):
    X = latin.sample(problem, 100)
    Y = model.predict(X)
    Y = np.nan_to_num(Y).flatten()
    Si_rbd = rbd_fast.analyze(problem, X, Y, print_to_console=False)

    # ---fast-rbd plot

    Si_key = {k: Si_rbd[k] for k in ['S1']}
    Si_df = pd.DataFrame(Si_key, index=problem['names'])

    Si_df['names'] = problem['names']

    sortcol = 'S1'
    names_sorted = sort_Si(Si_df, 'names', sortcol)
    val_sorted = sort_Si(Si_df, 'S1', sortcol)

    fig, ax = plt.subplots(1, 1, figsize=(20, 150), sharey=True)

    y_pos = np.arange(len(val_sorted))
    plot_names = names_sorted

    ax.barh(y_pos,
            val_sorted,
            align='center',
            ecolor='black',
            color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_names)
    ax.set_xlabel('delta')
    ax.set_ylim(min(y_pos) - 1, max(y_pos) + 1)

    model_name = args.model.split('/')[-1]
    file_name = os.path.join(args.data_dir, 'results') + '/rbd_fast_hbar_plot_' + model_name[
        -3] + '_' + args.feature_type
    plt.savefig(file_name + '.png',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    plt.savefig(file_name + '.pdf',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    plt.savefig(file_name + '.svg',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='svg')

    # make results dataframe and save it to the disk
    df = pd.DataFrame(Si_rbd)
    file_name = os.path.join(args.data_dir, 'results') + '/rbd_fast_table_' + model_name[-3] + '_' + args.feature_type
    df.to_csv(file_name + '.csv')

def delta_method(args, model, problem):
    X = latin.sample(problem, 100)
    Y = model.predict(X)
    Y = np.nan_to_num(Y).flatten()
    Si_delta = delta.analyze(problem, X, Y, print_to_console=False)

    # ---Si_delta plot
    Si_key = {k: Si_delta[k] for k in ['delta', 'delta_conf', 'S1', 'S1_conf']}
    Si_df = pd.DataFrame(Si_key, index=problem['names'])

    Si_df['names'] = problem['names']

    sortcol = 'delta'
    names_sorted = sort_Si(Si_df, 'names', sortcol)
    val_sorted = sort_Si(Si_df, 'delta', sortcol)
    err_sorted = sort_Si(Si_df, 'delta_conf', sortcol)

    val2_sorted = sort_Si(Si_df, 'S1', sortcol)
    err2_sorted = sort_Si(Si_df, 'S1_conf', sortcol)

    fig, ax = plt.subplots(1, 2, figsize=(20, 150), sharey=True)

    y_pos = np.arange(len(val_sorted))
    plot_names = names_sorted

    ax[0].barh(y_pos,
               val_sorted,
               xerr=err_sorted,
               align='center',
               ecolor='black',
               color='green')
    ax[0].set_yticks(y_pos)
    ax[0].set_yticklabels(plot_names)
    ax[0].set_xlabel('delta')
    ax[0].set_ylim(min(y_pos) - 1, max(y_pos) + 1)

    y_pos = np.arange(len(val_sorted))

    ax[1].barh(y_pos,
               val2_sorted,
               xerr=err2_sorted,
               align='center',
               ecolor='black',
               color='blue')
    ax[1].set_yticks(y_pos)
    ax[1].set_yticklabels(plot_names)
    ax[1].set_xlabel('S1')
    ax[1].set_ylim(min(y_pos) - 1, max(y_pos) + 1)

    model_name = args.model.split('/')[-1]
    file_name = os.path.join(args.data_dir, 'results') + '/delta_hbar_plot_' + model_name[
        -3] + '_' + args.feature_type
    plt.savefig(file_name + '.png',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    plt.savefig(file_name + '.pdf',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    plt.savefig(file_name + '.svg',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='svg')

    # make results dataframe and save it to the disk
    df = pd.DataFrame(Si_delta)
    file_name = os.path.join(args.data_dir, 'results') + '/delta_table_' + model_name[-3] + '_' + args.feature_type
    df.to_csv(file_name + '.csv')


def main(args):
    # load feature labels
    feature_labels = np.load(os.path.join(args.data_dir,
                                          args.feature_labels),
                             fix_imports=True,
                             encoding='latin1')[1:]

    # load feature real names
    realnames = np.load(os.path.join(args.data_dir, "data/featurelabels_real.npy"),
                        allow_pickle=True,
                        fix_imports=True,
                        encoding='latin1').item()

    # rename feature labels
    feature_labels_new = []
    for f in feature_labels:
        f = f.decode('utf-8')
        if (f in realnames.keys()):
            rn = realnames[f].replace(" ", "_")
            feature_labels_new.append(rn)
        else:
            feature_labels_new.append(f.replace(" ", "_"))

    # define morris problem
    features_num = len(feature_labels_new)
    problem = {
        'num_vars': features_num,
        'names': list(feature_labels_new),
        'bounds': [[0.0, 1.0]] * features_num  # standard scaled
    }

    # load model
    model = keras.models.load_model(os.path.join(args.data_dir, args.model))

    if args.method_name == 'morris':
        morris_method(args, model, problem)
    elif args.method_name == 'sobol':
        sobol_method(args, model, problem)
    elif args.method_name == 'dgsm':
        dgsm_method(args, model, problem)
    elif args.method_name == 'fast':
        fast_method(args, model, problem)
    elif args.method_name == 'rbd_fast':
        rbd_fast_method(args, model, problem)
    elif args.method_name == 'delta':
        delta_method(args, model, problem)
    else:
        raise Exception("select one of the available methods: morris, sobol, dgsm,"
                        "fast, rbd_fast, delta.")

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description='Morris analysis.')
    # Add the arguments
    parser.add_argument('--method_name',
                        default='morris',
                        type=str,
                        help='name of the method')
    parser.add_argument('--data_dir',
                        default='/home/reza/Documents/PROMIMOOC/tata_script',
                        type=str,
                        help='path to the data files')
    parser.add_argument('--model', default='models/dnn_v1_hgroup_2.h5',
                        type=str,
                        help='trained model file path')
    parser.add_argument('--feature_labels',
                        default='data/featurelabels_v7.npy',
                        type=str,
                        help='feature labels file name')
    parser.add_argument('--trajectory_number',
                        default=10,
                        type=int,
                        help='number of trajectories')
    parser.add_argument('--sample_number',
                        default=100,
                        type=int,
                        help='number of samples')
    parser.add_argument('--feature_type',
                        default='signal_oxide',
                        type=str,
                        help='type of features: can be "signal_oxide" or "combined"')
    # Execute the parse_args() method
    args = parser.parse_args()
    main(args)

