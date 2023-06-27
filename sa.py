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

def cordgraphs(SAresults):
    # Get list of parameters
    parameters = list(SAresults['S1'].keys())
    # Set min index value, for the effects to be considered significant
    index_significance_value = 0.01

    '''
    Define some general layout settings.
    '''
    node_size_min = 15  # Max and min node size
    node_size_max = 30
    border_size_min = 1  # Max and min node border thickness
    border_size_max = 8
    edge_width_min = 1  # Max and min edge thickness
    edge_width_max = 10
    edge_distance_min = 0.1  # Max and min distance of the edge from the center of the circle
    edge_distance_max = 0.6  # Only applicable to the curved edges

    '''
    Set up some variables and functions that will facilitate drawing circles and 
    moving items around.
    '''
    # Define circle center and radius
    center = [0.0, 0.0]
    radius = 1.0

    # Function to get distance between two points
    def distance(p1, p2):
        return np.sqrt(((p1 - p2) ** 2).sum())

    # Function to get middle point between two points
    def middle(p1, p2):
        return (p1 + p2) / 2

    # Function to get the vertex of a curve between two points
    def vertex(p1, p2, c):
        m = middle(p1, p2)
        curve_direction = c - m
        return m + curve_direction * (edge_distance_min + edge_distance_max * (1 - distance(m, c) / distance(c, p1)))

    # Function to get the angle of the node from the center of the circle
    def angle(p, c):
        # Get x and y distance of point from center
        [dx, dy] = p - c
        if dx == 0:  # If point on vertical axis (same x as center)
            if dy > 0:  # If point is on positive vertical axis
                return np.pi / 2.
            else:  # If point is on negative vertical axis
                return np.pi * 3. / 2.
        elif dx > 0:  # If point in the right quadrants
            if dy >= 0:  # If point in the top right quadrant
                return np.arctan(dy / dx)
            else:  # If point in the bottom right quadrant
                return 2 * np.pi + np.arctan(dy / dx)
        elif dx < 0:  # If point in the left quadrants
            return np.pi + np.arctan(dy / dx)

    '''
    First, set up graph with all parameters as nodes and draw all second order (S2)
    indices as edges in the network. For every S2 index, we need a Source parameter,
    a Target parameter, and the Weight of the line, given by the S2 index itself. 
    '''
    combs = [list(c) for c in list(itertools.combinations(parameters, 2))]

    Sources = list(list(zip(*combs))[0])
    Targets = list(list(zip(*combs))[1])
    # Sometimes computing errors produce negative Sobol indices. The following reads
    # in all the indices and also ensures they are between 0 and 1.
    Weights = [max(min(x, 1), 0) for x in [SAresults['S2'][Sources[i]][Targets[i]] for i in range(len(Sources))]]
    Weights = [0 if x < index_significance_value else x for x in Weights]

    # Set up graph
    G = nx.Graph()
    # Draw edges with appropriate weight
    for s, t, weight in zip(Sources, Targets, Weights):
        G.add_edges_from([(s, t)], w=weight)

    # Generate dictionary of node postions in a circular layout
    Pos = nx.circular_layout(G)

    '''
    Normalize node size according to first order (S1) index. First, read in S1 indices,
    ensure they're between 0 and 1 and normalize them within the max and min range
    of node sizes.
    Then, normalize edge thickness according to S2. 
    '''
    # Node size
    first_order = [max(min(x, 1), 0) for x in [SAresults['S1'][key] for key in SAresults['S1']]]
    first_order = [0 if x < index_significance_value else x for x in first_order]
    node_size = [node_size_min * (1 + (node_size_max - node_size_min) * k / max(first_order)) for k in first_order]
    # Node border thickness
    total_order = [max(min(x, 1), 0) for x in [SAresults['ST'][key] for key in SAresults['ST']]]
    total_order = [0 if x < index_significance_value else x for x in total_order]
    border_size = [border_size_min * (1 + (border_size_max - border_size_min) * k / max(total_order)) for k in
                   total_order]
    # Edge thickness
    edge_width = [edge_width_min * ((edge_width_max - edge_width_min) * k / max(Weights)) for k in Weights]

    '''
    Draw network. This will draw the graph with straight lines along the edges and 
    across the circle. 
    '''
    # nx.draw_networkx_nodes(G, Pos, node_size=node_size, node_color='#98B5E2',
    #                        edgecolors='#1A3F7A', linewidths = border_size)
    # nx.draw_networkx_edges(G, Pos, width=edge_width, edge_color='#2E5591', alpha=0.7)
    # names = nx.draw_networkx_labels(G, Pos, font_size=12, font_color='#0B2D61', font_family='sans-serif')
    # for node, text in names.items():
    #     position = (radius*1.1*np.cos(angle(Pos[node],center)), radius*1.1*np.sin(angle(Pos[node],center)))
    #     text.set_position(position)
    #     text.set_clip_on(False)
    # plt.gcf().set_size_inches(20, 20) # Make figure a square
    # plt.axis('off')

    '''
     We can now draw the network with curved lines along the edges and across the circle.
     Calculate all distances between 1 node and all the others (all distances are 
     the same since they're in a circle). We'll need this to identify the curves 
     we'll be drawing along the perimeter (i.e. those that are next to each other).
     '''
    min_distance = round(min([distance(Pos[list(G.nodes())[0]], Pos[n]) for n in list(G.nodes())[1:]]), 1)

    # Figure to generate the curved edges between two points
    def xy_edge(p1, p2):  # Point 1, Point 2
        m = middle(p1, p2)  # Get middle point between the two nodes
        # If the middle of the two points falls very close to the center, then the
        # line between the two points is simply straight
        if distance(m, center) < 1e-6:
            xpr = np.linspace(p1[0], p2[0], 10)
            ypr = np.linspace(p1[1], p2[1], 10)
        # If the distance between the two points is the minimum (i.e. they are next
        # to each other), draw the edge along the perimeter
        elif distance(p1, p2) <= min_distance:
            # Get angles of two points
            p1_angle = angle(p1, center)
            p2_angle = angle(p2, center)
            # Check if the points are more than a hemisphere apart
            if max(p1_angle, p2_angle) - min(p1_angle, p2_angle) > np.pi:
                radi = np.linspace(max(p1_angle, p2_angle) - 2 * np.pi, min(p1_angle, p2_angle))
            else:
                radi = np.linspace(min(p1_angle, p2_angle), max(p1_angle, p2_angle))
            xpr = radius * np.cos(radi) + center[0]
            ypr = radius * np.sin(radi) + center[1]
            # Otherwise, draw curve (parabola)
        else:
            edge_vertex = vertex(p1, p2, center)
            a = distance(edge_vertex, m) / ((distance(p1, p2) / 2) ** 2)
            yp = np.linspace(-distance(p1, p2) / 2, distance(p1, p2) / 2, 100)
            xp = a * (yp ** 2)
            xp += distance(center, edge_vertex)
            theta_m = angle(middle(p1, p2), center)
            xpr = np.cos(theta_m) * xp - np.sin(theta_m) * yp
            ypr = np.sin(theta_m) * xp + np.cos(theta_m) * yp
            xpr += center[0]
            ypr += center[1]
        return xpr, ypr

    '''
    Draw network. This will draw the graph with curved lines along the edges and 
    across the circle. 
    '''
    fig = plt.figure(figsize=(30, 25))
    ax = fig.add_subplot(1, 1, 1)
    for i, e in enumerate(G.edges()):
        x, y = xy_edge(Pos[e[0]], Pos[e[1]])
        ax.plot(x, y, '-', c='#2E5591', lw=edge_width[i], alpha=0.7)
    for i, n in enumerate(G.nodes()):
        ax.plot(Pos[n][0], Pos[n][1], 'o', c='#98B5E2', markersize=node_size[i] / 5, markeredgecolor='#1A3F7A',
                markeredgewidth=border_size[i] * 1.15)

    for i, text in enumerate(G.nodes()):
        if node_size[i] < 100:
            position = (
            radius * 1.05 * np.cos(angle(Pos[text], center)), radius * 1.05 * np.sin(angle(Pos[text], center)))
        else:
            position = (
            radius * 1.01 * np.cos(angle(Pos[text], center)), radius * 1.01 * np.sin(angle(Pos[text], center)))
        plt.annotate(text, position, fontsize=12, color='#0B2D61', family='sans-serif')
    ax.axis('off')
    ax.set_title('Cord graph for parameters with top {} $ST$ values'.format(len(SAresults['S1'].keys())))
    fig.tight_layout()
    model_name = args.model.split('/')[-1]
    file_name = os.path.join(args.data_dir, 'results') + '/sobol_cord_plot_' + model_name[-3] + '_' + args.feature_type
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

