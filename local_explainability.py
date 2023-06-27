from lime import lime_tabular
import os
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, \
    precision_recall_curve, plot_precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import shap
import argparse
import seaborn as sns

def rf_model(X, y, args):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = ExtraTreesClassifier(n_estimators=args.n_estimators,
                                random_state=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    data = [X_train, X_test, y_train, y_test]
    return data, model, pred

def lime_method(args, model, data, feature_labels_new):
    X_train = data[0]
    X_test = data[1]
    explainer = lime_tabular.LimeTabularExplainer(X_train,
                                                  feature_names=feature_labels_new,
                                                  class_names=args.defect_family,
                                                  discretize_continuous=True)
    # i = np.random.randint(0, X_test.shape[0])
    i = args.sample_number
    exp = explainer.explain_instance(X_test[i], model.predict_proba,
                                     num_features=X_test.shape[1], top_labels=1)

    # vals = [x[1] for x in exp]
    # names = [x[0] for x in exp]

    fig, vals, names = exp.as_pyplot_figure(label=0)
    vals.reverse()
    names.reverse()
    df = pd.DataFrame()
    df['name'] = names
    df['lime_value'] = vals
    file_name = args.results_dir + '/hbar_plot_lime_' + args.defect_family[0]
    fig.savefig(file_name + '_sample_{}.png'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    fig.savefig(file_name + '_sample_{}.pdf'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    fig.savefig(file_name + '_sample_{}.svg'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='svg')

    file_name = args.results_dir + '/lime_table_' + args.defect_family[0]
    df.to_csv(file_name + '.csv')

def shapley_method(args, model, data, feature_labels_new, pred):
    X_test = data[1]
    explainer = shap.TreeExplainer(model)
    n_samples = 20
    # select a set of random set samples to take an expectation over
    data_for_prediction = X_test[np.random.choice(X_test.shape[0],
                                                  n_samples, replace=False)]
    print('Evaluating shap values...')

    df1 = pd.DataFrame(data_for_prediction, columns=feature_labels_new)
    df2 = pd.DataFrame(data_for_prediction, columns=['F {}'.format(i)
                                                     for i in range(1, X_test.shape[1] + 1)])
    shap_values = explainer.shap_values(data_for_prediction)
    # i = np.random.randint(0, n_samples)
    i = args.sample_number
    print('Plot force plot ...')
    fig = shap.force_plot(explainer.expected_value[1], shap_values[1][i], df2.iloc[i], show=False, matplotlib=True)
    file_name = args.results_dir + '/force_plot_shap_' + args.defect_family[0]
    fig.savefig(file_name + '_sample_{}.svg'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='svg')
    fig.savefig(file_name + '_sample_{}.png'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    fig.savefig(file_name + '_sample_{}.pdf'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    plt.close('all')

    fig = shap.force_plot(explainer.expected_value[1], shap_values[1], df2, show=False, link='logit')
    file_name = args.results_dir + '/force_plot_all_shap_' + args.defect_family[0]
    shap.save_html(file_name+'.html', fig)
    plt.close('all')

    print('Plot summary plot ...')
    max_display=30
    shap.summary_plot(shap_values[1], df1, max_display=max_display, show=False)
    fig = plt.gcf()
    file_name = args.results_dir + '/summary_plot_shap_' + args.defect_family[0]
    fig.savefig(file_name + '_sample_{}.svg'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png'
                )
    fig.savefig(file_name + '_sample_{}.png'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    fig.savefig(file_name + '_sample_{}.pdf'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    plt.close('all')

    print('Plot summary bar plot ...')
    shap.summary_plot(shap_values[1], df1, plot_type='bar', max_display=max_display, show=False)
    fig = plt.gcf()
    file_name = args.results_dir + '/summary_bar_plot_shap_' + args.defect_family[0]
    fig.savefig(file_name + '_sample_{}.svg'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='svg')
    fig.savefig(file_name + '_sample_{}.png'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    fig.savefig(file_name + '_sample_{}.pdf'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    plt.close('all')

    source_feature = feature_labels_new[0]
    interaction_feature = feature_labels_new[1]

    shap.dependence_plot(source_feature, shap_values[1], df1,
                         interaction_index=interaction_feature, show=False)
    fig = plt.gcf()
    file_name = args.results_dir + '/dependence_plot_shap_' + args.defect_family[0]
    fig.savefig(file_name + '_source_{}_interaction_{}.svg'.format(source_feature, interaction_feature),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='svg')
    fig.savefig(file_name + '_sample_{}.png'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    fig.savefig(file_name + '_sample_{}.pdf'.format(i),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    plt.close('all')

    print("Plot interaction heat map...")
    # shap_interaction_values = explainer.shap_interaction_values(data_for_prediction)
    # shap_interaction_values = shap_interaction_values[1]
    # shap_interaction_values = np.mean(shap_interaction_values, axis=0)
    # ax = sns.heatmap(shap_interaction_values)
    # fig = ax.get_figure()
    # fig.savefig('heatmap.jpg')
    # plt.close('all')

    print('Plot decision plot...')
    shap_values = explainer.shap_values(df1.values)
    # First 31st features' shap values for the first sample on class 1
    n_display_range = 31
    shap.decision_plot(explainer.expected_value[1], shap_values[1][0], df1,
                       feature_display_range=slice(None, -n_display_range, -1), show=False)
    fig = plt.gcf()
    file_name = args.results_dir + '/decision_plot_shap_' + args.defect_family[0]
    fig.savefig(file_name + '_display_range_{}.svg'.format(n_display_range),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='svg')
    fig.savefig(file_name + '_display_range_{}.png'.format(n_display_range),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='png')
    fig.savefig(file_name + '_display_range_{}.pdf'.format(n_display_range),
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=200,
                format='pdf')
    plt.close('all')


def main(args):
    # load feature labels
    feature_labels = np.load(os.path.join(args.data_dir,
                                          args.feature_labels),
                             fix_imports=True,
                             encoding='latin1')[1:]

    # load feature real names
    realnames = np.load(os.path.join(args.data_dir, "featurelabels_real.npy"),
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

    features = np.load(os.path.join(args.data_dir, args.features_file),
                       allow_pickle=True, fix_imports=True, encoding='latin1')[:, 1:]
    for f in range(features.shape[0]):
        features[f, 3] = float(features[f, 3])
    features = features.astype(np.float16)
    X = normalize(np.nan_to_num(features),
                              norm='l2', axis=0, copy=True, return_norm=False)

    targets = np.load(os.path.join(args.data_dir, args.targets_file))

    defects_family = ["AFDW", "BLRND", "OHV", "OXOVR", "OXSLB",
                      "TIGST", "WALSV", "LUS", "AFDW6", "AFDW7", "STNGW"]

    ind = defects_family.index(args.defect_family[0])

    # if args.defect_family not in defects_family:
    #     raise Exception('The input defect family does not exist!')
    # else:
    #     ind = defects_family.index(args.defect_family)

    y = np.zeros_like(targets)
    y[targets != 0] = 1
    y = y[:, ind]

    # load model
    data, model, pred = rf_model(X, y, args)

    if args.method_name == 'lime':
        lime_method(args, model, data, feature_labels_new)
    elif args.method_name == 'shapley':
        shapley_method(args, model, data, feature_labels_new, pred)
    else:
        raise Exception("select one of the available methods: lime or shapley")


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Morris analysis.')
    # Add the arguments
    parser.add_argument('--method_name',
                        default='lime',
                        type=str,
                        help='name of the method')
    parser.add_argument('--results_dir',
                        default='/home/reza/Documents/PROMIMOOC/tata_script/results',
                        type=str,
                        help='path to the results files')
    parser.add_argument('--data_dir',
                        default='/home/reza/Documents/PROMIMOOC/tata_script/data',
                        type=str,
                        help='path to the data files')
    parser.add_argument('--features_file',
                        default='coilfeatures_v7_6.npy',
                        type=str,
                        help='name of the coil features file')
    parser.add_argument('--targets_file',
                        default='coiltargets_v7_6.npy',
                        type=str,
                        help='name of the coil targets file')
    parser.add_argument('--model', default='models/dnn_v1_hgroup_2.h5',
                        type=str,
                        help='trained model file path')
    parser.add_argument('--feature_labels',
                        default='featurelabels_v7.npy',
                        type=str,
                        help='feature labels file name')
    parser.add_argument('--sample_number',
                        default=0,
                        type=int,
                        help='sample number to generate SA results')
    parser.add_argument('--defect_family',
                        default=['AFDW'],
                        type=str,
                        help='type of family defects: "AFDW","BLRND",'
                             '"OHV","OXOVR","OXSLB","TIGST","WALSV",'
                             '"LUS","AFDW6","AFDW7","STNGW"')
    parser.add_argument('--feature_type',
                        default='signal_oxide',
                        type=str,
                        help='type of features: can be "signal_oxide" or "combined"')
    parser.add_argument('--n_estimators',
                        default=5,
                        type=int,
                        help='number of random forest estimators')
    # Execute the parse_args() method
    args = parser.parse_args()
    main(args)
