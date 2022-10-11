#
# MIT License
#
# Copyright (c) 2022.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import pathlib
import string
import matplotlib
matplotlib.use('TKAgg')

import optuna
import shap.plots
from optuna.integration.shap import ShapleyImportanceEvaluator
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial import TrialState
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from shap import TreeExplainer
from matplotlib import pyplot as plt


def load_study(model='SARD', problem='mortality'):
    storage = optuna.storages.get_storage(
        f'sqlite:///grid_search_publication/{model}/grid_search_{model}_{problem}_ultimate_revision.db')
    study = optuna.load_study(study_name=storage.get_all_study_summaries(True)[0].study_name, storage=storage)
    return study


def calculate_shap(study):
    evaluator = ShapleyImportanceEvaluator()

    if model == 'SARD':
        all_params = list(study.best_params.keys())
        params = [p for p in all_params if p != 'lr']
    else:
        params = list(study.best_params.keys())
    evaluator._backend_evaluator.evaluate(study=study, params=params)

    r_squared = evaluator._backend_evaluator._forest.score(evaluator._backend_evaluator._trans_params,
                                                           evaluator._backend_evaluator._trans_values)
    print(f'R squared of model is; {r_squared:.2f}')
    evaluator._explainer = TreeExplainer(evaluator._backend_evaluator._forest,
                                         feature_names=evaluator._backend_evaluator._param_names)
    shap_values = evaluator._explainer(evaluator._backend_evaluator._trans_params)

    return shap_values


def plot_shap(shap_values, problem='mortality', save=True):
    shap.plots.beeswarm(shap_values)
    plt.gca().set_xlabel('SHAP value (impact on model output)', fontweight='bold', fontsize=18)
    plt.gcf().axes[1].set_ylabel('Hyperparameter value', fontweight='bold', fontsize=14)
    plt.gcf().axes[1].set_yticklabels(labels=['Low', 'High'], fontweight='bold', fontsize=14)
    plt.gca().set_yticklabels(plt.gca().get_ymajorticklabels(), fontweight='bold', fontsize=14)
    plt.title(f'{problem.capitalize()}', fontweight='bold', fontsize=18)
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(f'./grid_search_publication/plots/{model}_{problem}_SHAP.svg')
    plt.close()


def interpolate_zmap(zmap, contour_plot_num):
    # implements interpolation formulation used in Plotly
    # to interpolate heatmaps and contour plots
    # https://github.com/plotly/plotly.js/blob/master/src/traces/heatmap/interp2d.js#L30
    # citing their doc:
    #
    # > Fill in missing data from a 2D array using an iterative
    # > poisson equation solver with zero-derivative BC at edges.
    # > Amazingly, this just amounts to repeatedly averaging all the existing
    # > nearest neighbors
    #
    # Plotly's algorithm is equivalent to solve the following linear simultaneous equation.
    # It is discretization form of the Poisson equation.
    #
    #     z[x, y] = zmap[(x, y)]                                  (if zmap[(x, y)] is given)
    # 4 * z[x, y] = z[x-1, y] + z[x+1, y] + z[x, y-1] + z[x, y+1] (if zmap[(x, y)] is not given)

    a_data = []
    a_row = []
    a_col = []
    b = np.zeros(contour_plot_num ** 2)
    for x in range(contour_plot_num):
        for y in range(contour_plot_num):
            grid_index = y * contour_plot_num + x
            if (x, y) in zmap:
                a_data.append(1)
                a_row.append(grid_index)
                a_col.append(grid_index)
                b[grid_index] = zmap[(x, y)]
            else:
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if 0 <= x + dx < contour_plot_num and 0 <= y + dy < contour_plot_num:
                        a_data.append(1)
                        a_row.append(grid_index)
                        a_col.append(grid_index)
                        a_data.append(-1)
                        a_row.append(grid_index)
                        a_col.append(grid_index + dy * contour_plot_num + dx)

    z = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix((a_data, (a_row, a_col))), b)

    return z.reshape((contour_plot_num, contour_plot_num))


def create_zmap(x_values, y_values, z_values, xi, yi):
    # creates z-map from trial values and params.
    # z-map is represented by hashmap of coordinate and trial value pairs
    #
    # coordinates are represented by tuple of integers, where the first item
    # indicates x-axis index and the second item indicates y-axis index
    # and refer to a position of trial value on irregular param grid
    #
    # since params were resampled either with linspace or logspace
    # original params might not be on the x and y axes anymore
    # so we are going with close approximations of trial value positions
    zmap = dict()
    for x, y, z in zip(x_values, y_values, z_values):
        xindex = int(np.argmin(np.abs(xi - x)))
        yindex = int(np.argmin(np.abs(yi - y)))
        zmap[(xindex, yindex)] = z

    return zmap


def create_grid(x_values, y_values, z_values, contour_point_num=200, axes_padding_ratio=5e-2, ylog=True, xlog=True):
    x_values_max = max(x_values)
    x_values_min = min(x_values)
    y_values_max = max(y_values)
    y_values_min = min(y_values)

    if xlog:
        padding_x = (np.log10(x_values_max) - np.log10(x_values_min)) * axes_padding_ratio
        x_values_min = np.power(10, np.log10(x_values_min) - padding_x)
        x_values_max = np.power(10, np.log10(x_values_max) + padding_x)
        xi = np.logspace(np.log10(x_values_min), np.log10(x_values_max), contour_point_num)
    else:
        padding_x = (x_values_max - x_values_min) * axes_padding_ratio
        x_values_min -= padding_x
        x_values_max += padding_x
        xi = np.linspace(x_values_min, x_values_max, contour_point_num)
    if ylog:
        padding_y = (np.log10(y_values_max) - np.log10(y_values_min)) * axes_padding_ratio
        y_values_min = np.power(10, np.log10(y_values_min) - padding_y)
        y_values_max = np.power(10, np.log10(y_values_max) + padding_y)
        yi = np.logspace(np.log10(y_values_min), np.log10(y_values_max), contour_point_num)
    else:
        padding_y = (y_values_max - y_values_min) * axes_padding_ratio
        y_values_min -= padding_y
        y_values_max += padding_y
        yi = np.linspace(y_values_min, y_values_max, contour_point_num)
    zmap = create_zmap(x_values, y_values, z_values, xi, yi)
    zi = interpolate_zmap(zmap, contour_point_num)

    return xi, yi, zi, [x_values_min, x_values_max], [y_values_min, y_values_max]


def extract_info_study(study, params):
    study_df = study.trials_dataframe()
    x_name = params[0]
    y_name = params[1]

    filtered_df = study_df[(study_df.state == 'COMPLETE') &
                           (np.isfinite(study_df.value)) &
                           (np.isfinite(study_df[f'params_{x_name}'])) &
                           (np.isfinite(study_df[f'params_{y_name}']))]
    z_values = filtered_df.value.values
    y_values = filtered_df[f'params_{y_name}'].values
    x_values = filtered_df[f'params_{x_name}'].values

    return z_values, y_values, x_values


def plot_contour(x_name, y_name, x_values, y_values, xi, yi, zi, ax, xlog, ylog,
                 x_range, y_range):
    cmap = plt.get_cmap('Blues_r')
    # plot:
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(x_name, fontweight='bold', fontsize=14, labelpad=1)
    ax.set_ylabel(y_name, fontweight='bold', fontsize=14, labelpad=1)
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    # make levels indicate an auc improvement of 0.5
    resolution = 0.5
    delta_z = zi.max() - zi.min()
    levels = int(np.ceil(delta_z/resolution * 100))

    if levels > 20:
        levels = 20

    ax.contour(xi, yi, zi, levels=levels, linewidths=0.5, colors='k')
    cs = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap.reversed())
    ax.scatter(x_values, y_values, marker="o", c="black", s=20, edgecolors="grey", linewidth=2.0)
    return cs


def is_log_scale(trials, param):
    for trial in trials:
        if param in trial.params:
            dist = trial.distributions[param]
            if isinstance(dist, (FloatDistribution, IntDistribution)):
                if dist.log:
                    return True
    return False


def is_numerical(trials, param):
    return all(
        (isinstance(t.params[param], int) or isinstance(t.params[param], float))
        and not isinstance(t.params[param], bool)
        for t in trials
        if param in t.params
    )


def create_contour_plot(model, problem, save=True):
    if model == 'SARD':
        x_variables = ['distill_lr', 'embedding_per_head', 'alpha', 'attn_depth']
        y_variables = ['finetune_lr', 'num_hidden', 'weight_decay', 'num_heads']
    elif model == 'Transformer':
        x_variables = ['lr', 'embedding_per_head', 'attn_depth', 'ffn_dropout']
        y_variables = ['weight_decay', 'num_hidden', 'num_heads', 'attention_dropout']
    elif model == 'RETAIN':
        x_variables = ['lr', 'dim_emb', 'dim_alpha', 'dropout_context']
        y_variables = ['weight_decay', 'num_layers', 'dim_beta', 'dropout_emb']
    elif model == 'GNN':
        x_variables = ['lr', 'dim_embedding', 'num_heads', 'num_layers']
        y_variables = ['weight_decay', 'num_layers', 'attention_dropout', 'dropout']
    else:
        RuntimeError(f'unknown model: {model}')


    study = load_study(model, problem)
    fig = plt.figure(figsize=[10, 8], constrained_layout=True)
    subfigure = fig.subfigures(1, 1)
    axes = subfigure.subplots(nrows=2, ncols=2)
    for i, ax in enumerate(axes.flat):
        x, y = x_variables[i], y_variables[i]
        z_values, y_values, x_values = extract_info_study(study=study, params=[x, y])
        xi, yi, zi, [x_values_min, x_values_max], [y_values_min, y_values_max] = create_grid(x_values=x_values,
                                                                                             y_values=y_values,
                                                                                             z_values=z_values,
                                                                                             contour_point_num=200,
                                                                                             axes_padding_ratio=5e-2,
                                                                                             xlog=is_log_scale(
                                                                                                 study.trials, x),
                                                                                             ylog=is_log_scale(
                                                                                                 study.trials, y))
        cs = plot_contour(x, y, x_values, y_values, xi, yi, zi, ax, is_log_scale(study.trials, x),
                          is_log_scale(study.trials, y), [x_values_min, x_values_max], [y_values_min, y_values_max])
        ax.text(-0.1, 1.05, f'{string.ascii_lowercase[i]})', transform=ax.transAxes, size=14, weight='bold')
    axcb = subfigure.colorbar(cs, ax=axes)
    axcb.set_label('AUC', fontweight='bold', fontsize=14)
    subfigure.suptitle(f'{model} {problem.capitalize()}', fontweight='bold', fontsize=18)
    plt.show()
    if save:
        plt.savefig(f'./grid_search_publication/plots/{model}_{problem}_contour_plot.svg')
    plt.close()


if __name__ == '__main__':
    model = 'SARD'
    problems = ['mortality']

    for problem in problems:
        study = load_study(model, problem)
        shap_values = calculate_shap(study)
        plot_shap(shap_values, problem, save=True)

    for model in ['RETAIN', 'SARD', 'Transformer', 'GNN']:
        for problem in problems:
            create_contour_plot(model, problem, save=True)

    # importances = optuna.importance.get_param_importances(study=study, evaluator=evaluator)

    # fig = optuna.visualization.matplotlib.plot_param_importances(study)
    # fig.show()

    # ax = optuna.visualization.matplotlib.plot_contour(study, params=['embedding_per_head', 'alpha'])
    # optuna.visualization.matplotlib.plot_contour(study, params=['num_hidden', 'attn_depth'])
