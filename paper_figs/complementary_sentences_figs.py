from pathlib import Path
import numpy as np
import pickle
from transformers import logging
import matplotlib.pyplot as plt
from featureinterp.record import ComplementaryRecordSource
from scripts import experiment


logging.set_verbosity_error()
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

plots_dir = Path('plots', 'complementary_sentences')
plots_dir.mkdir(parents=True, exist_ok=True)

colors = ['#120789', '#120789', '#FA9E3B', '#FA9E3B', '#C23D80', '#C23D80']

scores: list[list[float]] = []
polys: list[list[float]] = []
comps: list[list[float]] = []
for explainer_type in ['one_shot', 'tree']:
    exp_name: str = f'{explainer_type}_50_features_v4'
    with open(f'results/score_vs_layer/experiment_results_{exp_name}.pkl', 'rb') as f:
        layer_results: dict[str, list[list[experiment.ExperimentResult]]] = pickle.load(f)

    x: list[float] = []
    y_lst: list[list[float]] = []
    fp_lst: list[list[float]] = []
    poly: list[float] = []
    avgcomp: list[float] = []

    for layer_experiments in layer_results[ComplementaryRecordSource.RANDOM.value]:
        layer_index = layer_experiments[0].layer_index
        x.append(layer_index)
        mean_complexities = [np.nanmean(exp.explanation_complexities) for exp in layer_experiments
                            if exp.explanation_complexities != []]
        avgcomp.append(np.nanmean(mean_complexities))
        max_complexities = [np.nanmax(exp.explanation_complexities) for exp in layer_experiments
                            if exp.explanation_complexities != []]
        mean_poly = [np.mean(len(exp.explanation.components)) for exp in layer_experiments]
        poly.append(np.mean(mean_poly))
    polys.append(poly)
    comps.append(avgcomp)

    for strategy in [
        ComplementaryRecordSource.RANDOM,
        ComplementaryRecordSource.RANDOM_NEGATIVE,
        ComplementaryRecordSource.SIMILAR,
        ComplementaryRecordSource.SIMILAR_NEGATIVE,
    ]:
        y: list[float] = []
        fp: list[float] = []
        for layer_experiments in layer_results[strategy.value]:
            layer_scores = [exp.score for exp in layer_experiments]
            y.append(np.nanmean(layer_scores))
            layer_fp_rates = [exp.fp_rate * 32 for exp in layer_experiments]
            fp.append(np.nanmean(layer_fp_rates))
        y_lst.append(y)
        fp_lst.append(fp)
    scores.append(y_lst[3])  # similar non-activating complementary sentences

    explainer_name = 'one-shot' if 'one_shot' in exp_name else 'tree'
    complementary_names = [
        'Random',
        'Random (non-activating)',
        'Similar',
        'Similar (non-activating)',
    ]

    plt.figure(figsize=(4 / 1.2, 3 / 1.2))
    for i in range(4):
        plt.plot(x, y_lst[i], label=complementary_names[i], color=colors[i],
                alpha=0.3 if i % 2 == 0 else 0.8,
                linewidth=1.5 if i % 2 == 0 else 2,
                linestyle='--' if i % 2 == 0 else '-')
    plt.xlabel('Layer')
    plt.ylabel('Correlation score')
    plt.xlim((0, 40))
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.savefig(plots_dir / f'{exp_name}_score.pdf', bbox_inches='tight')

    plt.figure(figsize=(4 / 1.2, 3 / 1.2))
    for i in range(4):
        plt.plot(x, fp_lst[i], label=complementary_names[i], color=colors[i],
                alpha=0.3 if i % 2 == 0 else 0.8,
                linewidth=1.5 if i % 2 == 0 else 2,
                linestyle='--' if i % 2 == 0 else '-')
    plt.xlabel('Layer')
    plt.ylabel('False positives per sentence')
    plt.xlim((0, 40))
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.savefig(plots_dir / f'{exp_name}_fp.pdf', bbox_inches='tight')

plt.figure(figsize=(4 / 1.5, 3 / 1.5))
plt.plot(x, scores[0], alpha=0.8, linewidth=2, color=colors[0], label='One shot')
plt.plot(x, scores[1], alpha=0.8, linewidth=2, color=colors[2], label='Tree')
plt.xlabel('Layer')
plt.ylabel('Correlation score')
plt.xlim((0, 40))
plt.grid(True)
plt.legend()
plt.savefig(plots_dir / f'{exp_name[-14:]}_score.pdf', bbox_inches='tight')

plt.figure(figsize=(4 / 1.5, 3 / 1.5))
plt.plot(x, comps[0], alpha=0.8, linewidth=2, color=colors[0], label='One shot')
plt.plot(x, comps[1], alpha=0.8, linewidth=2, color=colors[2], label='Tree')
plt.xlabel('Layer')
plt.ylabel('Average complexity')
plt.xlim((0, 40))
plt.grid(True)
plt.legend()
plt.savefig(plots_dir / f'{exp_name[-14:]}_comp.pdf', bbox_inches='tight')

plt.figure(figsize=(4 / 1.5, 3 / 1.5))
plt.plot(x, polys[0], alpha=0.8, linewidth=2, color=colors[0], label='One-shot')
plt.plot(x, polys[1], alpha=0.8, linewidth=2, color=colors[2], label='Tree')
plt.xlabel('Layer')
plt.ylabel('Polysemanticity')
plt.xlim((0, 40))
plt.grid(True)
plt.legend()
plt.savefig(plots_dir / f'{exp_name[-14:]}_poly.pdf', bbox_inches='tight')
