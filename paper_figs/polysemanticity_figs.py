from pathlib import Path
import pickle
import numpy as np

from scripts.polysemanticity_sweep import LayerFeatureResult
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats


def mean_confidence_interval(data, confidence=0.80) -> tuple[float, float]:
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h 


# Set up LaTeX fonts for all plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


truncated_plasma = truncate_colormap(plt.get_cmap('plasma'), 0.0, 0.95)


def make_poly_heatmap(
    rules: list[int],
    layer_scores: dict[int, list[float]],
    file_path: Path,
    score_label: str,
) -> None:
    layers = sorted(layer_scores.keys())
    score_matrix = np.array([layer_scores[layer] for layer in layers])
    
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(
        score_matrix,
        cmap=truncated_plasma,
        aspect='auto',
        vmin=0,
        vmax=1,
    )
    
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    
    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels(rules)
    
    ax.set_ylabel('Layer')
    ax.set_xlabel('Rule Count') 
    ax.set_title('Polysemanticity Heatmap')
    
    cbar = plt.colorbar(im, ax=ax, label=score_label)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()


plots_path = Path('plots') / 'polysemanticity'
plots_path.mkdir(exist_ok=True)

subject_models = ['gemma', 'llama', 'gpt2']

subject_model_labels = {
    'gemma': 'Gemma 2 9b',
    'llama': 'Llama 3.1 8b',
    'gpt2': 'GPT-2 Small',
}

aggregate_mono_score_props_unflattened: dict[str, dict[int, list[list[float]]]] = {}
aggregate_complexities: dict[str, dict[int, tuple[float, float]]] = {}

for subject_model in subject_models:
    results_dir = Path(f"results/polysemanticity_sweep_{subject_model}/layer_results")
    
    experiment_results: list[list[dict[int, LayerFeatureResult]]] = []

    for file in results_dir.iterdir():
        if file.is_file() and file.name.endswith('.pkl'):
            layer = file.name.split('_')[1].split('.')[0]
            with open(file, 'rb') as f:
                layer_results = pickle.load(f)

                experiment_results.append([
                    {int(k): v for k, v in d.items()}
                    for d in layer_results
                ])
    
    experiment_results.sort(key=lambda x: x[0][1].experiment_result.layer_index)

    max_rule_cap = max(experiment_results[0][0].keys())
    rules = list(range(1, max_rule_cap + 1))

    # Scores for each rule count
    layer_scores: dict[int, list[float]] = {}
    layer_score_props: dict[int, list[float]] = {}
    
    # Monotonically increasing scores (take max of any rule count < i)
    layer_mono_scores: dict[int, list[float]] = {}
    layer_mono_score_props: dict[int, list[float]] = {}
    layer_mono_score_props_unflattened: dict[int, list[list[float]]] = {}

    # Average complexities for the max rule count
    layer_complexities: dict[int, float] = {}
    
    for layer_results in experiment_results:
        layer_index = layer_results[0][1].experiment_result.layer_index

        # For each rule cap, store the scores
        scores: list[list[float]] = [[] for _ in range(max_rule_cap + 1)]
        score_props: list[list[float]] = [[] for _ in range(max_rule_cap + 1)]
        mono_scores: list[list[float]] = [[] for _ in range(max_rule_cap + 1)]
        mono_score_props: list[list[float]] = [[] for _ in range(max_rule_cap + 1)]

        # Store the average complexities of the max rule cap
        complexities: list[float] = []
        
        for feature_rule_cap_result in layer_results:
            if len(feature_rule_cap_result) == 0:
                continue

            if np.isnan(feature_rule_cap_result[1].experiment_result.score):
                continue
            
            result_rules, result_scores = [], []
            for rule_cap, feature_result in feature_rule_cap_result.items():
                result_rules.append(rule_cap)
                result_scores.append(feature_result.experiment_result.score)
            
            result_rules = np.array(result_rules)
            result_scores = np.array(result_scores)
            
            max_score = max(result_scores)
            if max_score <= 0.0:
                continue

            for rule_count, score in zip(result_rules, result_scores, strict=True):
                scores[rule_count].append(score)
                score_props[rule_count].append(score / max_score)
            
            complexities.append(
                np.nanmean(
                    feature_rule_cap_result[max_rule_cap]
                    .experiment_result
                    .explanation_complexities
                )
            )
            
            for i in range(0, max_rule_cap+1):
                smaller_rules = result_rules <= i
                if np.sum(smaller_rules) == 0:
                    mono_scores[i].append(0)
                    mono_score_props[i].append(0)
                else:
                    max_smaller_score = max(result_scores[smaller_rules], default=np.nan)
                    if np.isnan(max_smaller_score):
                        import pdb; pdb.set_trace()
                    mono_scores[i].append(max_smaller_score)
                    mono_score_props[i].append(max_smaller_score / max_score)
        
        layer_scores[layer_index] = [np.nanmean(s) for s in scores[1:]]
        layer_score_props[layer_index] = [np.nanmean(s) for s in score_props[1:]]
        layer_mono_scores[layer_index] = [np.nanmean(s) for s in mono_scores[1:]]
        layer_mono_score_props[layer_index] = [np.nanmean(s) for s in mono_score_props[1:]]
        layer_mono_score_props_unflattened[layer_index] = mono_score_props
        layer_complexities[layer_index] = mean_confidence_interval(complexities)

    aggregate_mono_score_props_unflattened[subject_model] = layer_mono_score_props_unflattened
    aggregate_complexities[subject_model] = layer_complexities

    make_poly_heatmap(
        rules,
        layer_scores,
        plots_path / f'scores_{subject_model}.pdf',
        'Score',
    )
    make_poly_heatmap(
        rules,
        layer_score_props,
        plots_path / f'score_props_{subject_model}.pdf',
        'Proportion of Max Score',
    )
    make_poly_heatmap(
        rules,
        layer_mono_scores,
        plots_path / f'mono_scores_{subject_model}.pdf',
        'Score',
    )
    make_poly_heatmap(
        rules,
        layer_mono_score_props,
        plots_path / f'mono_score_props_{subject_model}.pdf',
        'Proportion of Max Score',
    )

    # Score line plot
    plt.figure(figsize=(4 / 1.2, 3 / 1.2))
    for layer_ind, scores in layer_scores.items():
        plt.plot(rules, scores, label=f'Layer {layer_ind}', linewidth=2, alpha=0.8)
    plt.grid(True)
    plt.xlabel('Rule Count')
    plt.ylabel('Score')
    plt.xlim(1, max_rule_cap)
    plt.legend(fontsize=6)
    plt.savefig(plots_path / f'score_plot_{subject_model}.pdf', bbox_inches='tight')
    plt.close()
    
my_colors = ['#120789', '#FA9E3B', '#C23D80']

# Plot average complexities per layer for each model
plt.figure(figsize=(4 / 1.2, 3 / 1.2))
for i, (subject_model, layer_complexities) in enumerate(aggregate_complexities.items()):
    layer_indices = sorted(list(layer_complexities.keys()))
    layer_comps = [layer_complexities[ind][0] for ind in layer_indices]
    layer_conf_intervals = [layer_complexities[ind][1] for ind in layer_indices]
    layer_index_props = [ind / max(layer_indices) * 100 for ind in layer_indices]
    plt.plot(
        layer_index_props,
        layer_comps,
        label=subject_model_labels[subject_model],
        color=my_colors[i],
        linewidth=2,
        alpha=0.8,
    )
    plt.fill_between(
        layer_index_props,
        np.array(layer_comps) - np.array(layer_conf_intervals),
        np.array(layer_comps) + np.array(layer_conf_intervals),
        color=my_colors[i],
        alpha=0.2,
    )
plt.xlabel('Layer (\%)')
plt.ylabel('Average complexity')
plt.xlim(0, 100)
plt.legend()
plt.grid(True)
plt.savefig(plots_path / 'avgcomp.pdf', bbox_inches='tight')
plt.close()
    
# Plot median proportion of max score across layers for each model
plt.figure(figsize=(4 / 1.2, 3 / 1.2))
for i, model in enumerate(aggregate_mono_score_props_unflattened):
    # Key is layer index
    # Value has outer index rule count, inner index is the feature
    # The float is the proportion of the max score for that feature
    layer_props: dict[int, list[list[float]]] = aggregate_mono_score_props_unflattened[model]
    layer_indices = list(layer_props.keys())
    
    # Compute rule count at which each feature first reaches a proportion of 0.9
    max_score_target_proportion = 0.90
    # Length of each value is equal to number of features
    # Value is the rule count at which the feature first reaches the target proportion
    rules_at_max_score_target_proportion: dict[int, list[int]] = {
        layer_index: [] for layer_index in layer_indices
    }

    for layer_index, layer_props_list in layer_props.items():
        # Transpose the list of lists
        features_first_list = list(zip(*layer_props_list))
        
        for feature_props in features_first_list:
            for rule_count, prop in enumerate(feature_props):
                if prop >= max_score_target_proportion:
                    rules_at_max_score_target_proportion[layer_index].append(rule_count)
                    break
    
    agg_props = [
        mean_confidence_interval(rules_at_max_score_target_proportion[layer_index])
        for layer_index in layer_indices
    ]
    means, confidences = zip(*agg_props)
    x_values = list(range(len(agg_props)))
    plt.plot(
        x_values,
        means,
        label=subject_model_labels[model],
        marker=None,
        color=my_colors[i],
        linewidth=2,
        alpha=0.8,
    )
    plt.fill_between(
        x_values,
        np.array(means) - np.array(confidences),
        np.array(means) + np.array(confidences),
        color=my_colors[i],
        alpha=0.2,
    )

plt.xlabel('Layer (\%)')
plt.ylabel('Mean rule count for 90\% of max score')
plt.gca().yaxis.label.set_position((0, 0.4))  # lower the label of y axis

num_layers = len(agg_props)
plt.xticks(
    np.arange(0, 1 + 0.2, 0.2) * (num_layers - 1),
    [f'{int(x)}' for x in np.arange(0, 1 + 0.2, 0.2) * 100],
)
plt.xlim(0, num_layers - 1)

plt.legend()
plt.grid(True)
plt.savefig(plots_path / 'line_plot.pdf', bbox_inches='tight')
plt.close()
