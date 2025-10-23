from pathlib import Path
import pickle
from typing import OrderedDict
import numpy as np

import matplotlib.pyplot as plt
import scipy.stats

import tabulate


def mean_confidence_interval(data, confidence=0.90):
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

plots_path = Path('plots')
plots_path.mkdir(exist_ok=True)

subject_models = ['gemma', 'llama', 'gpt2']

subject_model_labels = {
    'gemma': 'Gemma 2 9b',
    'llama': 'Llama 3.1 8b',
    'gpt2': 'GPT-2 Small',
}

suffix_combos = [
    {'structured': False, 'train_negatives': False, 'holistic_expressions': False},
    {'structured': True, 'train_negatives': False, 'holistic_expressions': False},
    {'structured': True, 'train_negatives': True, 'holistic_expressions': False},
    {'structured': True, 'train_negatives': False, 'holistic_expressions': True},
]

plot_layer_scores = OrderedDict()
for method in ['one-shot', 'tree']:
    for combo in suffix_combos:
        plot_layer_scores[frozenset({'method': method, **combo}.items())] = {}

for subject_model in subject_models:
    for method in ['one-shot', 'tree']:
        for combo in suffix_combos:
            structured = combo['structured']
            train_negatives = combo['train_negatives']
            holistic_expressions = combo['holistic_expressions']

            file_name = (
                f"results/explainer_comparison_{subject_model}/"
                f"data{subject_model}_explainerllama-4-scout_method{method}_"
                f"structured{structured}_trainnegatives{train_negatives}"
                f"_holisticexpressions{holistic_expressions}.pkl"
            )
            
            if not Path(file_name).exists():
                continue
            
            with open(file_name, 'rb') as f:
                layer_results = pickle.load(f)

            layer_scores = []
            for layer_experiments in layer_results:
                if layer_experiments:
                    layer_scores.extend([
                        (exp.score if not np.isnan(exp.score) else 0)
                        for exp in layer_experiments]
                    )
                    
            
            plot_layer_scores[
                frozenset({'method': method, **combo}.items())
            ][subject_model] = mean_confidence_interval(layer_scores)


x = np.arange(len(subject_models))
width = 0.1
multiplier = 1 - len(suffix_combos) // 2

plt.figure(figsize=(4 * 1.2, 3 * 1.2))
ax = plt.gca()

method_labels = {'one-shot': 'One-shot (baseline)', 'tree': 'Tree (ours)'}

combo_labels_and_patterns = [
    ('Unstructured (baseline)', ''),
    ('Structured (ours)', '//'),
    ('Structured + train neg', '\\\\'),
    ('Structured + holistic', '--'),
]

# my_colors = ['#120789', '#fa9e3b', '#c23d80']
my_colors = ['#49419c', '#fa9e3b', '#c23d80']
# custom_cycler = cycler(color=my_colors)

# Create bars without adding to legend
for i, method in enumerate(['one-shot', 'tree']):
    method_label = method_labels[method]
    color = my_colors[i]

    for j, combo in enumerate(suffix_combos):
        combo_label, combo_pattern = combo_labels_and_patterns[j]
        alpha = 1 - j * 0.15
        
        method_and_combo = frozenset({'method': method, **combo}.items())
        if method_and_combo not in plot_layer_scores:
            continue
        
        confidence_intervals: dict[str, tuple[float, float]] = plot_layer_scores[method_and_combo]

        if len(confidence_intervals) < len(x):
            continue

        confidence_intervals = [confidence_intervals[sm] for sm in subject_models]

        means = [ci[0] for ci in confidence_intervals]
        confidence_bounds = [ci[1] for ci in confidence_intervals]

        offset = width * multiplier + (0.03 if method == 'tree' else 0)
        rects = ax.bar(
            x + offset,
            means,
            width,
            yerr=confidence_bounds,
            hatch=combo_pattern,
            color=color,
            alpha=alpha,
        )
        multiplier += 1

# Create first legend for methods (colors)
legend_elements = [
    plt.Rectangle(
        (0, 0), 1, 1,
        facecolor=my_colors[i],
        label=method_labels[method]
    )
    for i, method in enumerate(['one-shot', 'tree'])
]
first_legend = ax.legend(
    handles=legend_elements,
    loc='lower left',
    bbox_to_anchor=(0.0, -0.0),
    fontsize=8,
)
ax.add_artist(first_legend)

# Create second legend for patterns
legend_elements = [
    plt.Rectangle(
        (0, 0), 1, 1,
        facecolor='white',
        hatch=pattern,
        label=label,
        edgecolor='black'
    )
    for label, pattern in combo_labels_and_patterns
]
ax.legend(
    handles=legend_elements,
    loc='lower right',
    bbox_to_anchor=(1.0, -0.0),
    fontsize=8,
)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Layer score')
ax.set_xticks(
    x + width * (len(suffix_combos) - len(suffix_combos) // 2 + 0.5) + 0.015,
    [subject_model_labels[sm] for sm in subject_models],
)

# Adjust the layout to make room for legends at the bottom
plt.subplots_adjust(bottom=0.2)

plt.savefig(plots_path / 'explainer_comparison.pdf', bbox_inches='tight')

# Create table of mean scores per model and method
table_data = []
for subject_model in subject_models:
    row = [subject_model_labels[subject_model]]
    
    max_in_row = None
    for method in ['one-shot', 'tree']:
        for combo in suffix_combos:
            key = frozenset({'method': method, **combo}.items())
            if key not in plot_layer_scores:
                continue

            scores = plot_layer_scores[key]
            if subject_model not in scores:
                continue
            
            mean, confidence_bounds = scores[subject_model]
            if max_in_row is None or mean > max_in_row:
                max_in_row = mean

    for method in ['one-shot', 'tree']:
        for combo in suffix_combos:
            key = frozenset({'method': method, **combo}.items())
            if key in plot_layer_scores:
                scores = plot_layer_scores[key]
                if subject_model not in scores:
                    row.append("N/A")
                    continue
                
                mean, confidence_bounds = scores[subject_model]
                if mean + 0.0001 > max_in_row:
                    mean_string = rf"\textbf{{{mean:.3f}}}"
                else:
                    mean_string = rf"{mean:.3f}"
                row.append(rf"${mean_string} \pm {confidence_bounds:.3f}$")
            else:
                row.append("N/A")
    table_data.append(row)

headers = [
    "Model",
    r"\makecell[t]{One-shot\\Unstructured}",
    r"\makecell[t]{One-shot\\Structured}", 
    r"\makecell[t]{One-shot\\Structured\\w/ Negatives}",
    r"\makecell[t]{One-shot\\Structured\\w/ Holistic}",
    r"\makecell[t]{Tree\\Unstructured}",
    r"\makecell[t]{Tree\\Structured}",
    r"\makecell[t]{Tree\\Structured\\w/ Negatives}",
    r"\makecell[t]{Tree\\Structured\\w/ Holistic}",
]

with open('plots/explainer_comparison_table.tex', 'w') as f:
    f.write(tabulate.tabulate(
        table_data,
        headers=headers,
        tablefmt="latex_raw"
    ))
