import numpy as np
import matplotlib.pyplot as plt


def get_corrcoef(ground_truth_positives: float, false_positive_chance: float) -> float:
    n = 10000
    max_activation = 5.0
    most_activating_pred_error = 0.3
    
    # "Most-activating" sequences
    true_group_a = (np.arange(n) / n) * max_activation
    zero_mask = np.random.choice(
        [0, 1],
        size=n,
        p=[1 - ground_truth_positives, ground_truth_positives],
    )
    true_group_a = true_group_a * zero_mask
    pred_group_a = true_group_a + np.random.randn(n) * most_activating_pred_error

    # Negative sequences
    true_group_b = np.zeros(n)
    pred_group_b = np.zeros(n)
    mask = np.random.choice(
        [0, 1],
        size=n,
        p=[1 - false_positive_chance, false_positive_chance],
    )
    random_vals = np.random.uniform(0, max_activation, size=n)
    pred_group_b = np.where(mask == 1, random_vals, 0)

    # Compute correlation coefficient
    true_all = np.concatenate([true_group_a, true_group_b])
    pred_all = np.concatenate([pred_group_a, pred_group_b])
    corr = np.corrcoef(true_all, pred_all)[0,1]

    return corr


def get_average_corrcoef(
    ground_truth_positives: float,
    false_positive_chance: float,
) -> float:
    corrs = [get_corrcoef(ground_truth_positives, false_positive_chance) for _ in range(10)]
    return np.mean(corrs)


ground_truth_positives = 0.05

for false_positive_chance in [0.0, 0.01, 0.03, 0.05, 0.1]:
    print(f"False positive chance: {false_positive_chance}")
    print(f"Average correlation coefficient: {get_average_corrcoef(ground_truth_positives, false_positive_chance):.3f}")
