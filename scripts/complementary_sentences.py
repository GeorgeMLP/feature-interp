from pathlib import Path
import numpy as np
import pickle
import os
from tqdm import tqdm
import warnings
import asyncio
from transformers import logging
from featureinterp import formatting
from featureinterp.explainer import (
    OneShotExplainer,
    TreeExplainer,
    OneShotExplainerParams,
    TreeExplainerParams,
)
from featureinterp.record import RecordSliceParams, ComplementaryRecordSource
from featureinterp.scoring import simulate_and_score
from scripts import experiment
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


def num_false_positives(true: list[float], predicted: list[float]) -> int:
    fp = sum(true[i] < 10 and predicted[i] > 1 for i in range(len(true)))
    return fp


INFERENCE_BATCH_SIZE = 2
EXPLAINER_MODEL_NAME = "google/gemini-flash-1.5-8b"
SIMULATOR_MODEL_NAME = "google/gemma-2-27b-it"
COMPLEXITY_MODEL_NAME = "google/gemma-2-27b-it"

simulator_factory = experiment.load_simulator_factory(
    SIMULATOR_MODEL_NAME,
    batch_size=INFERENCE_BATCH_SIZE,
)

complexity_analyzer = experiment.load_complexity_analyzer(
    COMPLEXITY_MODEL_NAME,
    batch_size=1,
)

explainer_type = 'tree'
rule_cap = 5
if explainer_type == 'one_shot':
    explainer = OneShotExplainer(
        model_name=EXPLAINER_MODEL_NAME,
        params=OneShotExplainerParams(
            include_holistic_expressions=False,
            rule_cap=rule_cap,
        )
    )
elif explainer_type == 'tree':
    explainer = TreeExplainer(
        model_name=EXPLAINER_MODEL_NAME,
        simulator_factory=simulator_factory,
        params=TreeExplainerParams(
            print_explanations=False,
            rule_cap=rule_cap,
            depth=3,
            width=3,
        ),
    )
else:
    raise NotImplementedError(f'Explainer {explainer_type} not implemented.')

save_cache: bool = True
save_results: bool = True
version: int = 4
if not save_cache:
    warnings.warn('Not saving cache to disk!', UserWarning)
if not save_results:
    warnings.warn('Not saving results to disk!', UserWarning)

results_path = Path('results', 'score_vs_layer')
results_path.mkdir(parents=True, exist_ok=True)

dataset_path = 'data/pile-uncopyrighted_gemma-2-9b/records'

train_record_params = RecordSliceParams(
    positive_examples_per_split=10,
)
valid_record_params = RecordSliceParams(
    positive_examples_per_split=10,
    complementary_examples_per_split=10,
    complementary_record_source=ComplementaryRecordSource.SIMILAR_NEGATIVE,
)
test_positive_examples_per_split: int = 10
test_complementary_examples_per_split: int = 10

cache_dir = Path(f'cache/pile-uncopyrighted_gemma-2-9b/exp_results_v{version}')
cache_dir.mkdir(parents=True, exist_ok=True)

async def main() -> None:
    layer_indices = range(0, 42, 2)
    feature_indices = range(50)

    layer_results: dict[str, list[list[experiment.ExperimentResult]]] = {
        s.value: [] for s in ComplementaryRecordSource
    }

    for layer_index in tqdm(layer_indices, desc='Layer'):
        layer_experiments: dict[str, list[experiment.ExperimentResult]] = {
            s.value: [] for s in ComplementaryRecordSource
        }

        for feature_index in tqdm(feature_indices, desc='Feature', leave=True):
            cache_path = cache_dir / Path('_'.join((
                str(layer_index),
                str(feature_index),
                explainer_type,
                str(train_record_params.positive_examples_per_split),
                'None' if train_record_params.complementary_record_source is None \
                    else str(train_record_params.complementary_record_source.value),
                str(train_record_params.complementary_examples_per_split),
                str(valid_record_params.positive_examples_per_split),
                'None' if valid_record_params.complementary_record_source is None \
                    else str(valid_record_params.complementary_record_source.value),
                str(test_positive_examples_per_split),
                str(test_complementary_examples_per_split),
            )) + '.pkl')

            if save_cache and os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    feature_experiments: dict[str, experiment.ExperimentResult] = pickle.load(f)
            else:
                sae_index_record = experiment.load_sae_index_record(
                    layer_index=layer_index,
                    latent_index=feature_index,
                    records_path=dataset_path,
                )
                train_records = formatting.format_records(
                    sae_index_record.train_records(train_record_params),
                    max_expression=sae_index_record.max_expression,
                    max_holistic_expression=sae_index_record.max_holistic_expression,
                )
                valid_records = formatting.format_records(
                    sae_index_record.train_records(valid_record_params),
                    max_expression=sae_index_record.max_expression,
                    max_holistic_expression=sae_index_record.max_holistic_expression,
                )

                explanations, explanation_extra_data = await explainer.generate_explanations(
                    train_records=train_records,
                    valid_records=valid_records,
                )
                assert len(explanations) == 1
                explanation = explanations[0]

                feature_experiments: dict[str, experiment.ExperimentResult] = {}
                for strategy in ComplementaryRecordSource:
                    test_record_params = RecordSliceParams(
                        positive_examples_per_split=test_positive_examples_per_split,
                        complementary_examples_per_split=test_complementary_examples_per_split,
                        complementary_record_source=strategy,
                    )
                    test_records = sae_index_record.test_records(test_record_params)
                    
                    simulator = simulator_factory(explanation)
                    scored_simulation = await simulate_and_score(simulator, test_records)
                    score = scored_simulation.get_preferred_score()

                    all_true: list[float] = []
                    all_pred: list[float] = []
                    for scored_sequence_simulation in \
                        scored_simulation.scored_sequence_simulations[test_positive_examples_per_split:]:
                        true = scored_sequence_simulation.true_expressions
                        pred = scored_sequence_simulation.simulation.expected_expressions
                        all_true.extend(true)
                        all_pred.extend(pred)
                    false_positives = num_false_positives(all_true, all_pred)
                    fp_rate = false_positives / len(all_true)
                    
                    complexities = None
                    if complexity_analyzer is not None:
                        complexities = complexity_analyzer.analyze_complexity(explanation)

                    result = experiment.ExperimentResult(
                        layer_index=sae_index_record.id.layer_index,
                        latent_index=sae_index_record.id.latent_index,
                        max_expression=sae_index_record.max_expression,
                        max_holistic_expression=sae_index_record.max_holistic_expression,
                        score=score,
                        explanation=explanation,
                        explanation_extra_data=explanation_extra_data,
                        explanation_complexities=complexities,
                        fp_rate=fp_rate,
                        scored_simulation=scored_simulation,
                    )

                    feature_experiments[strategy.value] = result
                
                if save_cache:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(feature_experiments, f)

            for s in ComplementaryRecordSource:
                layer_experiments[s.value].append(feature_experiments[s.value])
        
        for s in ComplementaryRecordSource:
            layer_results[s.value].append(layer_experiments[s.value])

    print("\n=== Experiment Results ===")
    for strategy in ComplementaryRecordSource:
        print(f"\nComplementary record source: {strategy.value}")
        for layer_exps in layer_results[strategy.value]:
            layer_index = layer_exps[0].layer_index
            print(f"  Layer {layer_index}:")
            scores = [exp.score for exp in layer_exps if not np.isnan(exp.score)]
            print(f"    Average Score: {np.mean(scores):.3f}")
            num_fps = [exp.fp_rate * 32 for exp in layer_exps]
            print(f"    Average False Positives: {np.mean(num_fps):.3f}")
            mean_complexities = [
                np.nanmean(exp.explanation_complexities) for exp in layer_exps \
                    if exp.explanation_complexities != []
            ]
            print(f"    Average Mean Complexity: {np.nanmean(mean_complexities):.3f}")
    
    if save_results:
        with open(results_path / f'experiment_results_{explainer_type}_{len(feature_indices)}_features_v{version}.pkl', 'wb') as f:
            pickle.dump(layer_results, f)


if __name__ == '__main__':
    asyncio.run(main())
