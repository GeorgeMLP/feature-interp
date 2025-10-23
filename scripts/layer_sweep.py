from pathlib import Path
import click
import pickle
import orjson
import tqdm
import warnings
from transformers import logging
import asyncio
from dataclasses import dataclass

from featureinterp.explainer import (
    OneShotExplainer,
    OneShotExplainerParams,
    TreeExplainer,
    TreeExplainerParams,
)
from featureinterp.record import RecordSliceParams, ComplementaryRecordSource
from featureinterp.core import StructuredExplanation
from featureinterp.scoring import simulate_and_score
from scripts import experiment


logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


@dataclass
class LayerFeatureResult:
    experiment_result: experiment.ExperimentResult
    
    all_explanations: list[StructuredExplanation]
    all_train_scores: list[float]
    all_valid_scores: list[float]
    all_iterations: list[int]
    
    all_test_scores: list[float]
    all_complexities: list[list[float]]


async def main_async(model_arg: str, explainer_type: str):
    results_path = Path('results', 'layer_sweep', 'layer_results')
    results_path.mkdir(exist_ok=True, parents=True)
    
    EXPLAINER_MODEL_NAME = "meta-llama/llama-4-scout"
    SIMULATOR_MODEL_NAME = "google/gemma-2-27b-it"
    COMPLEXITY_MODEL_NAME = "google/gemma-2-27b-it"
    
    INFERENCE_BATCH_SIZE = 1
    simulator_factory = experiment.load_simulator_factory(
        SIMULATOR_MODEL_NAME,
        batch_size=INFERENCE_BATCH_SIZE
    )
    complexity_analyzer = experiment.load_complexity_analyzer(
        COMPLEXITY_MODEL_NAME,
        batch_size=INFERENCE_BATCH_SIZE
    )
    
    if explainer_type == 'one-shot':
        explainer = OneShotExplainer(
            model_name=EXPLAINER_MODEL_NAME,
            params=OneShotExplainerParams(
                rule_cap=5,
                include_holistic_expressions=False,
                structured_explanations=True,
            ),
        )
    elif explainer_type == 'tree':
        explainer = TreeExplainer(
            model_name=EXPLAINER_MODEL_NAME,
            simulator_factory=simulator_factory,
            params=TreeExplainerParams(
                rule_cap=5,
                depth=2,
                width=2,
                branching_factor=2,
                include_holistic_expressions=False,
                print_explanations=False,
            )
        )
    
    train_record_params = RecordSliceParams(
        positive_examples_per_split=10,
    )
    valid_record_params = RecordSliceParams(
        positive_examples_per_split=10,
        complementary_examples_per_split=10,
        complementary_record_source=ComplementaryRecordSource.SIMILAR_NEGATIVE,
    )
    test_record_params = RecordSliceParams(
        positive_examples_per_split=10,
        complementary_examples_per_split=10,
        complementary_record_source=ComplementaryRecordSource.SIMILAR_NEGATIVE,
    )

    dataset_path = experiment.get_dataset_path(model_arg)
    layer_indices = experiment.get_layer_indices(model_arg, subpoints=None)
    feature_indices = range(30)

    for layer_index in tqdm.tqdm(layer_indices, desc='Layer'):
        layer_experiments: list[LayerFeatureResult] = []
        for feature_index in tqdm.tqdm(feature_indices, desc='Feature', leave=False):
            sae_index_record = experiment.load_sae_index_record(
                layer_index=layer_index,
                latent_index=feature_index,
                records_path=dataset_path,
            )
            result = await experiment.run_experiment(
                sae_index_record=sae_index_record,
                train_record_params=train_record_params,
                valid_record_params=valid_record_params,
                test_record_params=test_record_params,
                explainer=explainer,
                simulator_factory=simulator_factory,
                complexity_analyzer=complexity_analyzer,
            )
            
            extra = result.explanation_extra_data

            explanations: list[StructuredExplanation] = extra['all_explanations']
            train_scores: list[float] = extra['all_train_scores']
            valid_scores: list[float] = extra['all_valid_scores']
            iterations: list[int] = extra['all_iterations']
            
            test_records = sae_index_record.test_records(test_record_params)

            complexities, test_scores = [], []
            for explanation in explanations:
                simulator = simulator_factory(explanation)
                scored_simulation = await simulate_and_score(simulator, test_records)
                test_scores.append(scored_simulation.get_preferred_score())
                complexities.append(
                    await complexity_analyzer.analyze_complexity(explanation)
                )
            
            layer_feature_result = LayerFeatureResult(
                experiment_result=result,
                all_explanations=explanations,
                all_train_scores=train_scores,
                all_valid_scores=valid_scores,
                all_iterations=iterations,
                all_test_scores=test_scores,
                all_complexities=complexities,
            )
            
            if result is not None:
                layer_experiments.append(layer_feature_result)
            
        with open(results_path / f'layer_{layer_index}.json', 'wb') as f:
            f.write(orjson.dumps(layer_experiments))

        with open(results_path / f'layer_{layer_index}.pkl', 'wb') as f:
            pickle.dump(layer_experiments, f)


@click.command()
@click.option('--model', type=click.Choice(['gemma', 'gpt2', 'llama']), default='gemma')
@click.option('--explainer_type', type=click.Choice(['one-shot', 'tree']), default='one-shot')
def main(model: str, explainer_type: str):
    asyncio.run(main_async(model, explainer_type))


if __name__ == "__main__":
    main()
