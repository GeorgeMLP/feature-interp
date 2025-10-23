from pathlib import Path
import click
import numpy as np
import dacite
import pickle
import orjson
import tqdm
import warnings
from transformers import logging
import asyncio
from dataclasses import dataclass

from featureinterp import formatting
from featureinterp.explainer import OneShotExplainer, OneShotExplainerParams, TreeExplainer, TreeExplainerParams
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
    
    rule_cap: int
    one_shot: bool


async def async_main(model_arg: str, layer_index: int):
    print(f"Running experiments for {model_arg}, layer {layer_index}")

    results_path = Path('results', f'polysemanticity_sweep_{model_arg}', 'layer_results')
    results_path.mkdir(exist_ok=True, parents=True)
    
    EXPLAINER_MODEL_NAME = "meta-llama/llama-4-scout"
    SIMULATOR_MODEL_NAME = "google/gemma-2-27b-it"
    COMPLEXITY_MODEL_NAME = "google/gemma-2-27b-it"
    ONE_SHOT = True
    
    INFERENCE_BATCH_SIZE = 2
    simulator_factory = experiment.load_simulator_factory(
        SIMULATOR_MODEL_NAME,
        batch_size=INFERENCE_BATCH_SIZE
    )
    complexity_analyzer = experiment.load_complexity_analyzer(
        COMPLEXITY_MODEL_NAME,
        batch_size=1,
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
    feature_indices = range(30)
    rule_caps = [1, 2, 3, 4, 5]

    # String so we can serialize as json
    layer_experiments: list[dict[str, LayerFeatureResult]] = []

    for feature_index in tqdm.tqdm(feature_indices, desc='Feature', leave=False):
        rule_cap_experiments: dict[str, LayerFeatureResult] = {}
        for rule_cap in rule_caps:
            sae_index_record = experiment.load_sae_index_record(
                layer_index=layer_index,
                latent_index=feature_index,
                records_path=dataset_path,
            )

            if ONE_SHOT:
                explainer = OneShotExplainer(
                    model_name=EXPLAINER_MODEL_NAME,
                    params=OneShotExplainerParams(
                        rule_cap=rule_cap,
                        include_holistic_expressions=False,
                        structured_explanations=True,
                    ),
                )
            else:
                explainer = TreeExplainer(
                    model_name=EXPLAINER_MODEL_NAME,
                    simulator_factory=simulator_factory,
                    params=TreeExplainerParams(
                        rule_cap=rule_cap,
                        depth=2,
                        width=2,
                        structured_explanations=True,
                        include_holistic_expressions=False,
                        print_explanations=False,
                    ),
                )

            try:
                result = await asyncio.wait_for(
                    experiment.run_experiment(
                        sae_index_record=sae_index_record,
                        train_record_params=train_record_params,
                        valid_record_params=valid_record_params,
                        test_record_params=test_record_params,
                        explainer=explainer,
                        simulator_factory=simulator_factory,
                        complexity_analyzer=complexity_analyzer,
                    ),
                    timeout=1500,
                )
            except asyncio.exceptions.TimeoutError:
                print(f"Timeout on layer {layer_index}, feature {feature_index}")
                continue
                
            if result is None:
                continue
            
            if ONE_SHOT:
                explanations = []
                train_scores = []
                valid_scores = []
                iterations = []
            else:
                extra = result.explanation_extra_data
                explanations: list[StructuredExplanation] = extra['all_explanations']
                train_scores: list[float] = extra['all_train_scores']
                valid_scores: list[float] = extra['all_valid_scores']
                iterations: list[int] = extra['all_iterations']

            layer_feature_result = LayerFeatureResult(
                experiment_result=result,
                all_explanations=explanations,
                all_train_scores=train_scores,
                all_valid_scores=valid_scores,
                all_iterations=iterations,
                rule_cap=rule_cap,
                one_shot=ONE_SHOT,
            )
            
            if result is not None:
                rule_cap_experiments[str(rule_cap)] = layer_feature_result
        
        layer_experiments.append(rule_cap_experiments)
        
    with open(results_path / f'layer_{layer_index}.json', 'wb') as f:
        f.write(orjson.dumps(layer_experiments))

    with open(results_path / f'layer_{layer_index}.pkl', 'wb') as f:
        pickle.dump(layer_experiments, f)

    
@click.command()
@click.option('--model', type=click.Choice(['gemma', 'gpt2', 'llama']), default='gemma')
@click.option('--layer_index', type=int)
def main(model: str, layer_index: int):
    asyncio.run(async_main(model, layer_index))


if __name__ == "__main__":
    main()
