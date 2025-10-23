# Revising and Falsifying Sparse Autoencoder Feature Explanations

This is the code base for our NeurIPS 2025 paper "Revising and Falsifying Sparse Autoencoder Feature Explanations", by George Ma, Samuel Pfrommer, and Somayeh Sojoudi.

## Overview

This repository provides tools and methods for automatically generating, evaluating, and iteratively improving explanations of Sparse Autoencoder (SAE) features in neural networks. The key contributions include:

- **Automated explanation generation** using both one-shot and iterative tree-based methods
- **Simulation-based scoring** that evaluates explanations by predicting feature activations
- **Complementary negative examples** for more robust explanation evaluation
- **Support for multiple base models**: Gemma-2-9B, GPT-2, and Llama-3.1-8B

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for running experiments)
- Virtual environment tool (venv, conda, etc.)

### Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Run the setup script:
```bash
bash setup.sh
```

This will install system dependencies (`pv`, `pbzip2`) and the Python package with all required dependencies.

## Project Structure

```
feature-interp/
├── featureinterp/          # Core library code
│   ├── explainer.py        # OneShotExplainer and TreeExplainer implementations
│   ├── simulator.py        # Feature activation simulation
│   ├── scoring.py          # Explanation evaluation metrics
│   ├── record.py           # Data structures for feature activation records
│   ├── core.py             # Core data structures (StructuredExplanation, etc.)
│   ├── complexity.py       # Explanation complexity analysis
│   └── ...
├── scripts/                # Experiment scripts
│   ├── generate_records.py          # Generate activation records from models
│   ├── single_index_demo.py         # Minimal demo for single feature
│   ├── complementary_sentences.py   # Figure 3 experiments
│   ├── explainer_comparison.py      # Figure 4 experiments
│   ├── polysemanticity_sweep.py     # Figure 5 experiments
│   └── bash/                        # Shell scripts for running experiments
├── paper_figs/             # Plotting scripts for paper figures
└── tests/                  # Unit and integration tests
```

## Quick Start

### Minimal Demo

Run a minimal demonstration on a single SAE feature:

```bash
python scripts/single_index_demo.py
```

This will:
1. Load a pre-computed SAE feature record
2. Generate an explanation using an LLM
3. Simulate feature activations based on the explanation
4. Score the explanation quality
5. Display results including training examples, explanation, and simulation results

**Note**: This requires pre-generated records (see Data Generation below) and access to the specified LLM models.

## Usage

### 1. Data Generation

Before running experiments, you need to generate feature activation records from a base model and its corresponding SAE:

```bash
python scripts/generate_records.py \
    --write-sae-acts \
    --write-holistic-acts \
    --write-records
```

**Key parameters**:
- `--model-name`: Base model (e.g., `google/gemma-2-9b`)
- `--sae-name`: SAE model name (e.g., `gemma-scope-9b-pt-res-canonical`)
- `--dataset-name`: Dataset for activation collection (default: `monology/pile-uncopyrighted`)
- `--max-dataset-size`: Number of sequences to process (default: 100000)
- `--max-features`: Number of SAE features to analyze (default: 50)
- `--layers`: Which layers to process (`all`, `even`, `odd`, or comma-separated list)

This creates:
- **SAE activations**: Feature activation values for each token
- **Holistic activations**: Context-dependent activation attribution
- **Records**: Organized datasets including positive examples, negative examples, and similarity-based retrieval indices

### 2. Explanation Methods

The codebase supports two explanation generation methods:

#### One-Shot Explainer
Generates explanations in a single LLM call using few-shot prompting:

```python
from featureinterp.explainer import OneShotExplainer, OneShotExplainerParams

explainer = OneShotExplainer(
    model_name="meta-llama/llama-4-scout",
    params=OneShotExplainerParams(
        rule_cap=5,                         # Maximum number of rules
        include_holistic_expressions=False, # Include context information
        structured_explanations=True,       # Use structured JSON format
    )
)
```

#### Tree Explainer (Iterative)
Uses iterative refinement with simulation-based feedback:

```python
from featureinterp.explainer import TreeExplainer, TreeExplainerParams

explainer = TreeExplainer(
    model_name="meta-llama/llama-4-scout",
    simulator_factory=simulator_factory,
    params=TreeExplainerParams(
        depth=3,          # Tree search depth
        width=3,          # Number of candidates to keep at each level
        rule_cap=5,       # Maximum number of rules
        structured_explanations=True,
    )
)
```

### 3. Explanation Evaluation

Explanations are evaluated by simulating feature activations and comparing to ground truth:

```python
from featureinterp.scoring import simulate_and_score

simulator = simulator_factory(explanation)
scored_simulation = await simulate_and_score(simulator, test_records)
score = scored_simulation.get_preferred_score()  # Correlation coefficient
```

## Reproducing Paper Results

### Figure 3: Complementary Sentences
Evaluates the impact of different complementary negative example strategies:

```bash
python scripts/complementary_sentences.py
```

Then generate figures:
```bash
python paper_figs/complementary_sentences_figs.py
```

### Figure 4: Explainer Comparison
Compares one-shot vs. tree explainers with various configurations:

```bash
bash scripts/bash/explainer_comparison.sh
```

Then generate figures:
```bash
python paper_figs/explainer_comparison_figs.py
```

### Figure 5: Polysemanticity Sweep
Analyzes how explanation complexity (number of rules) affects quality:

```bash
bash scripts/bash/polysemanticity_sweep.sh
```

Then generate figures:
```bash
python paper_figs/polysemanticity_figs.py
```

## Key Concepts

### Structured Explanations

Explanations are represented as lists of rules, each with:
- **`activates_on`** (string): Pattern or content the feature responds to
- **`strength`** (int 0-5): Activation strength

Example:
```json
[
  {"activates_on": "quotation marks at the beginning of quotes", "strength": 4},
  {"activates_on": "the start of dialogue in fiction", "strength": 3}
]
```

### Complementary Record Sources

The method uses various strategies for selecting negative examples:
- **RANDOM**: Random sequences from the dataset
- **RANDOM_NEGATIVE**: Random sequences with zero activation
- **SIMILAR**: Semantically similar sentences ranked by Sentence Transformer
- **SIMILAR_NEGATIVE**: Semantically similar sequences with zero activation (our method)

### Holistic Expressions

Beyond token-level activations, the system can analyze **activation-causing tokens** - earlier tokens in the sequence that cause future activations. This provides richer context for understanding feature behavior.

## Model Requirements

The experiments use several types of models:

1. **Explainer models**: Generate natural language explanations (e.g., `meta-llama/llama-4-scout`, `google/gemini-flash-1.5-8b`)
2. **Simulator models**: Predict activations from explanations (e.g., `google/gemma-2-27b-it`)
3. **Complexity analyzer**: Evaluate explanation complexity (optional)

Models are loaded with quantization for efficient GPU usage (4-bit or 8-bit).

## Configuration

Key configuration options in experiment scripts:

```python
# Dataset and model selection
dataset_path = 'data/pile-uncopyrighted_gemma-2-9b/records'
EXPLAINER_MODEL_NAME = "meta-llama/llama-4-scout"
SIMULATOR_MODEL_NAME = "google/gemma-2-27b-it"

# Record selection parameters
train_record_params = RecordSliceParams(
    positive_examples_per_split=10,
    complementary_examples_per_split=10,
    complementary_record_source=ComplementaryRecordSource.SIMILAR_NEGATIVE,
)

# Inference parameters
INFERENCE_BATCH_SIZE = 2
```

## Data Storage

Generated data is organized as:
```
data/
└── {dataset}_{model}/
    ├── tokens.pt                    # Tokenized sequences
    ├── sae_acts/                    # SAE activations per layer
    │   └── {hook_name}.pt
    ├── holistic_acts/               # Holistic activations per layer
    │   └── {hook_name}.pt
    ├── similarity_retriever/        # Embedding indices for retrieval
    └── records/                     # Organized feature records
        └── {hook_name}/
            └── {feature_idx}.json
```

Results are cached in `cache/` and final results saved to `results/`.

## Weights & Biases Integration

The codebase supports uploading/downloading datasets via W&B:

```bash
# Upload generated data
python scripts/generate_records.py --upload-wandb

# Download pre-generated data
python scripts/generate_records.py --download-wandb
```

## Citation

If you use results from our paper or find our paper relevant or useful, please cite:

```bibtex
@inproceedings{ma2025revising,
    title={{Revising and Falsifying Sparse Autoencoder Feature Explanations}},
    author={Ma, George and Pfrommer, Samuel and Sojoudi, Somayeh},
    booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
    year={2025}
}
```
