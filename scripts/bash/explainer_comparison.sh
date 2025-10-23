# Gemma
model_arg="gemma"
python scripts/explainer_comparison.py --model $model_arg --explainer_type one-shot
python scripts/explainer_comparison.py --model $model_arg --explainer_type one-shot --structured_explanations
python scripts/explainer_comparison.py --model $model_arg --explainer_type one-shot --structured_explanations --include_holistic_expressions
python scripts/explainer_comparison.py --model $model_arg --explainer_type one-shot --structured_explanations --include_train_negatives

# Same thing but with tree
python scripts/explainer_comparison.py --model $model_arg --explainer_type tree
python scripts/explainer_comparison.py --model $model_arg --explainer_type tree --structured_explanations
python scripts/explainer_comparison.py --model $model_arg --explainer_type tree --structured_explanations --include_holistic_expressions
python scripts/explainer_comparison.py --model $model_arg --explainer_type tree --structured_explanations --include_train_negatives
