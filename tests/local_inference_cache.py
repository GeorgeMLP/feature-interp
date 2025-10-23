import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from featureinterp.local_inference import LocalInferenceManager
from featureinterp.prompt_builder import Message, Role


def test_kv_cache_equivalence():
    model_name = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

    header_messages = [
        Message(Role.SYSTEM, "You are a helpful assistant."),
        Message(Role.USER, "What is 2+1?"),
        Message(Role.ASSISTANT, "3"),
    ]
    
    body_messages = [
        Message(Role.USER, "What is 2+2?"),
        Message(Role.ASSISTANT, "4"),
    ]

    suffix_messages = [
        Message(Role.USER, "What is 3+3?"),
    ]

    head_cached_manager = LocalInferenceManager(tokenizer, model)
    head_cached_manager.append_to_cache(header_messages)
    head_cached_tokens, head_cached_logprobs = head_cached_manager.run_batched_inference(
        [body_messages + suffix_messages]
    )[0]

    head_and_body_cached_manager = LocalInferenceManager.clone(head_cached_manager)
    head_and_body_cached_manager.append_to_cache(body_messages)
    head_and_body_cached_tokens, head_and_body_cached_logprobs = (
        head_and_body_cached_manager.run_batched_inference(
            [suffix_messages]
        )[0]
    )

    no_cached_manager = LocalInferenceManager(tokenizer, model)
    no_cached_tokens, no_cached_logprobs = no_cached_manager.run_batched_inference(
        [header_messages + body_messages + suffix_messages]
    )[0]

    head_cached_tokens = head_cached_tokens[-len(head_and_body_cached_tokens):]
    head_cached_logprobs = head_cached_logprobs[-len(head_and_body_cached_logprobs):]
    no_cached_tokens = no_cached_tokens[-len(head_and_body_cached_tokens):]
    no_cached_logprobs = no_cached_logprobs[-len(head_and_body_cached_logprobs):]

    assert torch.allclose(head_cached_tokens, no_cached_tokens)
    assert torch.allclose(head_cached_logprobs, no_cached_logprobs, atol=1e-3)
    assert torch.allclose(head_and_body_cached_tokens, no_cached_tokens)
    assert torch.allclose(head_and_body_cached_logprobs, no_cached_logprobs, atol=1e-3)

    print("Test passed! Cached and uncached inference produce identical results.")


if __name__ == "__main__":
    test_kv_cache_equivalence()
