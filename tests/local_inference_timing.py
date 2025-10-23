import copy
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from featureinterp.local_inference import LocalInferenceManager
from featureinterp.prompt_builder import Message, Role


def test_kv_cache_timing():
    model_name = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

    header_messages = [
        Message(Role.SYSTEM, "You are a helpful assistant."),
        Message(Role.USER, "What is 2+1?"),
        Message(Role.ASSISTANT, "3"),
    ]

    suffix1_messages = [
        Message(Role.USER, "What is 3+3?"),
    ]
    suffix2_messages = [
        Message(Role.USER, "What is 432 + 41234244?"),
    ]

    trials = 5

    cached_manager = LocalInferenceManager(tokenizer, model, batch_size=trials)
    cached_manager.append_to_cache(header_messages)
    
    t0 = time.time()
    messages_list = []
    for _ in range(trials - 1):
        messages_list.append(copy.deepcopy(suffix1_messages))
    messages_list.append(copy.deepcopy(suffix2_messages))
    cached_results = cached_manager.run_batched_inference(messages_list)
    print(f"Head cached inference took {(time.time() - t0) / trials} seconds")


if __name__ == "__main__":
    test_kv_cache_timing()
