# %%
from tqdm import tqdm
from llm_evals_viz.benchmarks import *
from llm_evals_viz.models.claude3 import Claude3Model
from llm_evals_viz import run_eval

from datasets import load_dataset
from dataclasses import asdict

# %%
benchmark = MMLU()
model = Claude3Model("claude-3-haiku-20240307")
dataset = load_dataset(**asdict(benchmark))

# %%
dataset[0]

# %%
responses = [
    model.get_completion_for_prompt(
        benchmark.build_prompt(example["question"], example["choices"]))
    for example in tqdm(dataset, desc="Running model on dataset")]

# %%
correct = sum([int(response) == example["answer"] for response, example in zip(responses, dataset)])
f"Total questions: {len(dataset)}\nNumber correct: {correct}\nAccuracy: {correct / len(dataset) * 100:.2f}%"


# %%
run_eval()
