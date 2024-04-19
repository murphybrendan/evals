from llm_evals_viz.benchmarks import *
from llm_evals_viz.models.claude3 import Claude3Model

from datasets import load_dataset
from dataclasses import asdict

from tqdm import tqdm


def run_eval():
    """Run the evaluation."""
    benchmark = MMLU()
    model = Claude3Model("claude-3-haiku-20240307")
    dataset = load_dataset(**asdict(benchmark))
    responses = [
    model.get_completion_for_prompt(
        benchmark.build_prompt(example["question"], example["choices"]))
    for example in tqdm(dataset, desc=f"Running {model} on {benchmark}")]
    correct = sum([int(response) == example["answer"] for response, example in zip(responses, dataset)])
    return f"Total questions: {len(dataset)}\nNumber correct: {correct}\nAccuracy: {correct / len(dataset) * 100:.2f}%"





