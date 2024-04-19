
# %%
# Load the commonsense_qa dataset
from datasets import load_dataset

dataset = load_dataset("tau/commonsense_qa")
dataset

# %%
# Define the accuracy metric
import evaluate
import numpy as np

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# %%
# Preprocess the dataset
from transformers import AutoTokenizer
from functools import partial

bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
gpt_tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def preprocess_function(examples, tokenizer: AutoTokenizer, padding=False, truncation=False):
    choices = [[f"{label}. {text}" for label, text in zip(choice['label'], choice['text'])] for choice in examples['choices']]
    choices_prompts = ['\n'.join(c) for c in choices]
    prompts = [f"Question: {question}\n{choices_prompt}\nAnswer:{answer}" for question, choices_prompt, answer in zip(examples['question'], choices_prompts, examples['answerKey'])]
    return tokenizer(prompts, padding=padding, truncation=truncation)

# Don't pad the GPT model
bert_tokenized_dataset = dataset.map(partial(preprocess_function, tokenizer=bert_tokenizer, padding="max_length"), batched=True)
gpt_tokenized_dataset = dataset.map(partial(preprocess_function, tokenizer=gpt_tokenizer, padding=False), batched=True)


#%%
# Load pretrained models
from transformers import AutoModelForCausalLM

bert = AutoModelForCausalLM.from_pretrained("google-bert/bert-base-uncased")
gpt = AutoModelForCausalLM.from_pretrained("openai-community/openai-gpt")
gpt.resize_token_embeddings(len(gpt_tokenizer))

# %%
# Set up the training arguments and trainer for GPT
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

gpt_data_collator = DataCollatorForLanguageModeling(tokenizer=gpt_tokenizer, mlm=False)

gpt_training_args = TrainingArguments(
    output_dir="gpt_commonsense_qa",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

gpt_trainer = Trainer(
    model=gpt,
    tokenizer=gpt_tokenizer,
    args=gpt_training_args,
    train_dataset=gpt_tokenized_dataset["train"],
    eval_dataset=gpt_tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    data_collator=gpt_data_collator
)

# %%
# Train GPT
gpt_trainer.train()


# %%
from llm_evals_viz.sweep import EvaluationSweep

models = [
    ['hf', {'pretrained': 'openai-community/openai-gpt'}],
    ['hf', {'pretrained': 'google-bert/bert-base-uncased'}],
]
tasks = ['commonsense_qa']
sweep = EvaluationSweep(models=models, tasks=tasks)

# %%

results = sweep.run()

# %%
sweep.get_configurations()

# %%
for result in results:
    print(f"{result['config']}: {result['results']['commonsense_qa']}")


# %%
