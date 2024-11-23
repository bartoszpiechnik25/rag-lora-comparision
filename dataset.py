import datasets
import pandas as pd
from transformers import AutoTokenizer
from typing import Tuple

PROMPT_TEMPLATE = [
    {
        "role": "user",
        "content": """Context: {context}\nHere is the question you need to answer. Question: {question}\nInstruction: Answer with the shortest, most precise answer possible. Example: When did the first world war started?\nAnswer: 1914"""
    }
]
DATASET = "rajpurkar/squad"
LLM = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_PROMPT_LEN = 460


def prepare_dataset_for_training() -> Tuple[datasets.Dataset, datasets.Dataset]:
    dataset = datasets.load_dataset(DATASET, split='train')
    tokenizer = AutoTokenizer.from_pretrained(LLM)
    dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(dataset)
            .drop_duplicates(['context'])
    )
    dataset = dataset.filter(lambda x: len(tokenizer.encode(x['context'])) < MAX_PROMPT_LEN)
    print(f'Lenght of filterd dataset: {len(dataset)}')
    template = tokenizer.apply_chat_template(PROMPT_TEMPLATE, tokenize=False, add_generation_prompt=True)

    def map_chat_template_to_question(sample):
        q_c = template.format(context=sample['context'], question=sample['question'])
        targets = sample.get("answers", {}).get("text", [""])[0]
        return {'q_c': q_c, 'target_text': targets} 

    def tokenize_function(examples):
        q_c = template.format(context=examples['context'], question=examples['question'])
        targets = examples.get("answers", {}).get("text", [""])[0]
        model_inputs = tokenizer(q_c, max_length=512, truncation=True, padding="max_length", return_tensors='pt')
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length", return_tensors='pt')['input_ids'].squeeze(0)
        model_inputs['labels'] = labels
        model_inputs['input_ids'] = model_inputs['input_ids'].squeeze(0)
        return model_inputs

    dataset = dataset.map(tokenize_function, num_proc=12, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "labels"])
    dataset = dataset.train_test_split(0.2, shuffle=True, seed=2137)

    return dataset['train'], dataset['test']

if __name__ == '__main__':
    train, test = prepare_dataset_for_training()
    print(len(train), len(test))
    print(train[1])
