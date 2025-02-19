import datasets
import os 

from pathlib import Path

def prepare_triviaqa(save_dir):
    # Load the dataset
    ds = datasets.load_dataset("trivia_qa", "rc.nocontext")
    # ds["train"] = ds["train"].select(range(7000))  # Selects first 7000 samples
    # Function to process each example
    def process_example(example):
        # Extract 'aliases' from the 'answer' field and concatenate them
        concatenated_aliases = "; ".join(example["answer"]["aliases"])
        example["answer"] = concatenated_aliases  # Replace answer with the concatenated aliases
        return example

    # Apply transformation to all splits
    ds = ds.map(process_example)
    # Save the modified dataset
    ds.save_to_disk(str(Path(save_dir)))
    print(ds)
    return ds

def prepare_coqa(save_dir):
    ds = datasets.load_dataset("stanfordnlp/coqa")
    # print(ds)
    ds.save_to_disk(str(Path(save_dir)))
    return ds

def load_local_dataset(save_dir):
    """
    Loads a dataset from disk.
    """
    if os.path.exists(save_dir):
        print(f"Loading dataset from {save_dir} ...")
        return datasets.load_from_disk(str(save_dir))
    else: 
        if 'coqa' in save_dir:
            return prepare_coqa(save_dir)
        elif 'trivia_qa' in save_dir:
            return prepare_triviaqa(save_dir)
        else:
            print(f'ERROR, unknown dataset {save_dir}')
            return None
    
def flatten_coqa_dataset(coqa_dataset):
    """
    Flattens a CoQA dataset by converting rows with multiple questions and answers
    into multiple rows with one question-answer pair each.

    For each original example, this function expands it into one example per question-answer
    pair while preserving the other fields (e.g., 'source' and 'story').

    Parameters:
        coqa_dataset (datasets.Dataset or datasets.DatasetDict):
            The CoQA dataset to flatten.

    Returns:
        A new flattened dataset (or DatasetDict if the input was a DatasetDict)
        where each example contains a single question and its corresponding answer.
    """
    def expand_example(example):
        # Copy fields that are common across all question-answer pairs.
        base = {k: v for k, v in example.items() if k not in ["questions", "answers"]}
        questions = example["questions"]
        answers = example["answers"]
        expanded_examples = []
        # Expand each question-answer pair into its own example.
        for i, question in enumerate(questions):
            if len(question.split()) <= 3: # Remove simple but vague questions, e.g., 'where is it?'
                continue
            new_example = base.copy()
            new_example["question"] = question
            new_example["answer"] = answers["input_text"][i]
            new_example["answer_start"] = answers["answer_start"][i]
            new_example["answer_end"] = answers["answer_end"][i]
            expanded_examples.append(new_example)
            break # For now, only use one question for each context/passage
        return expanded_examples

    # Check if the input is a DatasetDict (with splits) or a single Dataset.
    if isinstance(coqa_dataset, datasets.DatasetDict):
        flattened_splits = {}
        for split, ds in coqa_dataset.items():
            new_examples = []
            for example in ds:
                new_examples.extend(expand_example(example))
            flattened_splits[split] = datasets.Dataset.from_list(new_examples)
        return datasets.DatasetDict(flattened_splits)
    else:
        # If it's a single Dataset, just iterate and collect all new examples.
        new_examples = []
        for example in coqa_dataset:
            new_examples.extend(expand_example(example))
        return datasets.Dataset.from_list(new_examples)