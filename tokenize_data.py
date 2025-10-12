from typing import Dict
import re
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial

QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_cot_example(
    example: Dict,
    tokenizer,
):
    raw_trace = example["deepseek_thinking_trajectory"]
    thinking_trajectory = [ raw_trace ]
    question = example["question"]
    answer = example["deepseek_attempt"]

    prompt = QUERY_TEMPLATE_NOANSWER.format(Question=question)
    answer = "Answer: " + answer if "Answer:" not in answer else answer
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
        {
            "role": "assistant", 
            "content": "<|im_start|>think\n" + "\n".join(thinking_trajectory).strip() + "\n<|im_start|>answer\n" + answer.strip()
        }
    ], tokenize=False)
    return dict(text=text)

def mathcot_sft(upload_data_path: str, num_proc: int, download_data_path: str):
    # Resolve absolute paths relative to the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, download_data_path)
    output_path = os.path.join(script_dir, upload_data_path)

    # Load dataset
    # dataset = load_dataset("csv", data_files={"train": csv_path})["train"]
    # Load dataset: local CSV if it exists, otherwise treat as HF Hub dataset id
    if os.path.exists(csv_path) or download_data_path.lower().endswith(".csv"):
        dataset = load_dataset("csv", data_files={"train": csv_path})["train"]
    else:
        ds = load_dataset(download_data_path)  # e.g., "P-TTS/P_TTS-Full" (login if gated)
        if hasattr(ds, "keys"):  # DatasetDict
            split = "train" if "train" in ds.keys() else next(iter(ds.keys()))
            dataset = ds[split]
        else:  # already a Dataset
            dataset = ds


    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
    process_example_map = partial(process_cot_example, tokenizer=tokenizer)
    dataset = dataset.map(
        process_example_map,
        num_proc=num_proc,
        desc="Tokenizing SFT data",
    )
    dataset.select_columns(["deepseek_thinking_trajectory","question","deepseek_attempt",'text']).to_csv(output_path)



if __name__ == "__main__":
    mathcot_sft(
        download_data_path="P-TTS/P_TTS-Full",
        upload_data_path="Deepseek_900_32B_tokonized.csv",
        num_proc=20
    )
