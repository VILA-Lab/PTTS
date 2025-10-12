#!/usr/bin/env python3
"""
deepseek_query.py

Fetch DeepSeek responses for prompts in a CSV and save them into DS_* columns.

Requirements:
- pandas
- openai==0.28 (OpenAI-compatible DeepSeek API)
- tqdm
"""

import pandas as pd
import openai
import time
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import os

# ---------------------- CONFIG ----------------------
# Set DeepSeek API key in environment variable for security
# export DEEPSEEK_API_KEY="your_api_key_here"
openai.api_key = os.getenv("DEEPSEEK_API_KEY")
openai.api_base = "https://api.deepseek.com"

# Input CSV path (should be output from build_prompt_variants.py)
INPUT_CSV = 'DataConstruction/variants.csv' #path/to/your/augmented_prompts.csv'

# Output CSV path
OUTPUT_CSV = 'DataConstruction/DS_responses.csv'

# Map prompt columns to output DS columns
columns_map = {
    "prompting_p6": "DS_p6",
    "prompting_p9": "DS_p9",
    "prompting_p10": "DS_p10",
    "prompting_p12": "DS_p12",
}

# ----------------------------------------------------

def get_completion(prompt):
    """Call DeepSeek API and return (response_text, reasoning_content, full_response)."""
    max_retries = 4
    attempt = 0
    while attempt < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=62000,
            )
            content = response["choices"][0]["message"]["content"].strip()
            # reasoning_content may not always exist
            reasoning_content = getattr(response.choices[0].message, "reasoning_content", "")
            return content, reasoning_content, response
        except Exception as e:
            print(f"Error: {e}")
            attempt += 1
            time.sleep(15)
    return None

def process_row(row_tuple):
    """Process a single row of prompts and return DS results as a dict."""
    idx, row_series = row_tuple
    row_dict = {}
    for prompt_col, ds_col in columns_map.items():
        prompt_value = row_series.get(prompt_col, None)
        if pd.isna(prompt_value) or not isinstance(prompt_value, str):
            row_dict[ds_col] = ""
            row_dict[ds_col + "_reasoning"] = ""
            continue
        result = get_completion(prompt_value)
        if result is None:
            row_dict[ds_col] = ""
            row_dict[ds_col + "_reasoning"] = ""
        else:
            answer, reasoning, _ = result
            row_dict[ds_col] = answer or ""
            row_dict[ds_col + "_reasoning"] = reasoning or ""
    return row_dict

def main():
    # Load CSV
    df = pd.read_csv(INPUT_CSV)

    # Process rows with 3 parallel threads
    with ThreadPool(processes=3) as pool:
        results_per_row = list(
            tqdm(pool.imap(process_row, df.iterrows()), total=len(df), desc="Processing Rows")
        )

    # Assign results back to DataFrame
    for i, row_result in enumerate(results_per_row):
        for col_name, value in row_result.items():
            df.at[df.index[i], col_name] = value

    # Save updated CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] DeepSeek responses saved to '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()
