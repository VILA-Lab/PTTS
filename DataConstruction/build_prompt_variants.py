#!/usr/bin/env python3
"""
build_prompt_variants.py

Create P-TTS prompt variants as new columns in a CSV.

- Reads an input CSV with a 'problem' column (or a specified column name).
- Adds prompting_* and PTTS_* columns.
- Writes the result to an output CSV.

Usage:
    python build_prompt_variants.py --in path/to/input.csv --out path/to/output.csv

"""

import argparse
import sys
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Add P-TTS prompt variant columns to a CSV.")
    parser.add_argument("--input", dest="inp", required=True,
                        help="Path to input CSV (must have a 'problem' column or specify with --problem-col).")
    parser.add_argument("--out", dest="out", required=True,
                        help="Path to save the augmented CSV.")
    parser.add_argument("--problem-col", default="problem",
                        help="Name of the source problem column (default: 'problem').")
    args = parser.parse_args()

    # Load CSV
    try:
        df = pd.read_csv(args.inp)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if args.problem_col not in df.columns:
        print(f"[ERROR] Column '{args.problem_col}' not found in CSV. "
              f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    base=df['problem']

    # Prompt variants
    df["prompting_p6"] = "I am going to tip $200000 for a better solution! " + base
    df["DS_p6"] = ""

    df["prompting_p9"] = "Your Task is to solve the following: " + base + " You must provide the correct answer!"
    df["DS_p9"] = ""

    df["prompting_p10"] = "You will be penalized if you provide the wrong answer. " + base
    df["DS_p10"] = ""

    df["prompting_p12"] = "Think step by step: " + base
    df["DS_p12"] = ""
    # Save CSV
    try:
        df.to_csv(args.out, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Preview
    print("[OK] Saved:", args.out)
    with pd.option_context("display.max_colwidth", 200):
        print(df[[args.problem_col, "prompting_p6", "prompting_p9", "prompting_p10", "prompting_p12"]].head())

if __name__ == "__main__":
    main()

