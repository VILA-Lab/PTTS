#!/usr/bin/env python3
"""
combine_deepseek_data.py

This script combines question prompts, DeepSeek responses,
and reasoning traces from multiple prompt variants into
a single CSV for training.

Usage:
    python combine_deepseek_data.py
"""

import pandas as pd

# -----------------------------
# Load datasets
# -----------------------------
# Main dataset with original problems and prompt variants
df1 = pd.read_csv('DataConstruction/DS_responses.csv')   # replace with actual path

# If you have a third dataset (e.g., extra prompt versions)
# uncomment and load it
# df3 = pd.read_csv('Deepseek/M1_data_extra.csv')

# -----------------------------
# Combine Question Variants
# -----------------------------
combined_q = pd.concat([
    df1['problem'],
    df1['prompting_p6'],
    df1['prompting_p6_v2'],
    df1['prompting_p6_v3'],
    df1['prompting_p6_v4'],
    df1['prompting_p6_v5'],
    df1['prompting_p6_v6'],  
    df1['prompting_p9'],
    df1['prompting_p10'],
    df1['prompting_p12'],
], ignore_index=True)

# -----------------------------
# Combine DeepSeek Responses
# -----------------------------
combined_resp_deep = pd.concat([
    df1['Deepseek_R1'],
    df1['DeepSeek_p6'],
    df1['DeepSeek_p6_v2'],
    df1['DeepSeek_p6_v3'],
    df1['DeepSeek_p6_v4'],
    df1['DeepSeek_p6_v5'],
    df1['DeepSeek_p6_v6'],  
    df1['DeepSeek_p9'],
    df1['DeepSeek_p10'],
    df1['DeepSeek_p12'],
], ignore_index=True)

# -----------------------------
# Combine Reasoning Trajectories
# -----------------------------
combined_thinking_deep = pd.concat([
    df1['Deepseek_R1_reasoning'],
    df1['DeepSeek_p6_reasoning'],
    df1['DeepSeek_p6_v2_reasoning'],
    df1['DeepSeek_p6_v3_reasoning'],
    df1['DeepSeek_p6_v4_reasoning'],
    df1['DeepSeek_p6_v5_reasoning'],
    df1['DeepSeek_p6_v6_reasoning'],  
    df1['DeepSeek_p9_reasoning'],
    df1['DeepSeek_p10_reasoning'],
    df1['DeepSeek_p12_reasoning'],
], ignore_index=True)

# -----------------------------
# Build Final DataFrame
# -----------------------------
final = pd.DataFrame({
    'question': combined_q,
    'deepseek_attempt': combined_resp_deep,
    'deepseek_thinking_trajectory': combined_thinking_deep,
})

# -----------------------------
# Save to CSV
# -----------------------------
output_path = 'Full-PTTS.csv'
final.to_csv(output_path, index=False)
print(f"✅ Combined dataset saved to: {output_path}")
