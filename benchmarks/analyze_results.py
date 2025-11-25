import pandas as pd
import glob
import os
from datetime import datetime

# Find latest results file
results_files = glob.glob("./results/results_*.csv")
if not results_files:
    print("No results files found!")
    exit(1)

latest_file = max(results_files, key=os.path.getmtime)
print(f"Analyzing: {latest_file}\n")

# Load data
df = pd.read_csv(latest_file)

# Ensure tolerance_mode exists (fill with 'boost' for old results if any)
if 'tolerance_mode' not in df.columns:
    df['tolerance_mode'] = 'boost'

# Generate Markdown Report
report = []
report.append("# DSON Benchmark Summary")
report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"**Source:** `{os.path.basename(latest_file)}`")
report.append(f"**Total Runs:** {len(df)}")
report.append("")

# 1. Overall Statistics by Mode
report.append("## 1. Overall Performance by Mode")
mode_stats = df.groupby('tolerance_mode').agg({
    'dson_acc': 'mean',
    'json_acc': 'mean',
    'dson_parsable': 'mean',
    'input_savings_%': 'mean',
    'output_savings_%': 'mean',
    'total_savings_%': 'mean'
}).reset_index()
mode_stats['dson_parsable'] = (mode_stats['dson_parsable'] * 100).round(1)
mode_stats['dson_acc'] = mode_stats['dson_acc'].round(3)
mode_stats['json_acc'] = mode_stats['json_acc'].round(3)
mode_stats['input_savings_%'] = mode_stats['input_savings_%'].round(1)
mode_stats['output_savings_%'] = mode_stats['output_savings_%'].round(1)
mode_stats['total_savings_%'] = mode_stats['total_savings_%'].round(1)

report.append(mode_stats.to_markdown(index=False))
report.append("")

# 2. Per Model Performance (Boost Mode)
report.append("## 2. Model Performance (Boost Mode)")
boost_df = df[df['tolerance_mode'] == 'boost']
model_stats = boost_df.groupby('model').agg({
    'dson_acc': 'mean',
    'json_acc': 'mean',
    'dson_parsable': 'mean',
    'input_savings_%': 'mean',
    'output_savings_%': 'mean',
    'total_savings_%': 'mean'
}).reset_index().sort_values('dson_acc', ascending=False)

model_stats['dson_parsable'] = (model_stats['dson_parsable'] * 100).round(1)
model_stats['dson_acc'] = model_stats['dson_acc'].round(3)
model_stats['json_acc'] = model_stats['json_acc'].round(3)
model_stats['input_savings_%'] = model_stats['input_savings_%'].round(1)
model_stats['output_savings_%'] = model_stats['output_savings_%'].round(1)
model_stats['total_savings_%'] = model_stats['total_savings_%'].round(1)

report.append(model_stats.to_markdown(index=False))
report.append("")

# 3. Strict vs Boost Comparison
report.append("## 3. Strict vs Boost Parsability")
pivot = df.pivot_table(index='model', columns='tolerance_mode', values='dson_parsable', aggfunc='mean')
pivot = (pivot * 100).round(1)
report.append(pivot.to_markdown())
report.append("")

# 4. Per Data Type Performance
report.append("## 4. Performance by Data Type")
dtype_stats = df.groupby(['dtype', 'tolerance_mode']).agg({
    'dson_acc': 'mean',
    'dson_parsable': 'mean'
}).reset_index()
dtype_stats['dson_parsable'] = (dtype_stats['dson_parsable'] * 100).round(1)
dtype_stats['dson_acc'] = dtype_stats['dson_acc'].round(3)

report.append(dtype_stats.to_markdown(index=False))
report.append("")

# 5. Low Accuracy Cases (< 0.85 in Boost Mode)
report.append("## 5. Low Accuracy Cases (Boost Mode < 0.85)")
low_acc = boost_df[boost_df['dson_acc'] < 0.85][['model', 'dtype', 'example_idx', 'dson_acc']].sort_values('dson_acc')
if len(low_acc) > 0:
    report.append(low_acc.to_markdown(index=False))
else:
    report.append("No low accuracy cases found!")
report.append("")

# Write to file
output_path = "benchmarks/results/benchmarks_summary.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"Report generated: {output_path}")
