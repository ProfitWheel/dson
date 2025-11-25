import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, '.')
from benchmarks.data_gen import generate_neutral_input
from benchmarks.run_bench import get_flat_model, EnhancedBenchmarkRunner

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
runner = EnhancedBenchmarkRunner(api_key)

model_name = "gpt-4o"
dtype = "Tabular"
ex_idx = 0

# Get example data
examples = generate_neutral_input(dtype, count=ex_idx + 1)
example = examples[ex_idx]
neutral_data = example["prompt_text"]
ground_truth = example["ground_truth"]
model_cls = get_flat_model(dtype)

print("=" * 80)
print("GROUND TRUTH (what we expect)")
print("=" * 80)
for i, item in enumerate(ground_truth):
    print(f"Item {i+1}: {item}")

print("\n" + "=" * 80)
print("NEUTRAL INPUT DATA (given to LLM)")
print("=" * 80)
print(neutral_data)

# JSON Prompts
print("\n" + "=" * 80)
print("JSON SYSTEM PROMPT")
print("=" * 80)
json_sys, json_user, _ = runner.get_prompts("JSON", dtype, neutral_data, model_cls)
print(json_sys)

print("\n" + "=" * 80)
print("JSON USER PROMPT")
print("=" * 80)
print(json_user)

# Get JSON response
json_resp = runner.client.chat.completions.create(
    model=model_name,
    messages=[{"role": "system", "content": json_sys}, {"role": "user", "content": json_user}],
    temperature=0.0
)
json_text = json_resp.choices[0].message.content.strip()

print("\n" + "=" * 80)
print("JSON LLM RESPONSE")
print("=" * 80)
print(json_text)

# Parse JSON
json_parsable, json_err, json_obj = runner.test_parsability(json_text, dtype, "JSON", model_cls)
print("\n" + "=" * 80)
print(f"JSON PARSED: {json_parsable}")
print("=" * 80)
if json_parsable:
    for i, item in enumerate(json_obj):
        print(f"Item {i+1}: {item.model_dump()}")
else:
    print(f"ERROR: {json_err}")

# DSON Prompts
print("\n" + "=" * 80)
print("DSON SYSTEM PROMPT")
print("=" * 80)
dson_sys, dson_user, _ = runner.get_prompts("DSON", dtype, neutral_data, model_cls)
print(dson_sys)

print("\n" + "=" * 80)
print("DSON USER PROMPT")
print("=" * 80)
print(dson_user)

# Get DSON response
dson_resp = runner.client.chat.completions.create(
    model=model_name,
    messages=[{"role": "system", "content": dson_sys}, {"role": "user", "content": dson_user}],
    temperature=0.0
)
dson_text = dson_resp.choices[0].message.content.strip()

print("\n" + "=" * 80)
print("DSON LLM RESPONSE")
print("=" * 80)
print(dson_text)

# Parse DSON
dson_parsable, dson_err, dson_obj = runner.test_parsability(dson_text, dtype, "DSON", model_cls, "boost")
print("\n" + "=" * 80)
print(f"DSON PARSED: {dson_parsable}")
print("=" * 80)
if dson_parsable:
    for i, item in enumerate(dson_obj):
        print(f"Item {i+1}: {item.model_dump()}")
else:
    print(f"ERROR: {dson_err}")

# Calculate accuracy
print("\n" + "=" * 80)
print("ACCURACY METRICS")
print("=" * 80)
if json_parsable:
    json_exact = runner.calc_exact_match(ground_truth, json_obj)
    json_struct = runner.calc_structural_accuracy(ground_truth, json_obj)
    print(f"JSON - Exact Match: {json_exact}, Structural: {json_struct:.3f}")

if dson_parsable:
    dson_exact = runner.calc_exact_match(ground_truth, dson_obj)
    dson_struct = runner.calc_structural_accuracy(ground_truth, dson_obj)
    print(f"DSON - Exact Match: {dson_exact}, Structural: {dson_struct:.3f}")
