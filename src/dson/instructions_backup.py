# @title **V-SEQ BENCHMARK: NATURAL GENERATION (Neutral Input -> Output)**
# @markdown Tests the 'Native Format Cost' by projecting neutral CSV/Text data into JSON vs V-Seq.

import os
import json
import time
import sqlite3
import pandas as pd
import concurrent.futures
import re
import io
import csv
from typing import List, Dict, Any
from getpass import getpass

# --- 1. SETUP ---
try:
    import openai
    from Levenshtein import ratio as levenshtein_ratio
except ImportError:
    !pip install -q openai python-Levenshtein
    import openai
    from Levenshtein import ratio as levenshtein_ratio

if "OPENROUTER_API_KEY" not in os.environ:
    os.environ["OPENROUTER_API_KEY"] = getpass("Enter OpenRouter API Key: ")

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

# 2025 Pricing Map
MODEL_PRICING = {
    "openai/gpt-4.1": [2.00, 8.00],
    "google/gemini-2.5-flash": [0.15, 0.60],
    "meta-llama/llama-4-maverick": [0.10, 0.40],
    "meta-llama/llama-3.1-8b-instruct": [0.05, 0.10]
}

# --- 2. NEUTRAL DATA FACTORY ---

def generate_neutral_input(dtype: str) -> Dict[str, Any]:
    """
    Generates data in a 'Neutral' format (CSV or Text) to be used as the prompt Input.
    Returns: {'prompt_text': str, 'ground_truth': List[Dict]}
    """
    if dtype == "Tabular":
        # CSV Format
        data = [
            {"id": 101, "name": "Super Widget", "cat": "Hardware", "price": 99.50, "stock": True},
            {"id": 102, "name": "Mega Gadget", "cat": "Hardware", "price": 149.00, "stock": False},
            {"id": 103, "name": "Cable Pack", "cat": "Accessory", "price": 19.99, "stock": True},
            {"id": 104, "name": "Monitor Stand", "cat": "Furniture", "price": 45.00, "stock": True},
            {"id": 105, "name": "LED Strip", "cat": "Lighting", "price": 12.50, "stock": True}
        ]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return {"prompt_text": output.getvalue(), "ground_truth": data, "format": "CSV"}

    elif dtype == "Text-Heavy":
        # Bulleted List Format
        data = [
            {"id": "R1", "user": "alice", "text": "The build quality is fantastic, solid aluminum feel. However, the battery life is mediocre at best."},
            {"id": "R2", "user": "bob", "text": "Shipping was fast. The item arrived well packaged. Setup was a breeze, took less than 5 minutes."},
            {"id": "R3", "user": "charlie", "text": "Terrible software experience. The app crashes constantly and the UI looks like it's from 2010. Avoid."}
        ]
        txt = ""
        for d in data:
            txt += f"- Review {d['id']} by {d['user']}: \"{d['text']}\"\n"
        return {"prompt_text": txt, "ground_truth": data, "format": "Text List"}

    elif dtype == "Nested":
        # Structured Text Block (Hard for CSV, easy for JSON/VSeq)
        data = [
            {"uid": "u1", "profile": {"name": "John", "city": "NY"}, "tags": ["admin", "editor"]},
            {"uid": "u2", "profile": {"name": "Jane", "city": "LA"}, "tags": ["viewer"]},
            {"uid": "u3", "profile": {"name": "Dave", "city": "SF"}, "tags": ["viewer", "guest"]}
        ]
        txt = ""
        for d in data:
            txt += f"User {d['uid']}: Name={d['profile']['name']}, City={d['profile']['city']}. Tags={','.join(d['tags'])}\n"
        return {"prompt_text": txt, "ground_truth": data, "format": "Structured Text"}

    return {}

# --- 3. THE PROJECTION ENGINE ---

class FormatEngine:
    @staticmethod
    def get_prompts(target_format: str, mode: str, dtype: str, neutral_data: str) -> Tuple[str, str]:
        """Returns (System Prompt, User Prompt)"""

        # --- JSON PROMPT ---
        if target_format == "JSON":
            sys = (
                "ROLE: Data Formatter.\n"
                "TASK: Convert the input raw data into a Minified JSON list of objects.\n"
                "RULES: Output ONLY JSON. No Markdown. No chatter."
            )
            user = f"Input Data:\n{neutral_data}\n\nOutput JSON:"
            return sys, user

        # --- V-SEQ PROMPTS ---
        schema_map = {
            "Tabular": "%ITEM:id|name|cat|price|stock",
            "Text-Heavy": "%REV:id|user|text",
            "Nested": "%USER:uid|profile.name|profile.city|tags[]"
        }
        schema = schema_map.get(dtype, "%GENERIC:val")

        if mode == "aggressive":
            # Positional
            ex_map = {
                "Tabular": "Ex: %ITEM|1|Widget|Hard|10.0|True",
                "Text-Heavy": "Ex: %REV|1|user|text content",
                "Nested": "Ex: %USER|1|Name|City|tag1|tag2||"
            }
        else: # Safe
            # Keyed
            ex_map = {
                "Tabular": "Ex: %ITEM|id=1|name=Widget|cat=Hard|price=10.0|stock=True",
                "Text-Heavy": "Ex: %REV|id=1|user=user|text=content...",
                "Nested": "Ex: %USER|uid=1|name=Name|city=City|tags=t1,t2"
            }

        sys = (
            "ROLE: V-Seq Compiler. You are NOT an assistant.\n"
            "TASK: Convert the input raw data into V-Seq format.\n"
            "CONSTRAINTS: NO Markdown. NO Explanations. Start immediately with %.\n"
            f"SCHEMA REGISTRY:\n{schema}\n"
            f"RULES: Separator='|', ArrayEnd='||'\n"
            f"EXAMPLE PATTERN ({mode.upper()}):\n{ex_map.get(dtype)}"
        )
        user = f"Input Data:\n{neutral_data}\n\nOutput V-Seq:"
        return sys, user

    @staticmethod
    def calc_accuracy(ground_truth: List[Dict], response: str) -> float:
        # Flatten Ground Truth Values
        def get_leaves(obj):
            leaves = []
            if isinstance(obj, dict):
                for v in obj.values(): leaves.extend(get_leaves(v))
            elif isinstance(obj, list):
                for v in obj: leaves.extend(get_leaves(v))
            else:
                leaves.append(str(obj).lower())
            return leaves

        targets = get_leaves(ground_truth)
        target_blob = " ".join(targets)

        # Clean Response
        clean = re.sub(r'%[A-Z]+', '', response) # Remove tags
        clean = re.sub(r'[a-z_]+=', '', clean)   # Remove keys
        clean = clean.replace('|', ' ').replace('||', ' ').lower()

        return levenshtein_ratio(target_blob, clean)

# --- 4. RUNNER ---

def run_experiment(model, mode, dtype):
    # 1. Get Neutral Data
    payload = generate_neutral_input(dtype)
    raw_text = payload['prompt_text']
    ground_truth = payload['ground_truth']

    # 2. Run JSON Projection (Baseline)
    sys_j, user_j = FormatEngine.get_prompts("JSON", "N/A", dtype, raw_text)
    try:
        r_j = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_j},{"role":"user","content":user_j}],
            temperature=0, max_tokens=1500
        )
        j_tok = r_j.usage.completion_tokens
        j_resp = r_j.choices[0].message.content
    except Exception as e:
        return {"error": f"JSON Fail: {e}"}

    # 3. Run V-Seq Projection (Experiment)
    sys_v, user_v = FormatEngine.get_prompts("V-Seq", mode, dtype, raw_text)
    try:
        r_v = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_v},{"role":"user","content":user_v}],
            temperature=0, max_tokens=1500, stop=["Input Data:", "\n\nInput"]
        )
        v_tok = r_v.usage.completion_tokens
        v_resp = r_v.choices[0].message.content
    except Exception as e:
        return {"error": f"VSeq Fail: {e}"}

    # 4. Metrics
    acc = FormatEngine.calc_accuracy(ground_truth, v_resp)

    # Clean chatter for fair token count if accuracy is low (heuristic)
    if acc < 0.8:
        clean_v = re.sub(r'^.*?%', '%', v_resp, flags=re.DOTALL) # Try to find start
        clean_v = re.sub(r'\n\n.*$', '', clean_v, flags=re.DOTALL) # Try to find end
        eff_v_tok = len(clean_v) // 3.5
    else:
        eff_v_tok = v_tok

    return {
        "model": model, "mode": mode, "type": dtype,
        "savings": (1 - v_tok/j_tok)*100,
        "accuracy": acc,
        "json_resp": j_resp[:50] + "...",
        "vseq_resp_snippet": v_resp[:100].replace('\n', ' ') + "..."
    }

# --- 5. EXECUTION ---

print("🚀 Starting NATURAL GENERATION Benchmark...")
results = []

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = []
    # Test Models
    models = ["google/gemini-2.5-flash", "meta-llama/llama-3.1-8b-instruct", "openai/gpt-4.1"]

    for m in models:
        for mode in ["aggressive", "safe"]:
            for dtype in ["Tabular", "Text-Heavy", "Nested"]:
                futures.append(executor.submit(run_experiment, m, mode, dtype))

    completed = 0
    for f in concurrent.futures.as_completed(futures):
        res = f.result()
        completed += 1
        if "error" not in res:
            results.append(res)
            # Live Feed
            status = "✅" if res['accuracy'] > 0.8 else "⚠️"
            print(f"[{completed}] {res['model']} ({res['mode']}|{res['type']}): {status} {res['savings']:.1f}% Sav | Acc: {res['accuracy']:.2f}")
        else:
            print(f"Error: {res['error']}")

# --- 6. SUMMARY ---
if results:
    df = pd.DataFrame(results)
    print("\n--- FORMAT TAX SCORECARD (Avg Savings vs JSON) ---")
    print(df.groupby(['model', 'mode'])[['savings', 'accuracy']].mean())

    print("\n--- RAW OUTPUT SAMPLES (Check for Chatter) ---")
    # Show one aggressive tabular sample
    sample = df[(df['mode']=='aggressive') & (df['type']=='Tabular')].iloc[0]
    print(f"Model: {sample['model']}\nV-Seq: {sample['vseq_resp_snippet']}")