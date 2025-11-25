import os
import json
import time
import sqlite3
import argparse
import re
import logging
from typing import List, Dict, Any, Tuple, Type
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from difflib import SequenceMatcher
from pydantic import BaseModel
import sys
sys.path.insert(0, '.')

from benchmarks.data_gen import generate_neutral_input
from src.dson.core import format_instructions, parse, parse_list

load_dotenv()

# Ensure benchmarks output directory exists
BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# Setup logging
log_filename = os.path.join(BENCHMARK_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Force UTF-8 for stdout/stderr on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from pydantic import BaseModel, create_model, Field
from typing import List, Dict, Any, Tuple, Type

# Flat model definitions matching benchmark schema
class TabularFlat(BaseModel):
    id: int
    name: str
    cat: str
    price: float
    stock: bool

class TextHeavyFlat(BaseModel):
    id: str
    user: str
    text: str

class NestedFlat(BaseModel):
    uid: str
    profile_name: str
    profile_city: str
    tags: List[str]

def get_flat_model(dtype: str) -> Type[BaseModel]:
    """Return a Pydantic model that matches the flattened DSON schema for the given dtype."""
    if dtype == "Tabular":
        return TabularFlat
    elif dtype == "Text-Heavy":
        return TextHeavyFlat
    elif dtype == "Nested":
        return NestedFlat
    else:
        raise ValueError(f"Unsupported dtype for flat model: {dtype}")


def init_db_v2(db_path=None):
    if db_path is None:
        db_path = os.path.join(BENCHMARK_DIR, "benchmarks.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS runs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  model TEXT,
                  dtype TEXT,
                  example_idx INTEGER,
                  format TEXT,
                  prompt_tokens INTEGER,
                  completion_tokens INTEGER,
                  total_tokens INTEGER,
                  exact_match BOOLEAN,
                  structural_accuracy REAL,
                  text_similarity REAL,
                  composite_accuracy REAL,
                  parsable BOOLEAN,
                  parse_error TEXT,
                  response TEXT,
                  tolerance_mode TEXT,
                  duration REAL)''')
    conn.commit()
    return conn

class EnhancedBenchmarkRunner:
    def __init__(self, api_key: str, db_path: str = None, verbose: bool = False):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        # Set default DB path if not provided
        if db_path is None:
            db_path = os.path.join(BENCHMARK_DIR, "benchmarks.db")
        self.db_path = db_path
        self.verbose = verbose
        
        # Initialize database schema (main thread)
        init_db_v2(db_path)
        
    def get_db_connection(self):
        """Get a thread-local database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_tag(self, dtype: str) -> str:
        """Get tag for data type - using %D to save tokens"""
        return "%D"
    
    def get_prompts(self, target_format: str, dtype: str, neutral_data: str, model_cls: Type[BaseModel]) -> Tuple[str, str, int]:
        """Get prompts and return (system, user, prompt_token_estimate)"""
        tag = self.get_tag(dtype)
        
        if target_format == "JSON":
            sys_prompt = (
                "ROLE: Data Formatter.\n"
                "TASK: Convert the input raw data into a Minified JSON list of objects.\n"
                "RULES: Output ONLY JSON. No Markdown. No chatter."
            )
            user_prompt = f"Input Data:\n{neutral_data}\n\nOutput JSON:"
        else:  # DSON
            # Use improved format_instructions
            sys_prompt = format_instructions(model_cls, tag)
            user_prompt = f"Input Data:\n{neutral_data}\n\nOutput DSON:"
        
        # Rough token estimate (4 chars = 1 token)
        prompt_token_estimate = (len(sys_prompt) + len(user_prompt)) // 4
        
        return sys_prompt, user_prompt, prompt_token_estimate

    def run_single(self, model_name: str, dtype: str, ex_idx: int) -> Dict[str, Any]:
        """Run a single benchmark example for the given model and data type.

        Returns a dict with metrics and response information.
        """
        # Generate neutral input examples and pick the requested index
        examples = generate_neutral_input(dtype, count=ex_idx + 1)
        example = examples[ex_idx]
        neutral_data = example["prompt_text"]
        ground_truth = example["ground_truth"]

        # Get the flat Pydantic model for this dtype
        model_cls = get_flat_model(dtype)

        # 1. Run JSON Baseline
        json_sys, json_user, json_est = self.get_prompts("JSON", dtype, neutral_data, model_cls)
        start = time.time()
        try:
            json_resp = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": json_sys}, {"role": "user", "content": json_user}],
                temperature=0.0,
                max_tokens=1024,
            )
            json_dur = time.time() - start
            json_text = json_resp.choices[0].message.content.strip()
            json_usage = getattr(json_resp, "usage", None)
            json_toks = json_usage.total_tokens if json_usage else json_est
            
            # Parse JSON
            can_parse_json, json_err, json_obj = self.test_parsability(json_text, dtype, "JSON", model_cls)
            json_exact = self.calc_exact_match(ground_truth, json_obj)
            json_struct = self.calc_structural_accuracy(ground_truth, json_obj)
            json_sim = self.calc_text_similarity(ground_truth, json_text)
            json_acc = (json_exact + json_struct + json_sim) / 3.0
            
            # Parsable = can parse AND exact match
            json_parsable = can_parse_json and json_exact
        except Exception as e:
            json_dur = time.time() - start
            json_text = ""
            json_toks = 0
            json_parsable = False
            json_acc = 0.0
            json_obj = None
            json_exact = False
            json_struct = 0.0
            json_sim = 0.0

        # 2. Run DSON (Dual Mode)
        dson_sys, dson_user, dson_est = self.get_prompts("DSON", dtype, neutral_data, model_cls)
        start = time.time()
        dson_resp = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": dson_sys}, {"role": "user", "content": dson_user}],
            temperature=0.0,
            max_tokens=1024,
        )
        dson_dur = time.time() - start
        dson_text = dson_resp.choices[0].message.content.strip()
        dson_usage = getattr(dson_resp, "usage", None)
        dson_toks = dson_usage.total_tokens if dson_usage else dson_est
        
        # Calculate Savings
        json_prompt = json_usage.prompt_tokens if json_usage else json_est
        json_compl = json_usage.completion_tokens if json_usage else 0
        
        dson_prompt = dson_usage.prompt_tokens if dson_usage else dson_est
        dson_compl = dson_usage.completion_tokens if dson_usage else 0
        
        input_savings = ((json_prompt - dson_prompt) / json_prompt * 100) if json_prompt > 0 else 0.0
        output_savings = ((json_compl - dson_compl) / json_compl * 100) if json_compl > 0 else 0.0
        total_savings = ((json_toks - dson_toks) / json_toks * 100) if json_toks > 0 else 0.0

        # Calculate DSON results for both boost and strict modes
        results = []
        for mode in ["boost", "strict"]:
            can_parse, parse_err, parsed_obj = self.test_parsability(dson_text, dtype, "DSON", model_cls, mode)
            
            # Calculate accuracy metrics (fuzzy match)
            exact = self.calc_exact_match(ground_truth, parsed_obj)
            structural = self.calc_structural_accuracy(ground_truth, parsed_obj)
            similarity = self.calc_text_similarity(ground_truth, dson_text)
            composite = (exact + structural + similarity) / 3.0
            
            # Parsable = can parse AND exact match (strict correctness)
            parsable = can_parse and exact
            
            results.append({
                "mode": mode,
                "parsable": parsable,  # TRUE only if exact match
                "parse_err": parse_err if not can_parse else ("" if exact else "Data mismatch"),
                "exact": exact,
                "structural": structural,
                "similarity": similarity,
                "composite": composite  # Fuzzy accuracy score
            })

        # Store result in DB (one row per mode: JSON, DSON-boost, DSON-strict)
        conn = self.get_db_connection()
        cur = conn.cursor()
        
        # Store JSON result
        if json_parsable:
            cur.execute(
                "INSERT INTO runs (timestamp, model, dtype, example_idx, format, prompt_tokens, completion_tokens, total_tokens, exact_match, structural_accuracy, text_similarity, composite_accuracy, parsable, parse_error, response, tolerance_mode, duration) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    datetime.now().isoformat(),
                    model_name,
                    dtype,
                    ex_idx,
                    "JSON",
                    json_prompt,
                    json_compl,
                    json_toks,
                    json_exact,
                    json_struct,
                    json_sim,
                    json_acc,
                    json_parsable,
                    "",
                    json_text,
                    "N/A",  # tolerance_mode not applicable for JSON
                    json_dur,
                ),
            )
        
        # Store DSON results (boost and strict)
        for res in results:
            cur.execute(
                "INSERT INTO runs (timestamp, model, dtype, example_idx, format, prompt_tokens, completion_tokens, total_tokens, exact_match, structural_accuracy, text_similarity, composite_accuracy, parsable, parse_error, response, tolerance_mode, duration) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    datetime.now().isoformat(),
                    model_name,
                    dtype,
                    ex_idx,
                    "DSON",
                    dson_prompt,
                    dson_compl,
                    dson_toks,
                    res["exact"],
                    res["structural"],
                    res["similarity"],
                    res["composite"],
                    res["parsable"],
                    res["parse_err"],
                    dson_text,
                    res["mode"],
                    dson_dur,
                ),
            )
        conn.commit()
        conn.close()

        # Return a summary dict (using boost mode for display)
        boost_res = results[0] if results[0]["mode"] == "boost" else results[1]
        return {
            "model": model_name,
            "dtype": dtype,
            "example_idx": ex_idx,
            "format": "DSON",
            "prompt_tokens": dson_prompt,
            "completion_tokens": dson_compl,
            "total_tokens": dson_toks,
            "dson_acc": boost_res["composite"],
            "dson_parsable": boost_res["parsable"],
            "output_savings_%": output_savings,
            "input_savings_%": input_savings,
            "total_savings_%": total_savings,
            "json_acc": json_acc,
            "json_parsable": json_parsable,
            "response": dson_text,
            "duration": dson_dur,
        }

    def test_parsability(self, response: str, dtype: str, format_type: str, model_cls: Type[BaseModel], tolerance_mode: str = "boost") -> Tuple[bool, str, Any]:
        """Test if the response can be parsed into the expected model."""
        if format_type == "JSON":
            try:
                # Strip markdown code blocks
                clean_resp = re.sub(r'```(?:json)?\n?', '', response)
                clean_resp = re.sub(r'```', '', clean_resp).strip()
                
                # Handle JSON list or single object
                parsed = json.loads(clean_resp)
                if isinstance(parsed, list):
                    # Convert to list of model instances
                    obj = [model_cls(**item) for item in parsed]
                else:
                    obj = [model_cls(**parsed)]
                return True, "", obj
            except Exception as e:
                return False, str(e), None
        else:  # DSON
            try:
                tag = self.get_tag(dtype)
                # Use parse_list to handle multiple objects
                obj = parse_list(response, model_cls, tag=tag, tolerance_mode=tolerance_mode)
                return True, "", obj
            except Exception as e:
                return False, str(e), None

    def calc_exact_match(self, ground_truth: List[Dict], parsed_obj: Any) -> bool:
        """Check if parsed object exactly matches ground truth (order independent)"""
        if not parsed_obj: return False
        try:
            # Convert parsed Pydantic objects to dicts
            if isinstance(parsed_obj, list):
                parsed_dicts = [p.model_dump() if hasattr(p, 'model_dump') else p for p in parsed_obj]
            else:
                parsed_dicts = [parsed_obj.model_dump()]
            
            if len(parsed_dicts) != len(ground_truth):
                return False
            
            # Sort both lists by ID to ensure order independence
            # Identify ID field
            id_field = 'id'
            if 'uid' in ground_truth[0]: id_field = 'uid'
            
            def get_sort_key(item):
                val = item.get(id_field)
                return str(val)

            sorted_gt = sorted(ground_truth, key=get_sort_key)
            return sorted_parsed == sorted_gt
        except:
            return False
    
    def calc_structural_accuracy(self, ground_truth: List[Dict], parsed_obj: Any) -> float:
        """Field-by-field accuracy comparison (order independent)"""
        if not parsed_obj: return 0.0
        try:
            if isinstance(parsed_obj, list):
                parsed_dicts = [p.model_dump() if hasattr(p, 'model_dump') else p for p in parsed_obj]
            else:
                parsed_dicts = [parsed_obj.model_dump()]
            
            if len(parsed_dicts) != len(ground_truth):
                return 0.0
            
            # Sort both lists by ID
            id_field = 'id'
            if 'uid' in ground_truth[0]: id_field = 'uid'
            
            def get_sort_key(item):
                val = item.get(id_field)
                return str(val)

            sorted_gt = sorted(ground_truth, key=get_sort_key)
            sorted_parsed = sorted(parsed_dicts, key=get_sort_key)
            
            total_fields = 0
            matching_fields = 0
            
            for gt_item, parsed_item in zip(sorted_gt, sorted_parsed):
                for key in gt_item.keys():
                    total_fields += 1
                    if key in parsed_item and parsed_item[key] == gt_item[key]:
                        matching_fields += 1
            
            return matching_fields / total_fields if total_fields > 0 else 0.0
        except:
            return 0.0

    def calc_text_similarity(self, ground_truth: List[Dict], response: str) -> float:
        """Existing difflib-based similarity"""
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
        response_lower = response.lower()
        return SequenceMatcher(None, target_blob, response_lower).ratio()

def run_sanity_check(api_key: str):
    """Run a quick sanity check on one example of each type."""
    print("\n🧪 Running Sanity Check...")
    runner = EnhancedBenchmarkRunner(api_key, verbose=True)
    
    dtypes = ["Tabular", "Text-Heavy", "Nested"]
    model = "google/gemini-2.5-flash"
    
    for dtype in dtypes:
        print(f"\nChecking {dtype}...")
        try:
            result = runner.run_single(model, dtype, 0)
            print(f"  Response length: {len(result['response'])}")
            print(f"  Parsable: {result['dson_parsable']}")
            print(f"  Accuracy: {result['dson_acc']:.2f}")
            print(f"  Input Overhead: {result['input_overhead_%']:.1f}%")
            print(f"  Output Savings: {result['output_savings_%']:.1f}%")
            
            if not result['dson_parsable']:
                print(f"  ❌ Failed to parse! Error: {result.get('parse_error', 'Unknown')}")
                print(f"  Response preview: {result['response'][:100]}...")
            else:
                print(f"  ✅ Passed")
        except Exception as e:
            print(f"  ❌ Exception: {e}")
        
        print("Sleeping 5s to avoid rate limits...")
        time.sleep(5)

    print("\nSanity check complete.")

def main():
    parser = argparse.ArgumentParser(description="Enhanced DSON Benchmark")
    parser.add_argument("--models", nargs="+", default=[
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemini-2.5-flash",
        "gpt-4o",
        "gpt-4.1-mini",
        "anthropic/claude-haiku-4.5"
    ], help="Models to benchmark")
    parser.add_argument("--dtypes", nargs="+", default=["Tabular", "Text-Heavy", "Nested"], help="Data types to test")
    parser.add_argument("--examples", type=int, default=10, help="Number of examples per type")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite DB")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--sanity", action="store_true", help="Run a quick sanity check")
    
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment.")
        print("Error: OPENROUTER_API_KEY not found.")
        return

    if args.sanity:
        run_sanity_check(api_key)
        return

    logger.info(f"Starting enhanced benchmark with {args.workers} parallel workers")
    logger.info(f"Results will be saved to: {args.db or 'benchmarks/results/benchmarks.db'}")
    logger.info(f"Log file: {log_filename}")
    
    runner = EnhancedBenchmarkRunner(api_key, args.db, args.verbose)
    
    all_results = []
    
    # Create all tasks
    tasks = []
    for model in args.models:
        for dtype in args.dtypes:
            for ex_idx in range(args.examples):
                tasks.append((model, dtype, ex_idx))
    
    total_runs = len(tasks)
    logger.info(f"Total benchmark runs: {total_runs}")
    print(f"\n🚀 Starting Enhanced Benchmark")
    print(f"   Models: {', '.join(args.models)}")
    print(f"   Types: {', '.join(args.dtypes)}")
    print(f"   Examples: {args.examples}")
    print(f"   Total runs: {total_runs}")
    print(f"   Workers: {args.workers}")
    print("=" * 80)
    
    # Execute in parallel
    run_count = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(runner.run_single, model, dtype, ex_idx): (model, dtype, ex_idx)
            for model, dtype, ex_idx in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            model, dtype, ex_idx = future_to_task[future]
            run_count += 1
            
            try:
                res = future.result()
                if "error" not in res:
                    all_results.append(res)
                    parse_status = "✓" if res['dson_parsable'] else "✗"
                    elapsed = time.time() - start_time
                    eta = (elapsed / run_count) * (total_runs - run_count)
                    print(f"[{run_count}/{total_runs}] {model[:25]:25} | {dtype:12} | Out Sav: {res['output_savings_%']:5.1f}% | In Load: {res['input_overhead_%']:5.1f}% | Acc: {res['dson_acc']:.3f} | Parse: {parse_status} | ETA: {eta/60:.1f}m")
                else:
                    logger.error(f"Run failed: {model} {dtype}-{ex_idx}: {res['error']}")
                    print(f"[{run_count}/{total_runs}] ERROR: {model} {dtype}-{ex_idx}")
            except Exception as e:
                logger.error(f"Task execution failed: {model} {dtype}-{ex_idx}: {str(e)}")
                print(f"[{run_count}/{total_runs}] EXCEPTION: {model} {dtype}-{ex_idx}")

    total_time = time.time() - start_time
    logger.info(f"Benchmark completed in {total_time:.2f}s ({total_time/60:.2f}m)")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 ENHANCED BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total time: {total_time/60:.2f} minutes")
    
    if all_results:
        import pandas as pd
        df = pd.DataFrame(all_results)
        
        print(f"\n✅ Successful runs: {len(df)}")
        print(f"\n💰 COST ARBITRAGE (The DSON Advantage):")
        print(f"  Input Overhead:     {df['input_overhead_%'].mean():.2f}% (Cheaper tokens)")
        print(f"  Output Savings:     {df['output_savings_%'].mean():.2f}% (Expensive tokens)")
        print(f"  Net Token Savings:  {df['total_savings_%'].mean():.2f}%")
        
        print(f"\n🎯 ACCURACY:")
        print(f"  DSON composite:     {df['dson_acc'].mean():.3f}")
        print(f"  JSON composite:     {df['json_acc'].mean():.3f}")
        
        print(f"\n✓ PARSABILITY (Exact Match):")
        print(f"  DSON parsable:      {df['dson_parsable'].sum()}/{len(df)} ({df['dson_parsable'].sum()/len(df)*100:.1f}%)")
        print(f"  JSON parsable:      {df['json_parsable'].sum()}/{len(df)} ({df['json_parsable'].sum()/len(df)*100:.1f}%)")
        
        # Save detailed results
        csv_filename = os.path.join(BENCHMARK_DIR, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(csv_filename, index=False)
        logger.info(f"Detailed results saved to: {csv_filename}")
        
        # Generate and save summary metrics
        summary_filename = os.path.join(BENCHMARK_DIR, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(summary_filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DSON BENCHMARK SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Runs: {len(df)}\n")
            f.write(f"Benchmark Duration: {total_time/60:.2f} minutes\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("COST ARBITRAGE (The DSON Advantage)\n")
            f.write("-" * 40 + "\n")
            f.write("DSON trades cheap input tokens for expensive output tokens.\n\n")
            f.write(f"Input Overhead:     {df['input_overhead_%'].mean():.2f}% (Cheaper tokens)\n")
            f.write(f"Output Savings:     {df['output_savings_%'].mean():.2f}% (Expensive tokens)\n")
            f.write(f"Net Token Savings:  {df['total_savings_%'].mean():.2f}%\n\n")
            
            f.write("ACCURACY (Fuzzy Match Score)\n")
            f.write("-" * 40 + "\n")
            f.write(f"DSON Average:       {df['dson_acc'].mean():.3f}\n")
            f.write(f"JSON Average:       {df['json_acc'].mean():.3f}\n\n")
            
            f.write("PARSABILITY (Exact Match Rate)\n")
            f.write("-" * 40 + "\n")
            f.write(f"DSON Exact Match:   {df['dson_parsable'].sum()}/{len(df)} ({df['dson_parsable'].sum()/len(df)*100:.1f}%)\n")
            f.write(f"JSON Exact Match:   {df['json_parsable'].sum()}/{len(df)} ({df['json_parsable'].sum()/len(df)*100:.1f}%)\n\n")
            
            f.write("PER-MODEL BREAKDOWN\n")
            f.write("-" * 40 + "\n")
            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                f.write(f"\n{model}:\n")
                f.write(f"  DSON Exact Match: {model_df['dson_parsable'].sum()}/{len(model_df)} ({model_df['dson_parsable'].sum()/len(model_df)*100:.1f}%)\n")
                f.write(f"  DSON Accuracy:    {model_df['dson_acc'].mean():.3f}\n")
                f.write(f"  JSON Exact Match: {model_df['json_parsable'].sum()}/{len(model_df)} ({model_df['json_parsable'].sum()/len(model_df)*100:.1f}%)\n")
                f.write(f"  JSON Accuracy:    {model_df['json_acc'].mean():.3f}\n")
                f.write(f"  Output Savings:   {model_df['output_savings_%'].mean():.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("METRIC DEFINITIONS:\n")
            f.write("- Input Overhead: Extra tokens used in prompt (cheaper)\n")
            f.write("- Output Savings: Reduction in generated tokens (expensive)\n")
            f.write("- Parsable (Exact Match): Output parsed correctly AND matches ground truth 100%\n")
            f.write("- Accuracy (Fuzzy): Score 0.0-1.0 showing how close output is to ground truth\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n📁 Results saved to:")
        print(f"   CSV:     {csv_filename}")
        print(f"   Summary: {summary_filename}")
        print(f"   DB:      {runner.db_path}")
        print(f"   Log:     {log_filename}")
        print(f"\n📊 View summary: cat {summary_filename}")

if __name__ == "__main__":
    main()
