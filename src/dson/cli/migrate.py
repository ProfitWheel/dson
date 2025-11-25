import os
import re
import argparse
import glob
from typing import List, Tuple
from openai import OpenAI

# Regex to find JSON-related instructions
JSON_PATTERNS = [
    r"json",
    r"JSON",
    r"output format",
    r"Output format",
    r"response format",
    r"Response format",
    r"return a list of",
    r"Return a list of",
    r"dictionary",
    r"Dictionary"
]

def scan_file(filepath: str) -> List[Tuple[int, str]]:
    matches = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            # Simple heuristic: line contains "json" or "JSON"
            # We want to catch multi-line strings too, but line-by-line is a start.
            # Actually, prompts are often multi-line strings.
            # For V1, let's just flag lines containing keywords.
            for pattern in JSON_PATTERNS:
                if re.search(pattern, line):
                    matches.append((i + 1, line.strip()))
                    break
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return matches

def rewrite_prompt(prompt: str, api_key: str) -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    sys_prompt = (
        "You are a Code Migration Assistant. "
        "Your task is to rewrite a System Prompt or User Prompt to use the DSON protocol instead of JSON.\n"
        "1. PRESERVE all intent, logic, and data requirements.\n"
        "2. REMOVE specific JSON formatting instructions (e.g. 'Return JSON', 'Use keys: ...', 'Format: {\"a\": 1}').\n"
        "3. REPLACE them with a placeholder: '{format_instructions}'.\n"
        "4. If the prompt defines a schema structure textually, keep the conceptual description but remove JSON syntax.\n"
        "5. Output ONLY the rewritten prompt text. Do not add quotes or markdown unless they were in the original."
    )
    
    try:
        resp = client.chat.completions.create(
            model="google/gemini-2.5-flash", # Fast & Cheap
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Original Prompt:\n{prompt}"}
            ],
            temperature=0
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error rewriting: {e}"

def main():
    parser = argparse.ArgumentParser(description="DSON Migration Tool")
    parser.add_argument("--folder", required=True, help="Folder to scan")
    parser.add_argument("--key", help="OpenRouter API Key (optional if env var set)")
    parser.add_argument("--auto", action="store_true", help="Auto-apply changes (dangerous)")
    args = parser.parse_args()

    api_key = args.key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: API Key required for rewriting.")
        return

    print(f"Scanning {args.folder}...")
    
    files = glob.glob(f"{args.folder}/**/*.py", recursive=True) + glob.glob(f"{args.folder}/**/*.txt", recursive=True)
    
    for filepath in files:
        matches = scan_file(filepath)
        if matches:
            print(f"\n📄 Found candidates in {filepath}:")
            for line_num, content in matches:
                print(f"  L{line_num}: {content[:80]}...")
            
            if args.auto:
                # TODO: Implement auto-replace logic (complex for multi-line strings)
                # For now, just listing.
                pass
            else:
                # Interactive mode could go here
                pass

if __name__ == "__main__":
    main()
