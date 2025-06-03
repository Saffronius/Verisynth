#!/usr/bin/env python3
"""
exp_1.py

This script generates a regular expression (regex) for each policy in the
Dataset folder, using the Quacky tool to compare original and LLM-generated policies. 
It accepts a `--size` argument to specify how many examples Quacky should sample 
when synthesizing regexes.

Usage:
  python3 exp_1.py --size <number>
"""

import os
import sys
import time
import json
import re
import logging
import subprocess
from pathlib import Path
from functools import wraps
import argparse

import pandas as pd
from tqdm import tqdm

import anthropic
import openai
from openai import OpenAI
from google import generativeai as genai

# ─── Global Configuration ─────────────────────────────────────────────────────

MAX_ATTEMPTS   = 3
INITIAL_DELAY  = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("exp_1.log"),
        logging.StreamHandler()
    ]
)

def retry(max_attempts: int = MAX_ATTEMPTS, initial_delay: int = INITIAL_DELAY):
    """
    Decorator to retry a function on exception up to max_attempts, doubling delay each time.
    If a 'rate_limit_error' appears in the exception message, back off 60s * attempt number.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            delay = initial_delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if "rate_limit_error" in str(e).lower():
                        backoff = 60 * attempts
                        logging.warning(f"Rate limit hit; backing off for {backoff}s")
                        time.sleep(backoff)
                    else:
                        logging.warning(f"{func.__name__} failed ({e}); retry {attempts}/{max_attempts} after {delay}s")
                        time.sleep(delay)
                    delay *= 2
            raise
        return wrapper
    return decorator

# ─── Client Initialization ───────────────────────────────────────────────────

CLAUDE_API_KEY      = "your-claude-api-key-here"  # Replace with your actual API key
claude_client       = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
CLAUDE_MODEL_NAME   = "claude-sonnet-4-20250514"


POLICY_FOLDER       = Path("Folder containing .json policies")  # replace with yoru actual folder path
QUACKY_PATH         = Path(__file__).parent.parent / "quacky/src/quacky.py"
WORKING_DIR         = QUACKY_PATH.parent

# Quacky-derived files
MODELS_A_PATH       = WORKING_DIR / "P1_not_P2.models"
MODELS_B_PATH       = WORKING_DIR / "not_P1_P2.models"
RESPONSE_A_PATH     = WORKING_DIR / "P1_not_P2.regex"
RESPONSE_B_PATH     = WORKING_DIR / "not_P1_P2.regex"

RESULT_DIR          = Path("exp_1")
RESULT_DIR.mkdir(parents=True, exist_ok=True)ß

MODEL_SLUG          = re.sub(r"[^A-Za-z0-9_\-]+", "_", CLAUDE_MODEL_NAME).strip("_")
RESULT_TABLE_PATH   = RESULT_DIR / f"{MODEL_SLUG}.csv"
PROGRESS_PATH       = RESULT_DIR / f"{MODEL_SLUG}.json"

# ─── Utilities ────────────────────────────────────────────────────────────────

def read_policy_file(path: Path) -> str:
    return path.read_text()

def save_generated_policy(json_text: str, path: Path):
    """Validate and write JSON policy to disk."""
    obj = json.loads(json_text)
    path.write_text(json.dumps(obj, indent=2))
    logging.info(f"Saved generated policy to {path}")

def run_quacky(args: list[str]) -> str:
    cmd = [sys.executable, str(QUACKY_PATH)] + args
    proc = subprocess.run(cmd, cwd=WORKING_DIR, capture_output=True, text=True)
    output = proc.stdout
    if proc.stderr:
        output += "\n" + proc.stderr
    return output

@retry(max_attempts=MAX_ATTEMPTS, initial_delay=INITIAL_DELAY)
def llm_generate_regex(client, model_name: str, examples: str) -> str:
    """
    Ask the LLM to produce exactly one regex (no anchors, no extra text) matching all given strings.
    """
    sys_prompt  = "When asked to give a regex, provide ONLY the regex pattern itself. No anchors or extra text and avoid non-capturing groups."
    user_prompt = f"Generate a single regex matching all of these strings, no explanation:\n\n{examples}"
    resp = client.messages.create(
        model=model_name,
        max_tokens=500,
        system=sys_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return resp.content[0].text.strip()

@retry(max_attempts=MAX_ATTEMPTS, initial_delay=INITIAL_DELAY)
def get_policy_description(policy_content: str) -> str:
    """
    Step 1: Generate a concise, complete description of the original policy.
    """
    try:
        obj = json.loads(policy_content)
        policy_content = json.dumps(obj, separators=(",", ":"))
    except json.JSONDecodeError:
        pass

    prompt = (
        "Generate a concise but complete natural-language description of the following AWS IAM policy, "
        "so that another LLM can reconstruct it exactly. Only return the description.\n\n"
        f"{policy_content}\n\nDescription:"
    )
    resp = claude_client.messages.create(
        model=CLAUDE_MODEL_NAME,
        max_tokens=1024,
        system="You analyze AWS IAM policies and describe them.",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text.strip()

@retry(max_attempts=4, initial_delay=5)
def generate_new_policy(description: str) -> str:
    """
    Step 2: Reconstruct an equivalent policy JSON from the description.
    """
    system_prompt = (
        "Reconstruct an AWS IAM policy in JSON format based on the description. "
        "Return only valid JSON without any extra text."
    )
    resp = claude_client.messages.create(
        model=CLAUDE_MODEL_NAME,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": description}]
    )
    text = resp.content[0].text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Invalid policy JSON response: {text}")
    return text[start : end + 1]

def sample_and_generate_regexes(orig_path: Path, new_path: Path, bound: int, num: int, m1: int, m2: int) -> tuple[Path, Path]:
    """
    1. Run Quacky to produce P1_not_P2.models and not_P1_P2.models (under WORKING_DIR).
    2. Read those model files and ask Claude Sonnet 4 to produce one regex each.
    3. Save the resulting regexes under WORKING_DIR and return their Paths.
    """
    args = [
        "-b", str(bound),
        "-p1", str(orig_path.resolve()),
        "-p2", str(new_path.resolve()),
        "-m", str(num),
        "-m1", str(m1),
        "-m2", str(m2)
    ]
    run_quacky(args)

    models_a = MODELS_A_PATH.read_text().strip()
    models_b = MODELS_B_PATH.read_text().strip()

    logging.info("Synthesizing regex A (P1_not_P2)…")
    rx_a = llm_generate_regex(claude_client, CLAUDE_MODEL_NAME, models_a)
    RESPONSE_A_PATH.write_text(rx_a)

    logging.info("Synthesizing regex B (not_P1_P2)…")
    rx_b = llm_generate_regex(claude_client, CLAUDE_MODEL_NAME, models_b)
    RESPONSE_B_PATH.write_text(rx_b)

    return RESPONSE_A_PATH.resolve(), RESPONSE_B_PATH.resolve()


def process_policy(path: Path, size: int) -> dict:
    """
    For one policy file:
      1. Describe it (get_policy_description).
      2. Reconstruct JSON (generate_new_policy).
      3. Run Quacky to compare original vs. reconstructed (generate_strings).
      4. Generate two regexes based on Quacky’s P1_not_P2 and not_P1_P2 models.
      5. Run final Quacky to compute Jaccard similarities with those regexes.
      6. Return a record dict for CSV.
    """
    start_time = time.time()
    orig = read_policy_file(path)

    desc = get_policy_description(orig)

    new_pol = generate_new_policy(desc)
    new_path = WORKING_DIR / "generated.json"
    save_generated_policy(new_pol, new_path)

    rx_a, rx_b = sample_and_generate_regexes(path, new_path, bound=100, num=size, m1=20, m2=100)
    logging.info(f"Model files at: {MODELS_A_PATH}, {MODELS_B_PATH}")
    logging.info(f"Regex files at: {rx_a}, {rx_b}")

    logging.info("Running final comparison…")
    final = run_quacky([
        "-p1", str(path.resolve()),
        "-p2", str(new_path.resolve()),
        "-b", "100",
        "-cr", str(rx_a),
        "-cr2", str(rx_b)
    ])
    logging.info(f"Final analysis output:\n{final}")

    # Parse Jaccard similarities
    parts = re.split(r"Policy\s*2\s*⇏\s*Policy\s*1", final)
    block1 = parts[0]
    block2 = parts[1] if len(parts) > 1 else ""

    def parse_similarity(block: str) -> float | None:
        num_m = re.search(r"jaccard(?:\s+index)?\s*numerator\s*[:=]\s*([0-9]+)", block, re.IGNORECASE)
        den_m = re.search(r"jaccard(?:\s+index)?\s*denominator\s*[:=]\s*([0-9]+)", block, re.IGNORECASE)
        if num_m and den_m:
            return int(num_m.group(1)) / int(den_m.group(1))
        logging.warning("Jaccard values not found in block")
        return None

    sim_a = parse_similarity(block1)
    sim_b = parse_similarity(block2)

    total_time = time.time() - start_time
    return {
        "Policy Number": int(path.stem),
        "Original Policy": orig,
        "Generated Policy": new_pol,
        "Policy Description": desc,
        "Size": size,
        "Regex A File": str(rx_a),
        "Regex B File": str(rx_b),
        "Similarity A": sim_a,
        "Similarity B": sim_b,
        "Final Analysis": final,
        "Total Time (s)": total_time
    }

def get_progress() -> dict:
    """Load last processed index from PROGRESS_PATH."""
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {"last": 0}

def update_progress(val: int):
    """Write last processed index to PROGRESS_PATH."""
    PROGRESS_PATH.write_text(json.dumps({"last": val}))

# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate regexes for AWS IAM policies using Quacky and Claude Sonnet 4.")
    parser.add_argument("--size", type=int, required=True, help="Size of strings for Quacky to sample when synthesizing regexes.")
    args = parser.parse_args()
    size = args.size

    files = sorted(POLICY_FOLDER.glob("*.json"), key=lambda p: int(p.stem))
    prog = get_progress()
    start_idx = prog.get("last", 0)

    columns = [
        "Policy Number", "Original Policy", "Generated Policy",
        "Policy Description", "Size", "Regex A File", "Regex B File",
        "Similarity A", "Similarity B", "Final Analysis", "Total Time (s)"
    ]
    if not RESULT_TABLE_PATH.exists() or RESULT_TABLE_PATH.stat().st_size == 0:
        pd.DataFrame(columns=columns).to_csv(RESULT_TABLE_PATH, index=False)

    for idx, f in enumerate(files[start_idx:], start=start_idx):
        logging.info(f"Processing {f.name} ({idx + 1}/{len(files)})")
        try:
            record = process_policy(f, size)
            pd.DataFrame([record]).to_csv(
                RESULT_TABLE_PATH,
                mode="a",
                header=False,
                index=False
            )
            update_progress(idx + 1)
        except Exception as e:
            logging.error(f"Error on {f.name}: {e}")
            continue

    logging.info("All done.")
