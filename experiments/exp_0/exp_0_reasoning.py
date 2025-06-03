


"""
exp_0_reasoning.py

This script processes AWS IAM policy JSON files by:
  1. Reading each policy file.
  2. Using one of five LLMs (Gemini, Grok, DeepSeek-R, GPT, Claude) to:
       a. Generate a comprehensive natural-language description (original prompt unchanged).
       b. Reconstruct the policy JSON from that description (original prompt unchanged).
     Each call includes the reasoning‐enabling arguments specific to that API.
  3. Running Quacky to compare the original vs. generated policies.
  4. Appending results to a CSV (result_table_path) and tracking progress (progress_file_path).

Usage:
  python3 exp_0_reasoning.py --model <gemini|grok|deepseek|gpt|claude>
"""


import os
import sys
import time
import json
import logging
import subprocess
import re
import argparse
from pathlib import Path
from functools import wraps

import pandas as pd
from tqdm import tqdm
from csv import QUOTE_NONE

# ─── Global configurations ──────────────────────────────────────────────────
MAX_ATTEMPT = 3
DELAY       = 5

# ─── Configuration ───────────────────────────────────────────────────────────
def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", text).strip("_")



RESULT_DIR         = Path("Exp-1")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
# result_table_path and progress_file_path will be defined after parsing model
result_table_path  = None
progress_file_path = None

policy_folder         = None # Directory with .json policy files
quacky_path           = None # Path to Quacky script
working_directory     = None # cwd when invoking Quacky
generated_policy_path = None # Path to save generated policy JSON

# Gemini
gemini_client       = None  # replace with your Gemini client initialization
gemini_model_name   = "gemini-2.5-flash"

# Grok
grok_client         = None  # replace with your Grok client initialization
grok_model_name     = "grok-3-latest"

# DeepSeek-R
deepseek_client     = None  # replace with your DeepSeek client initialization
deepseek_model_name = "deepseek-reasoner"

# OpenAI GPT
gpt_client          = None  # replace with your OpenAI client initialization
gpt_model_name      = "gpt-o4-mini"

# Anthropic Claude
claude_client       = None  # replace with your Anthropic client initialization
claude_model_name   = "claude-3-7-sonnet-20250219"


# ─── Utilities ───────────────────────────────────────────────────────────────

def read_policy_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

def retry(max_attempts=MAX_ATTEMPT, delay=DELAY):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    logging.warning(f"Attempt {attempts} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator


# Model‐Specific API Calls

# 1) Gemini
@retry()
def get_gemini_policy_description(policy_content: str) -> str:
    try:
        policy_content = policy_content.encode('utf-8').decode('utf-8')
    except UnicodeError:
        policy_content = policy_content.encode('ascii', 'ignore').decode('ascii')
    prompt = (
        "Generate a comprehensive and unambiguous natural language description of the provided AWS IAM policy. "
        "The description must meticulously detail every component of the policy so that another language model can "
        "perfectly reconstruct the original policy, at least semantically.\n\n"
        f"{policy_content}\n\nDescription:"
    )
    model = gemini_client.GenerativeModel(gemini_model_name)
    response = model.generate_content(
        prompt,
        reasoning_effort="high"
    )
    if hasattr(response, 'text') and response.text:
        return response.text.strip()
    if hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text'):
        return response.parts[0].text.strip()
    if hasattr(response, 'candidates') and response.candidates:
        cand = response.candidates[0]
        if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
            return cand.content.parts[0].text.strip()
    raise ValueError(f"No valid text in Gemini response: {response}")

@retry()
def generate_gemini_new_policy(description: str) -> str:
    try:
        description = description.encode('utf-8').decode('utf-8')
    except UnicodeError:
        description = description.encode('ascii', 'ignore').decode('ascii')
    system_prompt = """
    Reconstruct an AWS IAM policy in JSON format based on the provided natural language description. The generated policy must semantically match the description, including all specified effects, actions, resources, principals (if any), and conditions.
    Output Format:
        Provide only the JSON policy. Do not include any explanatory text before or after the JSON block.
        Example response format:
        {
        "Version": "2012-10-17",
        "Statement": [
            {
            "Effect": "Allow",
            "Action": ["s3:GetObject"],
            "Resource": "arn:aws:s3:::example-bucket/*"
            }
        ]
        }
    Ensure the generated policy matches the description and follows AWS IAM structure.
    """.strip()
    user_prompt = f"{system_prompt}\n\nGenerate the IAM policy JSON that satisfies: {description}"
    model = gemini_client.GenerativeModel(gemini_model_name)
    response = model.generate_content(
        user_prompt,
        reasoning_effort="high"
    )
    text = ""
    if hasattr(response, 'text') and response.text:
        text = response.text
    elif hasattr(response, 'parts') and response.parts:
        text = response.parts[0].text
    elif hasattr(response, 'candidates') and response.candidates:
        cand = response.candidates[0]
        if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
            text = cand.content.parts[0].text
    if not text:
        raise ValueError(f"No text in Gemini response: {response}")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in Gemini response: {text}")
    json_part = text[start:end+1].strip()
    json.loads(json_part)
    return json_part

# 2) Grok
@retry()
def get_grok_policy_description(policy_content: str) -> str:
    try:
        policy_content = policy_content.encode('utf-8').decode('utf-8')
    except UnicodeError:
        policy_content = policy_content.encode('ascii', 'ignore').decode('ascii')
    prompt = (
        "Generate a comprehensive and unambiguous natural language description of the provided AWS IAM policy. "
        "The description must meticulously detail every component of the policy so that another language model can "
        "perfectly reconstruct the original policy, at least semantically.\n\n"
        f"{policy_content}\n\nDescription:"
    )
    response = grok_client.chat.completions.create(
        model=grok_model_name,
        reasoning_effort="high",
        messages=[
            {"role": "system", "content": "You analyze AWS IAM policies and provide concise descriptions."},
            {"role": "user",   "content": prompt}
        ],
    )
    if response and response.choices and len(response.choices) > 0:
        return response.choices[0].message.content.strip()
    raise ValueError(f"No valid text in Grok response: {response}")

@retry()
def generate_grok_new_policy(description: str) -> str:
    try:
        description = description.encode('utf-8').decode('utf-8')
    except UnicodeError:
        description = description.encode('ascii', 'ignore').decode('ascii')
    system_prompt = """
    Reconstruct an AWS IAM policy in JSON format based on the provided natural language description. The generated policy must semantically match the description, including all specified effects, actions, resources, principals (if any), and conditions.
    Output Format:
        Provide only the JSON policy. Do not include any explanatory text before or after the JSON block.
        Example response format:
        {
        "Version": "2012-10-17",
        "Statement": [
            {
            "Effect": "Allow",
            "Action": ["s3:GetObject"],
            "Resource": "arn:aws:s3:::example-bucket/*"
            }
        ]
        }
    Ensure the generated policy matches the description and follows AWS IAM structure.
    """.strip()
    response = grok_client.chat.completions.create(
        model=grok_model_name,
        reasoning_effort="high",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Generate the IAM policy JSON that satisfies: {description}"}
        ],
        temperature=0.2
    )
    if not (response and response.choices and len(response.choices) > 0):
        raise ValueError(f"No valid JSON in Grok response: {response}")
    text = response.choices[0].message.content
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in Grok response: {text}")
    json_part = text[start:end+1].strip()
    json.loads(json_part)
    return json_part

# 3) DeepSeek-Reasoner
@retry()
def get_deepseek_policy_description(policy_content: str) -> str:
    try:
        policy_content = policy_content.encode('utf-8').decode('utf-8')
    except UnicodeError:
        policy_content = policy_content.encode('ascii', 'ignore').decode('ascii')
    prompt = (
        "Generate a comprehensive and unambiguous natural language description of the provided AWS IAM policy. "
        "The description must meticulously detail every component of the policy so that another language model can "
        "perfectly reconstruct the original policy, at least semantically.\n\n"
        f"{policy_content}\n\nDescription:"
    )
    response = deepseek_client.chat.completions.create(
        model=deepseek_model_name,
        messages=[
            {"role": "system", "content": "You analyze AWS IAM policies and provide concise descriptions."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7
    )
    if response and hasattr(response, 'choices') and response.choices:
        return response.choices[0].message.content.strip()
    raise ValueError(f"No valid text in DeepSeek response: {response}")

@retry()
def generate_deepseek_new_policy(description: str) -> str:
    try:
        description = description.encode('utf-8').decode('utf-8')
    except UnicodeError:
        description = description.encode('ascii', 'ignore').decode('ascii')
    system_prompt = """
    Reconstruct an AWS IAM policy in JSON format based on the provided natural language description. The generated policy must semantically match the description, including all specified effects, actions, resources, principals (if any), and conditions.
    Output Format:
        Provide only the JSON policy. Do not include any explanatory text before or after the JSON block.
        Example response format:
        {
        "Version": "2012-10-17",
        "Statement": [
            {
            "Effect": "Allow",
            "Action": ["s3:GetObject"],
            "Resource": "arn:aws:s3:::example-bucket/*"
            }
        ]
        }
    Ensure the generated policy matches the description and follows AWS IAM structure.
    """.strip()
    response = deepseek_client.chat.completions.create(
        model=deepseek_model_name,
        messages=[
            {
                "role": "system", 
                "content": system_prompt
            },
            
            {
                "role": "user",   
                "content": f"Generate the IAM policy JSON that satisfies: {description}"
                }
        ],

        temperature=0.7
    )
    if not (response and hasattr(response, 'choices') and response.choices):
        raise ValueError(f"No valid JSON in DeepSeek response: {response}")
    text = response.choices[0].message.content
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON block in DeepSeek response: {text}")
    json_part = text[start:end+1].strip()
    json.loads(json_part)
    return json_part

# 4) OpenAI GPT
@retry()
def get_gpt_policy_description(policy_content: str) -> str:
    try:
        policy_content = policy_content.encode('utf-8').decode('utf-8')
    except UnicodeError:
        policy_content = policy_content.encode('ascii', 'ignore').decode('ascii')
    prompt = (
        "Generate a comprehensive and unambiguous natural language description of the provided AWS IAM policy. "
        "The description must meticulously detail every component of the policy so that another language model can "
        "perfectly reconstruct the original policy, at least semantically.\n\n"
        f"{policy_content}\n\nDescription:"
    )
    response = gpt_client.responses.create(
        model=gpt_model_name,
        reasoning = {"effort": "high"},
        
        input=
        [
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens = 4000,
        temperature=0.7
    )
    if response and response.output_text and len(response.output_text) > 0:
        return response.output_text.strip()
    raise ValueError(f"No valid text in GPT response: {response}")

@retry()
def generate_gpt_new_policy(description: str) -> str:
    try:
        description = description.encode('utf-8').decode('utf-8')
    except UnicodeError:
        description = description.encode('ascii', 'ignore').decode('ascii')
    system_prompt = """
    Reconstruct an AWS IAM policy in JSON format based on the provided natural language description. The generated policy must semantically match the description, including all specified effects, actions, resources, principals (if any), and conditions.
    Output Format:
        Provide only the JSON policy. Do not include any explanatory text before or after the JSON block.
        Example response format:
        {
        "Version": "2012-10-17",
        "Statement": [
            {
            "Effect": "Allow",
            "Action": ["s3:GetObject"],
            "Resource": "arn:aws:s3:::example-bucket/*"
            }
        ]
        }
    Ensure the generated policy matches the description and follows AWS IAM structure.
    """.strip()
    response = gpt_client.responses.create(
        model=gpt_model_name,
        reasoning= {"effort": "high"},
        input=[
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Generate the IAM policy JSON that satisfies: {description}"}
        ],
        max_tokens=9000,
        temperature=0.7
    )
    if not (response and response.output_text and len(response.output_text) > 0):
        raise ValueError(f"No valid JSON in GPT response: {response}")
    text = response.output_text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in GPT response: {text}")
    json_part = text[start:end+1].strip()
    json.loads(json_part)
    return json_part

# 5) Anthropic Claude
@retry()
def get_claude_policy_description(policy_content: str) -> str:
    try:
        policy_content = policy_content.encode('utf-8').decode('utf-8')
    except UnicodeError:
        policy_content = policy_content.encode('ascii', 'ignore').decode('ascii')

    prompt = (
        "Generate a comprehensive and unambiguous natural language description of the provided AWS IAM policy. "
        "The description must meticulously detail every component of the policy so that another language model can "
        "perfectly reconstruct the original policy, at least semantically.\n\n"
        f"{policy_content}\n\nDescription:"
    )

    response = claude_client.messages.create(
        model=claude_model_name,
        max_tokens=9000,
        thinking={"type": "enabled", "budget_tokens": 4000},
        messages=[
            {"role": "system", "content": "You analyze AWS IAM policies and provide concise descriptions."},
            {"role": "user",   "content": prompt}
        ]
    )

    if response and response.content:
        for content_block in response.content:
            if content_block.type == 'text':
                return content_block.text.strip()

    raise ValueError(f"No valid text in Claude response: {response}")


@retry()
def generate_claude_new_policy(description: str) -> str:
    try:
        description = description.encode('utf-8').decode('utf-8')
    except UnicodeError:
        description = description.encode('ascii', 'ignore').decode('ascii')

    system_prompt = """
    Reconstruct an AWS IAM policy in JSON format based on the provided natural language description. The generated policy must semantically match the description, including all specified effects, actions, resources, principals (if any), and conditions.
    Output Format:
        Provide only the JSON policy. Do not include any explanatory text before or after the JSON block.
        Example response format:
        {
        "Version": "2012-10-17",
        "Statement": [
            {
            "Effect": "Allow",
            "Action": ["s3:GetObject"],
            "Resource": "arn:aws:s3:::example-bucket/*"
            }
        ]
        }
    Ensure the generated policy matches the description and follows AWS IAM structure.
    """.strip()

    response = claude_client.messages.create(
        model=claude_model_name,
        max_tokens=9000,
        thinking={"type": "enabled", "budget_tokens": 5000},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Generate the IAM policy JSON that satisfies: {description}"}
        ]
    )

    text = ""
    if response and response.content:
        for content_block in response.content:
            if content_block.type == 'text':
                text = content_block.text
                break

    if not text:
        raise ValueError(f"No valid text in Claude response: {response}")

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in Claude response: {text}")

    json_part = text[start:end+1].strip()
    json.loads(json_part)
    return json_part


#Quacky & Progress Helpers

def save_generated_policy(policy_content: str, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    json_obj = json.loads(policy_content)
    with open(file_path, "w") as f:
        json.dump(json_obj, f, indent=2)
    logging.info(f"Generated policy saved to {file_path}")

def generate_strings(original_policy_path: str, generated_policy_path: str, size: int) -> bool:
    command = [
        "python3", os.path.abspath(quacky_path),
        "-p1", os.path.abspath(original_policy_path),
        "-p2", os.path.abspath(generated_policy_path),
        "-b", str(size)
    ]
    result = subprocess.run(command, cwd=os.path.abspath(working_directory),
                             capture_output=True, text=True)
    if result.stderr:
        logging.error(f"Quacky strings stderr: {result.stderr}")
    return result.returncode == 0

@retry()
def run_final_analysis(original_policy_path: str, generated_policy_path: str) -> str:
    command = [
        "python3", os.path.abspath(quacky_path),
        "-p1", os.path.abspath(original_policy_path),
        "-p2", os.path.abspath(generated_policy_path),
        "-b", "100"
    ]
    result = subprocess.run(command, cwd=os.path.abspath(working_directory),
                             capture_output=True, text=True)
    logging.info("Quacky Final Analysis Output:")
    logging.info(result.stdout)
    if result.stderr:
        logging.error(f"Quacky final stderr: {result.stderr}")
    return result.stdout

def get_progress() -> dict:
    if os.path.exists(progress_file_path):
        with open(progress_file_path, 'r') as f:
            return json.load(f)
    return {"last_processed": 0}

def update_progress(last_processed: int):
    with open(progress_file_path, 'w') as f:
        json.dump({"last_processed": last_processed}, f)



@retry(max_attempts=MAX_ATTEMPT, delay=10)
def process_policy(policy_path: str, size: int) -> dict:
    start_time = time.time()
    original_policy = read_policy_file(policy_path)

    if POLICY_MODEL == "gemini":
        policy_description = get_gemini_policy_description(original_policy)
        new_policy        = generate_gemini_new_policy(policy_description)

    elif POLICY_MODEL == "grok":
        policy_description = get_grok_policy_description(original_policy)
        new_policy        = generate_grok_new_policy(policy_description)

    elif POLICY_MODEL == "deepseek":
        policy_description = get_deepseek_policy_description(original_policy)
        new_policy        = generate_deepseek_new_policy(policy_description)

    elif POLICY_MODEL == "gpt":
        policy_description = get_gpt_policy_description(original_policy)
        new_policy        = generate_gpt_new_policy(policy_description)

    elif POLICY_MODEL == "claude":
        policy_description = get_claude_policy_description(original_policy)
        new_policy        = generate_claude_new_policy(policy_description)

    else:
        raise ValueError(f"Unknown POLICY_MODEL: {POLICY_MODEL}")

    logging.info("Generated Policy JSON:")
    logging.info(new_policy)
    save_generated_policy(new_policy, generated_policy_path)

    if not generate_strings(policy_path, generated_policy_path, size):
        raise Exception("Failed to generate strings with Quacky.")

    final_analysis = run_final_analysis(policy_path, generated_policy_path)
    total_processing_time = time.time() - start_time #time it takes to run the entire pipleline in seconds

    return {
        "model_name": POLICY_MODEL,
        "Original Policy": original_policy,
        "Generated Policy": new_policy,
        "Policy Description": policy_description,
        "Size": size,
        "Final Analysis": final_analysis,
        "Total Processing Time (seconds)": total_processing_time
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AWS IAM policies with various LLMs.")
    parser.add_argument(
        "--model",
        choices=["gemini", "grok", "deepseek", "gpt", "claude"],
        required=True,
        help="Which LLM to use for processing policies."
    )
    args = parser.parse_args()
    POLICY_MODEL = args.model
    MODEL_SLUG    = slugify(POLICY_MODEL)
    result_table_path  = RESULT_DIR / f"{MODEL_SLUG}.csv"
    progress_file_path = RESULT_DIR / f"{MODEL_SLUG}.json"

    size = 500
    policy_files = sorted(
        [f for f in os.listdir(policy_folder) if f.endswith('.json')],
        key=lambda x: int(Path(x).stem)
    )
    total = len(policy_files)
    print(f"Found {total} policy files.")

    progress = get_progress()
    last_stem = progress.get("last_processed", 0)
    start_index = 0
    if last_stem:
        try:
            start_index = next(
                i for i, fname in enumerate(policy_files)
                if int(Path(fname).stem) == last_stem
            )
        except StopIteration:
            print(f"Warning: {last_stem}.json not found; starting at 0")

    if start_index >= total:
        print(f"All {total} policies processed. Nothing to do.")
        sys.exit(0)

    print(f"Resuming from policy number {policy_files[start_index]} (index {start_index})")

    required_columns = [
        "Policy Number", "model_name", "Original Policy", "Generated Policy",
        "Policy Description", "Size", "Final Analysis", "Total Processing Time (seconds)"
    ]
    if not os.path.exists(result_table_path) or os.stat(result_table_path).st_size == 0:
        pd.DataFrame(columns=required_columns).to_csv(result_table_path, index=False)
    else:
        df_existing = pd.read_csv(
            result_table_path, quoting=QUOTE_NONE, sep=",", low_memory=False
        )
        
        for col in set(required_columns) - set(df_existing.columns):
            df_existing[col] = ""
        df_existing.to_csv(result_table_path, index=False)
        

    processed = 0
    for i in tqdm(range(start_index, total), desc="Processing policies"):
        fname = policy_files[i]
        policy_path = os.path.join(policy_folder, fname)
        stem = int(Path(fname).stem)
        logging.info(f"\nProcessing policy: {fname}")

        try:
            new_entry = process_policy(policy_path, size)
            new_entry["Policy Number"] = stem
            pd.DataFrame([new_entry]).to_csv(
                result_table_path,
                mode='a',
                header=False,
                index=False
            )
            update_progress(stem)
            processed += 1

        except Exception as e:
            logging.error(f"Failed to process {fname}: {e}")
            continue

    logging.info("\nProcessing complete. Final results table saved to: " + str(result_table_path))
    next_policy = (
        policy_files[start_index + processed]
        if processed + start_index < total else "done"
    )
    print(f"Processed {processed} policies. Next run will start from policy number {next_policy}")
