#!/usr/bin/env python3
"""
Report Generator for Gemini Models API Benchmarking Tool
Uses Gemini to analyze benchmark results from reports_ai_studio and reports_vertex_ai
and write a unified Markdown report comparing both developer surfaces.
"""

import os
import sys
import json
import glob
import re
import time
from typing import Dict, Any, Tuple, Optional, List

# Load environment variables from .env if available
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

# Map API_KEY to GEMINI_API_KEY if needed
if "API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["API_KEY"]

# Import prompt library
import prompts

# Import the unified Google Gen AI SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: The 'google-genai' SDK is not installed in the current environment.", file=sys.stderr)
    print("Please install it using: pip install google-genai", file=sys.stderr)
    sys.exit(1)


def initialize_client(location: Optional[str] = None) -> genai.Client:
    """Initializes the GenAI Client using AI Studio or Vertex AI depending on available env vars."""
    api_key = os.environ.get("API_KEY")
    project = os.environ.get("PROKECT_ID") 
    if location is None:
        location = "global"
    
    if api_key:
        print("[Report Gen] Initializing client using AI Studio (GEMINI_API_KEY)...")
        return genai.Client(api_key=api_key)
    elif project:
        print(f"[Report Gen] Initializing client using Vertex AI (Project: {project}, Location: {location})...")
        return genai.Client(vertexai=True, project=project, location=location)
    else:
        print("[Report Gen] Warning: Neither GEMINI_API_KEY nor GOOGLE_CLOUD_PROJECT are set in environment.")
        print("[Report Gen] Attempting default Client initialization (might fail if no credentials)...")
        return genai.Client()


def generate_content_with_retry(
    client: genai.Client,
    model: str,
    contents: Any,
    config: Optional[types.GenerateContentConfig] = None,
    max_retries: int = 5,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0
) -> Any:
    """Calls generate_content with exponential backoff retry on transient errors (503, 429, 500)."""
    import random
    
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response
        except Exception as e:
            err_str = str(e).lower()
            status_code = 500
            if "429" in err_str or "rate limit" in err_str or "resource_exhausted" in err_str:
                status_code = 429
            elif "503" in err_str or "unavailable" in err_str:
                status_code = 503
            elif "500" in err_str or "internal" in err_str:
                status_code = 500
                
            if status_code in [429, 500, 503] and attempt < max_retries:
                delay = initial_delay * (backoff_factor ** attempt) + random.uniform(0, 1)
                print(f"[Retry] API call failed (status {status_code}). Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                raise e


def extract_response_text(response: Any) -> str:
    """Extracts text content from response candidates/parts when code execution tool is used."""
    text_chunks = []
    if hasattr(response, "candidates") and response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text_chunks.append(part.text)
    if text_chunks:
        return "\n\n".join(text_chunks).strip()
    return getattr(response, "text", "")


def load_summary_metrics(file_path: str) -> Dict[str, Any]:
    """Loads a summary lifecycle metrics JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_name_and_metadata(config_path: str, file_prefix: str) -> Tuple[str, bool, bool]:
    """
    Extracts model name, streaming setting, and multimodal presence from a yaml config.
    Returns (model_name, is_streaming, is_multimodal).
    """
    is_streaming = True
    is_multimodal = False
    
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            
            if cfg:
                if "api" in cfg and "streaming" in cfg["api"]:
                    is_streaming = cfg["api"]["streaming"]
                if "data" in cfg and "multimodal" in cfg["data"]:
                    is_multimodal = cfg["data"]["multimodal"] is not None
        except Exception as e:
            print(f"[Warning] Failed to parse config {config_path}: {e}")
            
    # Always derive model_name from filename prefix to guarantee exact matching
    clean_pref = file_prefix
    if clean_pref.startswith("ai_studio_"):
        clean_pref = clean_pref[len("ai_studio_"):]
    elif clean_pref.startswith("vertex_ai_"):
        clean_pref = clean_pref[len("vertex_ai_"):]
        
    if clean_pref.endswith("_"):
        clean_pref = clean_pref[:-1]
        
    model_name = clean_pref.replace("_", "-")
    if "multimodal" in model_name or "image" in model_name:
        is_multimodal = True
            
    return model_name, is_streaming, is_multimodal


def parse_summary_metrics(summary_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts performance metrics from a summary lifecycle metrics structure."""
    successes = summary_data.get("successes") or {}
    failures = summary_data.get("failures") or {}
    load_summary = summary_data.get("load_summary") or {}
    
    latency = successes.get("latency") or {}
    throughput = successes.get("throughput") or {}
    
    ttft = latency.get("time_to_first_token") or {}
    ttlt = latency.get("request_latency") or {}
    tpot = latency.get("time_per_output_token") or {}
    itl = latency.get("inter_token_latency") or {}
    ntpot = latency.get("normalized_time_per_output_token") or {}
    
    total_count = load_summary.get("count") or 0
    success_count = successes.get("count") or 0
    failure_count = failures.get("count") or 0
    
    if total_count == 0:
        total_count = success_count + failure_count
        
    success_rate = (success_count / total_count * 100.0) if total_count > 0 else 0.0
    
    request_size_bytes = successes.get("request_size_bytes") or {}
    prompt_len = successes.get("prompt_len") or {}
    output_len = successes.get("output_len") or {}
    prompt_tokens = successes.get("prompt_tokens") or {}
    output_tokens = successes.get("output_tokens") or {}
    
    image_stats = successes.get("image") or {}
    video_stats = successes.get("video") or {}
    audio_stats = successes.get("audio") or {}
    
    def get_mean(obj):
        if isinstance(obj, dict):
            return obj.get("mean")
        return None
    
    return {
        "benchmark_time_seconds": summary_data.get("benchmark_time_seconds"),
        "success_rate": success_rate,
        "success_count": success_count,
        "failure_count": failure_count,
        "total_requests": total_count,
        "ttft": {
            "mean": ttft.get("mean"),
            "median": ttft.get("median"),
            "p90": ttft.get("p90"),
            "p99": ttft.get("p99")
        },
        "ttlt": {
            "mean": ttlt.get("mean"),
            "median": ttlt.get("median"),
            "p90": ttlt.get("p90"),
            "p99": ttlt.get("p99")
        },
        "tpot": {
            "mean": tpot.get("mean"),
            "median": tpot.get("median"),
            "p90": tpot.get("p90"),
            "p99": tpot.get("p99")
        },
        "itl": {
            "mean": itl.get("mean"),
            "median": itl.get("median"),
            "p90": itl.get("p90"),
            "p99": itl.get("p99")
        },
        "ntpot": {
            "mean": ntpot.get("mean"),
            "median": ntpot.get("median"),
            "p90": ntpot.get("p90"),
            "p99": ntpot.get("p99")
        },
        "throughput": {
            "output_tokens_per_sec": throughput.get("output_tokens_per_sec"),
            "input_tokens_per_sec": throughput.get("input_tokens_per_sec"),
            "total_tokens_per_sec": throughput.get("total_tokens_per_sec"),
            "requests_per_sec": throughput.get("requests_per_sec"),
            "images_per_sec": throughput.get("images_per_sec", 0.0),
            "videos_per_sec": throughput.get("videos_per_sec", 0.0),
            "audios_per_sec": throughput.get("audios_per_sec", 0.0)
        },
        "request_size_bytes": {
            "mean": request_size_bytes.get("mean"),
            "median": request_size_bytes.get("median"),
            "p90": request_size_bytes.get("p90"),
            "p99": request_size_bytes.get("p99")
        },
        "prompt_len": {
            "mean": prompt_len.get("mean"),
            "median": prompt_len.get("median"),
            "p90": prompt_len.get("p90"),
            "p99": prompt_len.get("p99")
        },
        "output_len": {
            "mean": output_len.get("mean"),
            "median": output_len.get("median"),
            "p90": output_len.get("p90"),
            "p99": output_len.get("p99")
        },
        "prompt_tokens": {
            "total": prompt_tokens.get("total"),
            "cached": prompt_tokens.get("cached"),
            "uncached": prompt_tokens.get("uncached")
        },
        "output_tokens": {
            "total": output_tokens.get("total")
        },
        "modalities": {
            "image_count_mean": get_mean(image_stats.get("count")),
            "video_count_mean": get_mean(video_stats.get("count")),
            "audio_count_mean": get_mean(audio_stats.get("count"))
        }
    }


def generate_comparison_report(
    ai_studio_dir: str,
    vertex_ai_dir: str,
    output_dir: str,
    model_name: str = "gemini-3.5-flash",
    location: str = "global",
    overlap_only: bool = False
) -> None:
    """
    Collects summary metrics from reports_ai_studio and reports_vertex_ai,
    aggregates them, and asks Gemini to output a unified comparison report.
    """
    print("\n" + "=" * 70)
    print("                 COLLECTING BENCHMARK SUMMARY METRICS")
    print("=" * 70)
    
    # Auto-detect alternate directories if the specified directories do not exist
    if not os.path.exists(ai_studio_dir):
        for alt_path in ["./reports_ai_studio", "reports_ai_studio", "./reports_aistudio", "reports_aistudio"]:
            if os.path.exists(alt_path):
                print(f"[Collector] Directory '{ai_studio_dir}' not found. Using detected alternative: '{alt_path}'")
                ai_studio_dir = alt_path
                break

    if not os.path.exists(vertex_ai_dir):
        for alt_path in ["./reports_vertex", "./reports_vertex_ai", "reports_vertex", "reports_vertex_ai"]:
            if os.path.exists(alt_path):
                print(f"[Collector] Directory '{vertex_ai_dir}' not found. Using detected alternative: '{alt_path}'")
                vertex_ai_dir = alt_path
                break

    all_data = {}
    
    # Define directories and surfaces
    scan_configs = [
        (ai_studio_dir, "ai_studio"),
        (vertex_ai_dir, "vertex_ai")
    ]
    
    for dir_path, surface in scan_configs:
        if not os.path.exists(dir_path):
            print(f"[Info] Directory {dir_path} does not exist. Skipping.")
            continue
            
        summary_pattern = os.path.join(dir_path, "*_summary_lifecycle_metrics.json")
        summary_files = glob.glob(summary_pattern)
        
        # Filter out generic summary_lifecycle_metrics.json if we have specific prefix files
        specific_files = [f for f in summary_files if os.path.basename(f) != "summary_lifecycle_metrics.json"]
        if not specific_files and summary_files:
            specific_files = summary_files
            
        print(f"[Collector] Found {len(specific_files)} summary file(s) in '{dir_path}' for surface '{surface}':")
        for file_path in specific_files:
            filename = os.path.basename(file_path)
            print(f"  - {filename}")
            
            # Determine prefix
            prefix = filename.replace("summary_lifecycle_metrics.json", "")
            config_path = os.path.join(dir_path, f"{prefix}config.yaml")
            
            # Extract metadata
            m_name, is_streaming, is_multimodal = get_model_name_and_metadata(config_path, prefix)
            
            try:
                summary_json = load_summary_metrics(file_path)
                metrics = parse_summary_metrics(summary_json)
                metrics["is_streaming"] = is_streaming
                metrics["is_multimodal"] = is_multimodal
                
                if m_name not in all_data:
                    all_data[m_name] = {}
                
                all_data[m_name][surface] = metrics
            except Exception as e:
                print(f"[Error] Failed to parse metrics from {file_path}: {e}")
                
    if overlap_only:
        all_data = {
            m_name: surfaces 
            for m_name, surfaces in all_data.items() 
            if "ai_studio" in surfaces and "vertex_ai" in surfaces
        }
        print(f"\n[Collector] Filtering for overlap only. Models with both surfaces: {len(all_data)}")

    if not all_data:
        print("[Error] No benchmark summary metrics files found in either directory matching criteria.", file=sys.stderr)
        sys.exit(1)
        
    print("\n[Collector] Aggregated Model Data Summary:")
    for m_name, surfaces in all_data.items():
        print(f"  - {m_name}: Surfaces: {list(surfaces.keys())}")
        
    # Initialize client and generate report
    client = initialize_client(location=location)
    prompt = prompts.get_comparison_report_prompt(all_data, {})
    
    # Configure Code Execution tool for exact math calculations
    code_exec_config = types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())],
        max_output_tokens=65536
    )
    
    print(f"\n[Report Gen] Requesting comparison report from {model_name} (Location: {location}, Code Execution enabled)...")
    try:
        response = generate_content_with_retry(client, model_name, prompt, config=code_exec_config)
        report_content = extract_response_text(response)
        
        # Save output report
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        report_filename = os.path.join(output_dir, "benchmark_report.md")
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        print(f"[Report Gen] Saved comparison report to: {report_filename}")
        print("\n" + "=" * 70)
        print("                 BENCHMARK REPORT COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}", file=sys.stderr)
        sys.exit(1)


# Legacy report functions preserved for backward compatibility
def generate_report(
    summary_json_path: str,
    raw_csv_path: Optional[str],
    output_dir: str,
    timestamp_str: str,
    model_name: str = "gemini-3.5-flash",
    location: str = "global"
) -> None:
    # Read the JSON summary
    with open(summary_json_path, "r") as f:
        summary_data = json.load(f)
        
    error_summary = {}
    client = initialize_client(location=location)
    is_master = timestamp_str == "master"
    prompt = prompts.get_report_prompt(summary_data, error_summary, is_master=is_master)
    
    code_exec_config = types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())],
        max_output_tokens=65536
    )
    
    print(f"[Report Gen] Requesting legacy report from {model_name}...")
    try:
        response = generate_content_with_retry(client, model_name, prompt, config=code_exec_config)
        report_content = extract_response_text(response)
        
        report_filename = os.path.join(output_dir, f"benchmark_report_{timestamp_str}.md")
        static_report_filename = os.path.join(output_dir, "benchmark_report.md")
        
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report_content)
        if not is_master:
            with open(static_report_filename, "w", encoding="utf-8") as f:
                f.write(report_content)
    except Exception as e:
        print(f"Error generating legacy report: {e}", file=sys.stderr)


def generate_master_report(
    output_dir: str,
    model_name: str = "gemini-3.5-flash",
    location: str = "global"
) -> None:
    report_files = glob.glob(os.path.join(output_dir, "benchmark_report_*.md"))
    report_files = [
        f for f in report_files 
        if "master" not in os.path.basename(f) and os.path.basename(f) != "benchmark_report.md"
    ]
    
    reports_content = []
    if report_files:
        for path in report_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                reports_content.append(f"=== REPORT FILE: {os.path.basename(path)} ===\n{content}\n")
            except Exception as e:
                print(f"Error: {e}")
                
    if not reports_content:
        return
        
    client = initialize_client(location=location)
    prompt = prompts.get_master_report_prompt(reports_content)
    
    code_exec_config = types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())],
        max_output_tokens=65536
    )
    
    try:
        response = generate_content_with_retry(client, model_name, prompt, config=code_exec_config)
        report_content = extract_response_text(response)
        master_report_path = os.path.join(output_dir, "benchmark_report_master.md")
        with open(master_report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark comparison report between AI Studio and Vertex AI.")
    parser.add_argument("--model", "-m", default="gemini-3.5-flash", help="Model to generate report (default: gemini-3.5-flash)")
    parser.add_argument("--location", "-l", default="global", help="Location of the model (default: global)")
    parser.add_argument("--ai-studio-dir", default="./reports_ai_studio", help="AI Studio reports directory (default: ./reports_ai_studio)")
    parser.add_argument("--vertex-ai-dir", default="./reports_vertex_ai", help="Vertex AI reports directory (default: ./reports_vertex_ai)")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory for the generated report (default: current directory)")
    parser.add_argument("--overlap-only", action="store_true", help="Only include models benchmarked on both AI Studio and Vertex AI")
    args = parser.parse_args()

    generate_comparison_report(
        ai_studio_dir=args.ai_studio_dir,
        vertex_ai_dir=args.vertex_ai_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        location=args.location,
        overlap_only=args.overlap_only
    )


if __name__ == "__main__":
    main()
