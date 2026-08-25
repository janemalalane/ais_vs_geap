#!/usr/bin/env python3
"""
Report Generator for Gemini Models API Benchmarking Tool
Uses Gemini to analyze benchmark results from reports_ai_studio and reports_geap
and write a unified Markdown report comparing both developer surfaces.
"""

import os
import sys
import json
import glob
import re
import time
from datetime import datetime, timezone
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

try:
    import prompts
except ImportError:
    from gemini_report_synthesis import prompts

# Import the unified Google Gen AI SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    print(
        "Error: The 'google-genai' SDK is not installed in the current environment.",
        file=sys.stderr,
    )
    print("Please install it using: pip install google-genai", file=sys.stderr)
    sys.exit(1)


def initialize_client(location: Optional[str] = None) -> genai.Client:
    """Initializes the GenAI Client using AI Studio or GEAP Cloud backend depending on available env vars."""
    api_key = os.environ.get("API_KEY") or os.environ.get("GEMINI_API_KEY")
    project = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if location is None:
        location = "global"

    if api_key:
        print("[Report Gen] Initializing client using AI Studio (API_KEY)...")
        return genai.Client(api_key=api_key)
    elif project:
        print(
            f"[Report Gen] Initializing client using GEAP Cloud backend (Project: {project}, Location: {location})..."
        )
        return genai.Client(vertexai=True, project=project, location=location)
    else:
        print(
            "[Report Gen] Warning: Neither API_KEY nor PROJECT_ID are set in environment."
        )
        print(
            "[Report Gen] Attempting default Client initialization (might fail if no credentials)..."
        )
        return genai.Client()


def generate_content_with_retry(
    client: genai.Client,
    model: str,
    contents: Any,
    config: Optional[types.GenerateContentConfig] = None,
    max_retries: int = 5,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
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
            if (
                "429" in err_str
                or "rate limit" in err_str
                or "resource_exhausted" in err_str
            ):
                status_code = 429
            elif "503" in err_str or "unavailable" in err_str:
                status_code = 503
            elif "500" in err_str or "internal" in err_str:
                status_code = 500

            if status_code in [429, 500, 503] and attempt < max_retries:
                delay = initial_delay * (backoff_factor**attempt) + random.uniform(0, 1)
                print(
                    f"[Retry] API call failed (status {status_code}). Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})..."
                )
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


def extract_timestamp(
    paths: Optional[List[Optional[str]]] = None,
    files: Optional[List[str]] = None,
) -> str:
    """
    Extracts or formats a human-readable benchmark timestamp (Date & Time).
    Tries:
    1. Regex match for YYYYMMDD_HHMMSS in provided paths.
    2. Most recent mtime of input files.
    3. Current UTC datetime.
    """
    if paths:
        for p in paths:
            if p:
                match = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", p)
                if match:
                    year, month, day, hour, minute, second = match.groups()
                    return f"{year}-{month}-{day} {hour}:{minute}:{second} UTC"

    if files:
        latest_mtime = 0.0
        for f in files:
            if os.path.exists(f):
                mtime = os.path.getmtime(f)
                if mtime > latest_mtime:
                    latest_mtime = mtime
        if latest_mtime > 0:
            dt = datetime.fromtimestamp(latest_mtime, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def format_report_with_timestamp(report_content: str, timestamp_str: str) -> str:
    """Ensures the report contains the Date and Time in the header metadata block."""
    lines = report_content.split("\n")
    first_few_lines = "\n".join(lines[:15]).lower()
    if "date & time" in first_few_lines or "date:" in first_few_lines or "timestamp:" in first_few_lines:
        return report_content

    # Find the title line (e.g. # Title...)
    title_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("# "):
            title_idx = i
            break

    if title_idx != -1:
        # Insert Date & Time directly under the title as a styled blockquote
        meta_line = f"\n> **Date & Time:** `{timestamp_str}`\n"
        lines.insert(title_idx + 1, meta_line)
        return "\n".join(lines)
    else:
        # Prepend to content
        return f"> **Date & Time:** `{timestamp_str}`\n\n{report_content}"


def load_summary_metrics(file_path: str) -> Dict[str, Any]:
    """Loads a summary lifecycle metrics JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_name_and_metadata(
    config_path: str, file_prefix: str
) -> Tuple[str, bool, bool]:
    """
    Extracts model name, streaming setting, and multimodal presence from a yaml config.
    Returns (model_name, is_streaming, is_multimodal).
    """
    is_streaming = True
    is_multimodal = False
    model_name_from_cfg = None

    if os.path.exists(config_path):
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            if cfg:
                if "server" in cfg and "model_name" in cfg["server"]:
                    model_name_from_cfg = cfg["server"]["model_name"]
                if "api" in cfg and "streaming" in cfg["api"]:
                    is_streaming = cfg["api"]["streaming"]
                if "data" in cfg and "multimodal" in cfg["data"]:
                    is_multimodal = cfg["data"]["multimodal"] is not None
        except Exception as e:
            print(f"[Warning] Failed to parse config {config_path}: {e}")

    clean_pref = file_prefix
    if clean_pref.startswith("ai_studio_"):
        clean_pref = clean_pref[len("ai_studio_") :]
    elif clean_pref.startswith("geap_"):
        clean_pref = clean_pref[len("geap_") :]

    if clean_pref.endswith("_"):
        clean_pref = clean_pref[:-1]

    # Strip trailing _multimodal or -multimodal from prefix
    if clean_pref.endswith("_multimodal"):
        clean_pref = clean_pref[: -len("_multimodal")]
    elif clean_pref.endswith("-multimodal"):
        clean_pref = clean_pref[: -len("-multimodal")]

    if model_name_from_cfg:
        model_name = model_name_from_cfg.replace("_", "-")
        if model_name.endswith("-multimodal"):
            model_name = model_name[: -len("-multimodal")]
    else:
        model_name = clean_pref.replace("_", "-")

    if model_name.startswith("google/"):
        model_name = model_name[len("google/") :]
    elif model_name.startswith("google-"):
        model_name = model_name[len("google-") :]

    if "image" in model_name or "image" in file_prefix:
        is_multimodal = True

    return model_name, is_streaming, is_multimodal


def parse_summary_metrics(
    summary_data: Dict[str, Any], is_multimodal: bool = False
) -> Dict[str, Any]:
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

    # For multimodal image models, TTFT is not applicable (focus is on Images/sec and TTLT)
    ttft_metrics = {
        "mean": None if is_multimodal else ttft.get("mean"),
        "median": None if is_multimodal else ttft.get("median"),
        "p90": None if is_multimodal else ttft.get("p90"),
        "p99": None if is_multimodal else ttft.get("p99"),
    }

    return {
        "benchmark_time_seconds": summary_data.get("benchmark_time_seconds"),
        "success_rate": success_rate,
        "success_count": success_count,
        "failure_count": failure_count,
        "total_requests": total_count,
        "ttft": ttft_metrics,
        "ttlt": {
            "mean": ttlt.get("mean"),
            "median": ttlt.get("median"),
            "p90": ttlt.get("p90"),
            "p99": ttlt.get("p99"),
        },
        "tpot": {
            "mean": tpot.get("mean"),
            "median": tpot.get("median"),
            "p90": tpot.get("p90"),
            "p99": tpot.get("p99"),
        },
        "itl": {
            "mean": itl.get("mean"),
            "median": itl.get("median"),
            "p90": itl.get("p90"),
            "p99": itl.get("p99"),
        },
        "ntpot": {
            "mean": ntpot.get("mean"),
            "median": ntpot.get("median"),
            "p90": ntpot.get("p90"),
            "p99": ntpot.get("p99"),
        },
        "throughput": {
            "output_tokens_per_sec": throughput.get("output_tokens_per_sec"),
            "input_tokens_per_sec": throughput.get("input_tokens_per_sec"),
            "total_tokens_per_sec": throughput.get("total_tokens_per_sec"),
            "requests_per_sec": throughput.get("requests_per_sec"),
            "images_per_sec": throughput.get("images_per_sec", 0.0),
            "videos_per_sec": throughput.get("videos_per_sec", 0.0),
            "audios_per_sec": throughput.get("audios_per_sec", 0.0),
        },
        "request_size_bytes": {
            "mean": request_size_bytes.get("mean"),
            "median": request_size_bytes.get("median"),
            "p90": request_size_bytes.get("p90"),
            "p99": request_size_bytes.get("p99"),
        },
        "prompt_len": {
            "mean": prompt_len.get("mean"),
            "median": prompt_len.get("median"),
            "p90": prompt_len.get("p90"),
            "p99": prompt_len.get("p99"),
        },
        "output_len": {
            "mean": output_len.get("mean"),
            "median": output_len.get("median"),
            "p90": output_len.get("p90"),
            "p99": output_len.get("p99"),
        },
        "prompt_tokens": {
            "total": prompt_tokens.get("total"),
            "cached": prompt_tokens.get("cached"),
            "uncached": prompt_tokens.get("uncached"),
        },
        "output_tokens": {"total": output_tokens.get("total")},
        "modalities": {
            "image_count_mean": get_mean(image_stats.get("count")),
            "video_count_mean": get_mean(video_stats.get("count")),
            "audio_count_mean": get_mean(audio_stats.get("count")),
        },
    }


def find_subfolder(parent: str, prefixes: List[str]) -> Optional[str]:
    """Finds a matching subfolder in parent directory by checking prefix/pattern."""
    if not os.path.isdir(parent):
        return None
    entries = sorted(os.listdir(parent))
    # Direct match first, then prefix match
    for prefix in prefixes:
        for entry in entries:
            full = os.path.join(parent, entry)
            if os.path.isdir(full) and (
                entry.lower() == prefix.lower()
                or entry.lower().startswith(prefix.lower())
            ):
                return full
    return None


def resolve_report_directories(
    run_dir: Optional[str] = None,
    ai_studio_dir: Optional[str] = None,
    geap_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Resolves AI Studio directory, GEAP directory, and Output directory.
    Supports:
    - Dedicated run_dir (e.g. reports/report_YYYYMMDD_HHMMSS/ with ai_studio/ and geap/ subdirectories).
    - Explicit ai_studio_dir and geap_dir.
    - Auto-discovery of the latest timestamped run under reports/ or ./
    - Fallback to legacy reports_ai_studio and reports_geap.
    """
    # 1. Explicit run_dir provided
    if run_dir:
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(
                f"Specified run directory does not exist: {run_dir}"
            )
        ais_path = find_subfolder(run_dir, ["ai_studio", "aistudio", "ais"])
        geap_path = find_subfolder(run_dir, ["geap"])
        out_path = output_dir if output_dir and output_dir != "." else run_dir
        return (
            ais_path or os.path.join(run_dir, "ai_studio"),
            geap_path or os.path.join(run_dir, "geap"),
            out_path,
        )

    # 2. Explicit ai_studio_dir or geap_dir provided
    if ai_studio_dir or geap_dir:
        ais_path = ai_studio_dir or "./reports_ai_studio"
        geap_path = geap_dir or "./reports_geap"
        out_path = output_dir or "."
        return ais_path, geap_path, out_path

    # 3. Auto-discovery: search for latest run folder in reports/ or ./
    candidates = []
    for base in ["reports", "."]:
        if os.path.isdir(base):
            for entry in os.listdir(base):
                full = os.path.join(base, entry)
                if os.path.isdir(full) and (
                    entry.startswith("report_")
                    or entry.startswith("reports_")
                    or entry.startswith("run_")
                ):
                    if entry in ["reports_ai_studio", "reports_geap"]:
                        continue
                    candidates.append((os.path.getmtime(full), full))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        latest_dir = candidates[0][1]
        ais_path = find_subfolder(latest_dir, ["ai_studio", "aistudio", "ais"])
        geap_path = find_subfolder(latest_dir, ["geap"])
        if ais_path or geap_path:
            print(
                f"[Auto-Discovery] Detected latest benchmark run directory: '{latest_dir}'"
            )
            out_path = output_dir if output_dir and output_dir != "." else latest_dir
            return (
                ais_path or os.path.join(latest_dir, "ai_studio"),
                geap_path or os.path.join(latest_dir, "geap"),
                out_path,
            )

    # 4. Fallback defaults
    return "./reports_ai_studio", "./reports_geap", output_dir or "."


def generate_comparison_report(
    ai_studio_dir: str,
    geap_dir: str,
    output_dir: str,
    model_name: str = "gemini-3.7-flash",
    location: str = "global",
    overlap_only: bool = False,
) -> None:
    """
    Collects summary metrics from reports_ai_studio and reports_geap,
    aggregates them, and asks Gemini to output a unified comparison report.
    """
    print("\n" + "=" * 70)
    print("                 COLLECTING BENCHMARK SUMMARY METRICS")
    print("=" * 70)

    # Auto-detect alternate directories if the specified directories do not exist
    if not os.path.exists(ai_studio_dir):
        for alt_path in [
            "./reports_ai_studio",
            "reports_ai_studio",
            "./reports_aistudio",
            "reports_aistudio",
        ]:
            if os.path.exists(alt_path):
                print(
                    f"[Collector] Directory '{ai_studio_dir}' not found. Using detected alternative: '{alt_path}'"
                )
                ai_studio_dir = alt_path
                break

    if not os.path.exists(geap_dir):
        for alt_path in ["./reports_geap", "reports_geap"]:
            if os.path.exists(alt_path):
                print(
                    f"[Collector] Directory '{geap_dir}' not found. Using detected alternative: '{alt_path}'"
                )
                geap_dir = alt_path
                break

    all_data = {}
    all_scanned_files = []

    # Define directories and surfaces
    scan_configs = [(ai_studio_dir, "ai_studio"), (geap_dir, "geap")]

    for dir_path, surface in scan_configs:
        if not os.path.exists(dir_path):
            print(f"[Info] Directory {dir_path} does not exist. Skipping.")
            continue

        summary_pattern = os.path.join(dir_path, "*_summary_lifecycle_metrics.json")
        summary_files = glob.glob(summary_pattern)

        # Filter out generic summary_lifecycle_metrics.json if we have specific prefix files
        specific_files = [
            f
            for f in summary_files
            if os.path.basename(f) != "summary_lifecycle_metrics.json"
        ]
        if not specific_files and summary_files:
            specific_files = summary_files

        print(
            f"[Collector] Found {len(specific_files)} summary file(s) in '{dir_path}' for surface '{surface}':"
        )
        for file_path in specific_files:
            all_scanned_files.append(file_path)
            filename = os.path.basename(file_path)
            print(f"  - {filename}")

            # Determine prefix
            prefix = filename.replace("summary_lifecycle_metrics.json", "")
            config_path = os.path.join(dir_path, f"{prefix}config.yaml")

            # Extract metadata
            m_name, is_streaming, is_multimodal = get_model_name_and_metadata(
                config_path, prefix
            )

            try:
                summary_json = load_summary_metrics(file_path)
                metrics = parse_summary_metrics(
                    summary_json, is_multimodal=is_multimodal
                )
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
            if "ai_studio" in surfaces and "geap" in surfaces
        }
        print(
            f"\n[Collector] Filtering for overlap only. Models with both surfaces: {len(all_data)}"
        )

    if not all_data:
        print(
            "[Error] No benchmark summary metrics files found in either directory matching criteria.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n[Collector] Aggregated Model Data Summary:")
    for m_name, surfaces in all_data.items():
        print(f"  - {m_name}: Surfaces: {list(surfaces.keys())}")

    timestamp_str = extract_timestamp(
        paths=[output_dir, ai_studio_dir, geap_dir],
        files=all_scanned_files,
    )
    print(f"[Report Gen] Benchmark Date & Time: {timestamp_str}")

    # Initialize client and generate report
    client = initialize_client(location=location)
    prompt = prompts.get_comparison_report_prompt(
        all_data, {}, timestamp_str=timestamp_str
    )

    # Configure Code Execution tool for exact math calculations
    code_exec_config = types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())],
        max_output_tokens=65536,
    )

    print(
        f"\n[Report Gen] Requesting comparison report from {model_name} (Location: {location}, Code Execution enabled)..."
    )
    try:
        response = generate_content_with_retry(
            client, model_name, prompt, config=code_exec_config
        )
        report_content = extract_response_text(response)
        report_content = format_report_with_timestamp(report_content, timestamp_str)

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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate benchmark comparison report between AI Studio and GEAP."
    )
    parser.add_argument(
        "--run-dir",
        "-r",
        default=None,
        help="Benchmark run directory containing ai_studio and geap subfolders (e.g., reports/report_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gemini-3.7-flash",
        help="Model to generate report (default: gemini-3.7-flash)",
    )
    parser.add_argument(
        "--location",
        "-l",
        default="global",
        help="Location of the model (default: global)",
    )
    parser.add_argument(
        "--ai-studio-dir", default=None, help="AI Studio reports directory"
    )
    parser.add_argument(
        "--geap-dir", dest="geap_dir", default=None, help="GEAP reports directory"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for the generated report",
    )
    parser.add_argument(
        "--overlap-only",
        action="store_true",
        help="Only include models benchmarked on both AI Studio and GEAP",
    )
    args = parser.parse_args()

    ai_studio_dir, geap_dir, output_dir = resolve_report_directories(
        run_dir=args.run_dir,
        ai_studio_dir=args.ai_studio_dir,
        geap_dir=args.geap_dir,
        output_dir=args.output_dir,
    )

    generate_comparison_report(
        ai_studio_dir=ai_studio_dir,
        geap_dir=geap_dir,
        output_dir=output_dir,
        model_name=args.model,
        location=args.location,
        overlap_only=args.overlap_only,
    )


if __name__ == "__main__":
    main()
