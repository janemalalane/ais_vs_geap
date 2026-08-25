"""
Prompt definitions for Gemini Benchmarking Tool Report Generator.
Contains persona instructions, code execution guidance, and prompt builders for
individual, comparison, and master benchmark reports.
"""

import json
from typing import Dict, Any, List, Optional


SYSTEM_PERSONA = (
    "You are an expert Principal Systems Engineer and Performance Analyst specializing in Cloud AI infrastructure."
)

CODE_EXECUTION_INSTRUCTIONS = (
    "### Advanced Data Analytics & Computer Use (Code Execution Enabled):\n"
    "You are equipped with Python Code Execution capabilities. "
    "To ensure maximum accuracy and depth in your systems analysis, write and execute Python code blocks to inspect, "
    "parse, aggregate, and compute exact telemetry metrics from the benchmark datasets provided.\n\n"
    "Use code execution to:\n"
    "1. **Latencies & Percentiles**: Calculate exact arithmetic means, medians (p50), and tail latencies (p90, p99) for TTFT and TTLT.\n"
    "2. **Throughput Deltas**: Calculate exact Tokens Per Second (TPS) differences and percentage deltas between Google AI Studio and GEAP.\n"
    "3. **Ratio Bucket Performance**: Analyze TTFT, TTLT, and TPS per ratio bucket (1:1, 2:1, 1:2, 10:1, 50:1).\n"
    "4. **Error & Tenancy Breakdown**: Analyze error status codes (e.g., 503 UNAVAILABLE vs 429 RATE_LIMIT) and calculate surface-level availability rates.\n\n"
    "Include the exact values computed by your Python code execution directly in your markdown report tables and narrative.\n\n"
)


def get_comparison_report_prompt(
    all_data: Dict[str, Any],
    error_summary: Dict[str, Any],
    timestamp_str: Optional[str] = None,
) -> str:
    """Constructs the prompt for generating a comparison report from multi-model, multi-surface files."""
    prompt = f"{SYSTEM_PERSONA}\n\n"
    prompt += (
        "Your task is to write a comprehensive, professional, and publication-ready Markdown report comparing "
        "Gemini models across two developer surfaces: Google AI Studio (Gemini Developer API) and "
        "Gemini Enterprise Agent Platform (GEAP) based on benchmark files collected from 'reports_ai_studio' and "
        "'reports_geap'.\n\n"
    )

    prompt += CODE_EXECUTION_INSTRUCTIONS

    prompt += (
        "Here is the collected multi-model, multi-surface benchmark data:\n"
        "```json\n"
        f"{json.dumps(all_data, indent=2)}\n"
        "```\n\n"
    )

    if error_summary:
        prompt += (
            "Here are specific raw error messages/failure details observed across the surfaces:\n"
            "```json\n"
            f"{json.dumps(error_summary, indent=2)}\n"
            "```\n\n"
        )
    else:
        prompt += "No significant errors or failures were logged during these benchmark runs.\n\n"

    prompt += (
        "Please write a comprehensive, professional, and data-driven systems engineering benchmark report for engineering teams. "
        "The report should focus strictly on empirical benchmark results, exact percentiles, and measured differences between Google AI Studio and GEAP. "
        "Maintain an objective, factual tone: present the measured data clearly, report observed latency distributions and HTTP status codes directly, and avoid making speculative assertions or unverified claims about internal backend architectures or tenancy pool mechanics.\n\n"
        "The report MUST include:\n"
        "1. **Title & Metadata**: An elegant, descriptive title for the benchmark results, followed immediately by the benchmark date and time"
    )
    if timestamp_str:
        prompt += f" (`> **Date & Time:** `{timestamp_str}``).\n"
    else:
        prompt += " (e.g., `> **Date & Time:** `YYYY-MM-DD HH:MM:SS UTC``).\n"

    prompt += (
        "2. **Executive Summary**: High-level factual takeaways summarizing key measured differences in latency (TTFT/TTLT), throughput (TPS/Images/sec), and success rates across model tiers.\n"
        "3. **Benchmark Methodology & Experimental Setup**: A clear, rigorous technical overview of the benchmarking setup for reproducibility, including:\n"
        "   - **Deterministic Seed**: Fixed `base_seed: 1782978773116` across both surfaces, ensuring identical prompt lengths, token contents, and arrival intervals.\n"
        "   - **Workload Parameters**:\n"
        "     * *Text Workloads*: Uniform [100, 500] input tokens (mean 250), [50, 200] target output tokens (mean 100), stepped traffic ramp (5 req/s for 20s, 15 req/s for 45s, 25 req/s for 89s -> 3,000 requests total per model tier).\n"
        "     * *Multimodal Vision*: 1–3 synthetic images per request (80% 1080p, 20% 4K) at prompt prefix with [100, 500] input tokens across stepped traffic ramp (1 req/s for 20s, 2 req/s for 20s, 3 req/s for 15s -> 105 requests total).\n"
        "   - **Test Harness Architecture**: `inference-perf` framework with a standardized tokenizer (`gemma-4-31B-it`), routing through an asynchronous local Dual Proxy Gateway to native Google AI Studio and GEAP endpoints, executed sequentially per model configuration to prevent client-side host contention.\n"
        "4. **Unified Surface Performance Comparison Matrix**: A single clean, highly readable Markdown table comparing all models side-by-side:\n"
        "   - Columns: `Model Name`, `Surface`, `Success Rate`, `Avg TTFT (s)`, `Median TTFT (s)`, `P99 TTFT (s)`, `Avg TTLT (s)`, `P99 TTLT (s)`, `Output TPS`, `Images/sec`, `Requests/sec`.\n"
        "   - **Surface Color Distinction**:\n"
        "     * Format AI Studio in the Surface column in bold Google Blue: `<span style=\"color: #1a73e8; font-weight: bold;\">AI Studio</span>`\n"
        "     * Format GEAP in the Surface column in bold Purple: `<span style=\"color: #7b1fa2; font-weight: bold;\">GEAP</span>`\n"
        "     * Keep all metric numbers as plain clean text for maximum readability.\n"
        "     * Group by model so both surfaces for the same model appear on consecutive rows.\n"
        "     * For multimodal image models, display `N/A` for TTFT columns.\n"
        "     * For text models, display `—` for `Images/sec`.\n"
        "   - Fill in the values precisely using Python code execution to verify percentiles and perform comparisons.\n"
        "5. **Deep-Dive Latency & Throughput Analysis**:\n"
        "   - **Prefill Latency Comparison (TTFT)**: Compare time to first token across text-only model tiers (p50, avg, tail p99). Do NOT evaluate multimodal image models under TTFT; evaluate them under Multimodal Throughput (Images/sec) and Total Request Latency (TTLT).\n"
        "   - **Generation Throughput Comparison (TPS & Images/sec)**: Compare output token generation speed for text models and image processing throughput (Images/sec) for multimodal models.\n"
        "   - **Total Request Latency (TTLT)**: Analyze end-to-end turnaround times and tail latency behaviors across all models.\n"
        "6. **Model-by-Model Breakdown**: Provide a concise and direct bullet-point breakdown for EVERY model tested, highlighting key metrics (latencies, throughput, request counts).\n"
        "7. **Observed Reliability & Status Codes**:\n"
        "   - Report observed HTTP status codes (e.g. 200 OK, 429 Rate Limit, 503 Unavailable) and success rates for each surface objectively based on the benchmark logs.\n\n"
        "Please structure the report cleanly with markdown tables, bold text, and bullet points. "
        "Write in an objective, engineering-focused tone. Include the date and time metadata header under the main title. Do NOT wrap the entire response in a markdown block (e.g. ```markdown ... ```), "
        "just output the raw markdown directly so it is ready to be written to a file."
    )
    return prompt
