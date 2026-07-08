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
    "2. **Throughput Deltas**: Calculate exact Tokens Per Second (TPS) differences and percentage deltas between Google AI Studio and Vertex AI (GEAP).\n"
    "3. **Ratio Bucket Performance**: Analyze TTFT, TTLT, and TPS per ratio bucket (1:1, 2:1, 1:2, 10:1, 50:1).\n"
    "4. **Error & Tenancy Breakdown**: Analyze error status codes (e.g., 503 UNAVAILABLE vs 429 RATE_LIMIT) and calculate surface-level availability rates.\n\n"
    "Include the exact values computed by your Python code execution directly in your markdown report tables and narrative.\n\n"
)


def get_comparison_report_prompt(all_data: Dict[str, Any], error_summary: Dict[str, Any]) -> str:
    """Constructs the prompt for generating a comparison report from multi-model, multi-surface files."""
    prompt = f"{SYSTEM_PERSONA}\n\n"
    prompt += (
        "Your task is to write a comprehensive, professional, and publication-ready Markdown report comparing "
        "Gemini models across two developer surfaces: Google AI Studio (Gemini Developer API) and Google Vertex AI "
        "(Google Enterprise AI Platform / GEAP) based on benchmark files collected from 'reports_ai_studio' and "
        "'reports_vertex_ai'.\n\n"
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
        "Please write a comprehensive, professional systems engineering benchmark report. "
        "Your analysis MUST focus on comparing the two surfaces (Google AI Studio vs Vertex AI/GEAP) across different model tiers and modalities. "
        "If data for one of the surfaces (e.g., Vertex AI) is missing or pending, explain that Vertex AI results are currently pending/not yet created, "
        "and focus the analysis on evaluating the AI Studio results while providing systems-architectural context for what to expect on Vertex AI.\n\n"
        "The report MUST include:\n"
        "1. **Title**: An elegant, descriptive title for the comparison report.\n"
        "2. **Executive Summary**: High-level takeaways comparing the two surfaces, highlighting key performance, reliability, and latency trade-offs (multi-tenant best-effort developer pool vs. enterprise-tenant isolated SLA pool).\n"
        "3. **Unified Surface Performance Comparison Matrix**: A detailed Markdown table showing side-by-side performance of all models found in the dataset:\n"
        "   - Columns: `Model Name`, `Surface`, `Success Rate`, `Avg TTFT (s)`, `Median TTFT (s)`, `P99 TTFT (s)`, `Avg TTLT (s)`, `P99 TTLT (s)`, `Output TPS`, `Images/sec` (if applicable), `Requests/sec`.\n"
        "   - Fill in the values precisely using Python code execution to verify percentiles and perform comparisons.\n"
        "4. **Deep-Dive Latency & Throughput Analysis**:\n"
        "   - **Prefill Latency Comparison (TTFT)**: Analyze which surface delivers the first token faster across all tiers. Look at standard vs tail (p99) latencies.\n"
        "   - **Generation Throughput Comparison (TPS & Images/sec)**: Compare output tokens per second or image processing throughput (for multimodal models) across model tiers (lite, standard, pro) and identify if one surface has a clear computational advantage.\n"
        "   - **Autoregressive Decode & Overall Latency (TTLT)**: Discuss how TTFT, TPS, and multimodal processing overhead combine to impact the total roundtrip response latency (TTLT).\n"
        "5. **Model-by-Model Breakdown**: Provide a concise and direct bullet-point breakdown for EVERY model found in the dataset (including text models and multimodal/image models like `gemini-3.1-flash-lite-image` and `gemini-3.1-flash-image`). Keep each model's summary concise, direct, and highlighting key metrics (like latencies, TPS, or Images/sec).\n"
        "6. **Platform Reliability & Tenancy Profiles**:\n"
        "   - Contrast the reliability of the surfaces based on success rates and error codes (like 503 errors on AI Studio vs Vertex's isolated enterprise architecture).\n"
        "   - Explain image/multimodal model behaviors and failure rates as configuration/routing issues rather than platform outages, showing how API structures impact developer integration.\n\n"
        "Please structure the report beautifully with clear headings, bullet points, bold text, and markdown tables. "
        "Write in an objective, analytical, and professional tone. Do NOT wrap the entire response in a markdown block (e.g. ```markdown ... ```), "
        "just output the raw markdown directly so it is ready to be written to a file."
    )
    return prompt


def get_report_prompt(summary_data: Dict[str, Any], error_summary: Dict[str, Any], is_master: bool = False) -> str:
    """Constructs the prompt for individual benchmark report generation (retained for backward compatibility)."""
    prompt = f"{SYSTEM_PERSONA}\n\n"
    prompt += (
        "Your task is to analyze the benchmark results comparing Gemini models across two developer surfaces: "
        "Google AI Studio (Gemini Developer API) and Google Vertex AI / GEAP (Google Enterprise AI Platform).\n\n"
    )

    prompt += CODE_EXECUTION_INSTRUCTIONS

    if is_master:
        prompt += (
            "IMPORTANT: This is a MASTER AGGREGATED report. The data provided represents the AVERAGE and aggregated performance "
            "across multiple independent benchmark runs. This aggregation is intended to 'denoise' the performance metrics "
            "and show the true, stable differences between the surfaces. Highlight this multi-run, denoised nature "
            "of the data in your Executive Summary and throughout the analysis. Frame the findings as a stable baseline.\n"
            "You MUST explicitly list out all sub-reports/run summaries ingested, and ensure EVERY model tested across all runs "
            "is analyzed individually with dedicated latency, throughput, and error metrics.\n\n"
        )

    prompt += (
        "Here is the aggregated benchmark summary JSON data:\n"
        "```json\n"
        f"{json.dumps(summary_data, indent=2)}\n"
        "```\n\n"
    )

    if error_summary:
        prompt += (
            "Here are the specific raw error messages encountered during failed requests:\n"
            "```json\n"
            f"{json.dumps(error_summary, indent=2)}\n"
            "```\n\n"
        )

    # Inject prompt library & ratio performance instructions if present
    if "__ratio_summary__" in summary_data:
        prompt += (
            "### Prompt Library & Ratio Performance Analysis Instructions:\n"
            "The benchmark data contains a detailed breakdown of results across different **input-to-output token ratio buckets** "
            "(located under the '__ratio_summary__' key in the JSON). This represents a structured stress-test using a prompt library "
            "across 5 ratios (1:1, 2:1, 1:2, 10:1, 50:1) and 6 themes (tech, science, legal, fiction, finance, logs).\n\n"
            "In your report, you MUST include a dedicated section analyzing this prompt library run, discussing:\n"
            "1. **Latency & Throughput Scaling by Ratio**: How did different ratio-driven generation workloads impact TTFT and TPS? "
            "Does prefill latency dominate for heavy-context inputs, or does decoding throughput dominate for heavy-generation targets?\n\n"
        )

    prompt += (
        "Please write a comprehensive, professional, and publication-ready Markdown report analyzing these results. "
        "Your analysis MUST focus on comparing the two surfaces (Google AI Studio vs Vertex AI/GEAP) rather than just the models. "
        "The central question you must answer is: **How does Google AI Studio perform compared to Vertex AI/GEAP across different model tiers and modalities?**\n\n"
        "The report MUST include:\n"
        "1. **Executive Summary**: High-level takeaways comparing the two surfaces, highlighting key performance, reliability, and latency trade-offs.\n"
        "2. **Surface-to-Surface Performance Analysis (The Core Comparison)**:\n"
        "   - **Prefill Latency Comparison (TTFT)**: Analyze whether AI Studio or Vertex AI/GEAP is faster in delivering the first token across all models. Examine if this holds true for flash-lite, flash, and pro models. Discuss average, p50, and tail (p99) latency differences between the two surfaces.\n"
        "   - **Generation Throughput Comparison (TPS & Images/sec)**: Compare computational performance of the surfaces. Analyze the throughput delta (AI Studio vs Vertex AI) for each text model tier (lite, standard, pro) and multimodal model, and identify if one surface has a clear computational advantage.\n"
        "   - **Autoregressive Decode & Overall Latency (TTLT)**: Compare overall response times. Analyze if throughput differences translate directly into overall latency wins, or if prefill latency (TTFT), multimodal input processing, or queue times dominate.\n"
        "   - **Dedicated Model Breakdown**: Provide a concise and direct bullet-point breakdown for EVERY tested model (including text models and multimodal image models like `gemini-3.1-flash-lite-image`, `gemini-3.1-flash-image`, `gemini-3-pro-image`). Keep each model's summary concise, direct, and formatted as bullet points highlighting key metrics and observed behavior.\n"
        "   - **Multimodal Image Performance Results**: Explicitly share and analyze results for multimodal image models across both surfaces, including TTLT, latency deltas between AI Studio and Vertex AI, success rates, Images/sec throughput, and API routing overhead.\n"
        "3. **Surface Reliability & Tenancy Profiles**:\n"
        "   - Contrast the reliability of the two surfaces. Deep dive into failure rates (e.g. 503 UNAVAILABLE) of models on AI Studio due to public shared queue saturation, compared to Vertex AI. Use this to explain the architectural difference between AI Studio's multi-tenant, best-effort developer pool and Vertex AI's enterprise tenant isolation and SLA guarantees.\n"
        "   - Explain image/multimodal model performance and failure rates as a configuration/routing issue (legacy ML Predict on Vertex vs invalid model names on AI Studio) rather than platform outages, showing how API structures impact developer integration.\n\n"
        "Please structure the report beautifully with headers, bullet points, bold text, and markdown tables where helpful. "
        "Write in an objective, analytical, and professional tone. Do NOT wrap the entire response in a markdown block (e.g. ```markdown ... ```), just output the raw markdown directly so it is ready to be written to a file."
    )
    return prompt


def get_master_report_prompt(reports_content: List[str]) -> str:
    """Constructs the prompt for master report aggregation."""
    prompt = f"{SYSTEM_PERSONA}\n\n"
    prompt += (
        "Your task is to take multiple individual benchmark reports comparing Google AI Studio and Vertex AI (GEAP), "
        "and aggregate them into a single, definitive 'Master Report'.\n\n"
        "The goal of this master report is to **average the benchmarking metrics across several runs** to observe a "
        "more denoised, stable view of the differences in performance between AI Studio and GEAP.\n\n"
    )

    prompt += CODE_EXECUTION_INSTRUCTIONS

    prompt += (
        "Here are the individual reports to aggregate:\n\n"
        f"{'\n'.join(reports_content)}\n\n"
        "### Instructions for Aggregation:\n"
        "1. **List of Sub-Reports**: At the top of the Master Report (within the Executive Summary or in a dedicated section titled 'Analyzed Sub-Reports'), "
        "you MUST explicitly list out all individual sub-reports and run files (e.g., `benchmark_report_<timestamp>.md` or summary JSON files) "
        "that were ingested and aggregated to build this master analysis.\n"
        "2. **Analysis of Every Model Across Runs**: In your performance sections, you MUST provide explicit analysis "
        "for EVERY single model present across the benchmark runs (including all text models like `gemini-2.5-flash`, `gemini-2.5-flash-lite`, "
        "`gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, etc., and all multimodal image models like `gemini-2.5-flash-image`, `gemini-3.1-flash-image`, "
        "`gemini-3-pro-image`, `gemini-3.1-flash-lite-image`). Ensure every model has its own distinct data points, latency/throughput comparisons "
        "(TTFT, TTLT, TPS, or Images/sec), error/tenancy profiles, and cross-run stability evaluation.\n"
        "3. **Concise Dedicated Model Breakdown & Multimodal Image Results**: In the 'Dedicated Model Breakdown' section, use a concise, direct bullet-point format for each model. "
        "Make sure to explicitly share results about multimodal image models (e.g. `gemini-3.1-flash-lite-image`, `gemini-3.1-flash-image`, `gemini-3-pro-image`), detailing their TTLT, latency differences between AI Studio and Vertex AI, and routing behaviors.\n"
        "4. **Average the Metrics**: For every table (e.g., Prefill Latency, Generation Throughput, Ratio Performance, Reliability), "
        "extract the numeric values for each model/surface combination from all the provided reports, "
        "calculate their arithmetic average (mean) using Python code execution, and present these averaged values in the final tables. "
        "Be precise with your calculations.\n"
        "5. **Denoised Narrative**: Synthesize the qualitative insights and systems insights into a single, cohesive narrative. Frame the report as a stable baseline that has been 'denoised' by averaging across multiple runs. Explain in the introduction that this report represents the aggregate of multiple runs.\n"
        "6. **Structure**: Maintain a professional, publication-ready structure:\n"
        "   - **1. Executive Summary & Sub-Reports Index**: High-level takeaways, Key Takeaways, and an explicit bulleted list of all ingested sub-reports/run files.\n"
        "   - **2. Model-by-Model & Surface Performance Analysis**: Comprehensive evaluation of EVERY model across runs, including Prefill Latency (TTFT) table and analysis per model, Generation Throughput (TPS / Images/sec) table and analysis per model, Overall Latency (TTLT) analysis per model, concise bullet-point Dedicated Model Breakdown, and explicit Multimodal Image results. Use the averaged metrics.\n"
        "   - **3. Prompt Library & Ratio Performance Analysis** (if present in the reports): Analyze latency and throughput scaling across ratio buckets (averaging TTFT, TTLT, and TPS metrics per model).\n"
        "   - **4. Surface Reliability & Tenancy Profiles**: Discuss reliability, 503 errors, image API routing overhead, and tenancy differences based on the collective experience of all runs for every model.\n"
        "7. **Tone**: Objective, analytical, and professional.\n\n"
        "Do NOT wrap the entire response in a markdown block (e.g. ```markdown ... ```), just output the raw markdown directly."
    )
    return prompt
