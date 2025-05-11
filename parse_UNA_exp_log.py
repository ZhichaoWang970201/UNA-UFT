import re
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
'''
    Parsing the results of UNA experiments' logs, 6 methods on 12 tasks. Hard problem!
'''

# --- Configuration ---
METHOD_DEFINITIONS = {
    "gemma-3-4b-pt": "hf (pretrained=google/gemma-3-4b-pt,trust_remote_code=True,dtype=bfloat16)",
    "gemma-3-4b+dpo_0.03": "hf (pretrained=google/gemma-3-4b-pt,peft=./dpo_0.03/final_checkpoint,trust_remote_code=True,dtype=bfloat16)",
    "gemma-3-4b+kto_0.03": "hf (pretrained=google/gemma-3-4b-pt,peft=./kto_0.03/final_checkpoint,trust_remote_code=True,dtype=bfloat16)",
    "gemma-3-4b+una_binary_MSE": "hf (pretrained=google/gemma-3-4b-pt,peft=./una_binary_MSE_5e-6_0.01/final_checkpoint,trust_remote_code=True,dtype=bfloat16)",
    "gemma-3-4b+una_binary_BCE": "hf (pretrained=google/gemma-3-4b-pt,peft=./una_binary_BCE_5e-6_0.01/final_checkpoint,trust_remote_code=True,dtype=bfloat16)",
    "gemma-3-4b+una_score_MSE": "hf (pretrained=google/gemma-3-4b-pt,peft=./una_score_MSE_3e-5_0.03/final_checkpoint,trust_remote_code=True,dtype=bfloat16)",
}
ORDERED_METHOD_NAMES = [
    "gemma-3-4b-pt", "gemma-3-4b+dpo_0.03", "gemma-3-4b+kto_0.03",
    "gemma-3-4b+una_binary_MSE", "gemma-3-4b+una_binary_BCE", "gemma-3-4b+una_score_MSE",
]

TASKS_CONFIG = [
    {"col_name": "bbh", "log_name": "leaderboard_bbh", "target_metric": "acc_norm", "type": "direct_prefer_groups"},
    {"col_name": "gpqa", "log_name": "leaderboard_gpqa", "target_metric": "acc_norm", "type": "direct_prefer_groups"},
    {"col_name": "mmlu_pro", "log_name": "leaderboard_mmlu_pro", "target_metric": "acc", "type": "direct"},
    {"col_name": "musr", "log_name": "leaderboard_musr", "target_metric": "acc_norm", "type": "direct_prefer_groups"},
    {"col_name": "ifeval", "log_name": "leaderboard_ifeval", "type": "ifeval_average_4_metrics",
     "metrics_to_average": ["inst_level_loose_acc", "inst_level_strict_acc", "prompt_level_loose_acc", "prompt_level_strict_acc"]},
    {"col_name": "math_hard", "log_name": "leaderboard_math_hard", "target_metric": "exact_match", "type": "direct_prefer_groups"},
    {"col_name": "gsm8k", "log_name": "gsm8k", "target_metric": "exact_match", "type": "gsm8k_average_2_lines"},
    {"col_name": "truthfulqa", "type": "truthfulqa_average_mc1_mc2",
     "mc1_log_name": "truthfulqa_mc1", "mc2_log_name": "truthfulqa_mc2", "target_metric": "acc"},
    {"col_name": "winogrande", "log_name": "winogrande", "target_metric": "acc", "type": "direct"},
    # Updated type for arc_c and hellaswag
    {"col_name": "arc_c", "log_name": "arc_challenge", "target_metric_on_continuation": "acc_norm", "type": "metric_on_continuation"},
    {"col_name": "hellaswag", "log_name": "hellaswag", "target_metric_on_continuation": "acc_norm", "type": "metric_on_continuation"},
    {"col_name": "mmlu", "log_name": "mmlu", "target_metric": "acc", "type": "direct_prefer_groups"},
]
# --- End of Configuration ---

def extract_value_from_row(line: str, task_name_on_line_pattern: str, target_metric: str, debug_mode: bool = False, task_col_name_for_debug: str = "") -> Optional[float]:
    task_name_segment = re.escape(task_name_on_line_pattern) if task_name_on_line_pattern != r"\s*" else task_name_on_line_pattern
    regex = (
        r"\|\s*" + task_name_segment + r"\s*\|" 
        r"(?:[^|]*\|)*?"                                  
        r"\s*" + re.escape(target_metric) + r"\s*\|"      
        r"[^|]*\|"                                        
        r"\s*([\d.-]+)\s*\|"                              
    )
    if debug_mode: print(f"        [DEBUG] Regex for '{task_col_name_for_debug}': {regex} ON LINE: \"{line}\"")
    match = re.search(regex, line)
    if match:
        try:
            val_str = match.group(1)
            if debug_mode: print(f"          [DEBUG] Regex Matched! Value_str: '{val_str}', Metric: '{target_metric}'")
            return float(val_str)
        except ValueError:
            if debug_mode: print(f"          [DEBUG] Regex Matched but ValueError converting '{val_str}'")
            return None
    elif debug_mode:
        # For detailed failure analysis if needed later:
        # if re.search(r"\|\s*" + task_name_segment + r"\s*\|", line):
        #     print(f"        [DEBUG] Line matched task pattern for '{task_name_on_line_pattern}' but not target metric '{target_metric}' or rest of pattern for '{task_col_name_for_debug}'.")
        # else:
        #     print(f"        [DEBUG] Line did NOT match task pattern for '{task_name_on_line_pattern}' for '{task_col_name_for_debug}'.")
        pass
    return None

def finalize_truthfulqa(temp_model_scores: Dict[str, Any], debug_mode: bool = False) -> Optional[str]: # Changed type hint
    mc1 = temp_model_scores.get("_truthfulqa_mc1_acc")
    mc2 = temp_model_scores.get("_truthfulqa_mc2_acc")
    if mc1 is not None and mc2 is not None:
        avg = (mc1 + mc2) / 2
        if debug_mode: print(f"  [DEBUG] Finalizing truthfulqa: mc1={mc1}, mc2={mc2}, avg={avg:.4f}")
        return f"{avg:.4f}"
    elif debug_mode:
        if mc1 is None and temp_model_scores.get("_truthfulqa_mc1_attempted", False) : print(f"  [DEBUG] Finalizing truthfulqa: mc1_acc MISSING for model {temp_model_scores.get('model_key_for_debug','?')}.")
        if mc2 is None and temp_model_scores.get("_truthfulqa_mc2_attempted", False) : print(f"  [DEBUG] Finalizing truthfulqa: mc2_acc MISSING for model {temp_model_scores.get('model_key_for_debug','?')}.")
    return None

def generate_table_from_log(file_content: str, debug_mode: bool = False) -> str:
    task_cols_ordered = [task["col_name"] for task in TASKS_CONFIG]
    results: Dict[str, Dict[str, str]] = defaultdict(lambda: {col: "n/a" for col in task_cols_ordered})
    
    temp_scores_per_model: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "ifeval_metrics_collected": [],
        "_truthfulqa_mc1_acc": None, "_truthfulqa_mc1_attempted": False,
        "_truthfulqa_mc2_acc": None, "_truthfulqa_mc2_attempted": False,
        "model_key_for_debug": "" # Store current model key for debug in finalize
    })

    current_model_key: str = ""
    lines = file_content.splitlines()
    i = 0 

    if debug_mode: print("--- Starting Log Parsing (Debug Mode) ---")

    while i < len(lines):
        line = lines[i].strip()
        current_iteration_line_index = i 

        if not line: 
            i += 1
            continue
        
        if debug_mode and current_model_key:
            print(f"\n[DEBUG] Line {i+1} (Model: '{current_model_key}'): \"{line}\"")
        elif debug_mode and not current_model_key and line.startswith("|"):
             print(f"\n[DEBUG] Line {i+1} (No Model Yet): \"{line}\"")

        previous_model_key = current_model_key
        identified_method_this_line = False
        for method_name, search_string in METHOD_DEFINITIONS.items():
            if search_string in line: 
                current_model_key = method_name
                identified_method_this_line = True
                if previous_model_key != current_model_key:
                    if previous_model_key: 
                        tqa_avg = finalize_truthfulqa(temp_scores_per_model[previous_model_key], debug_mode)
                        if tqa_avg is not None: results[previous_model_key]["truthfulqa"] = tqa_avg
                    
                    temp_scores_per_model[current_model_key] = { 
                        "ifeval_metrics_collected": [],
                        "_truthfulqa_mc1_acc": None, "_truthfulqa_mc1_attempted": False,
                        "_truthfulqa_mc2_acc": None, "_truthfulqa_mc2_attempted": False,
                        "model_key_for_debug": current_model_key
                    }
                    if debug_mode: print(f"\n[DEBUG] Model Switch/Identified: '{current_model_key}' on line {i+1}. Resetting temp scores.")
                break 
        
        if identified_method_this_line:
            i += 1 
            continue

        if not current_model_key:
            i += 1
            continue
        
        line_processed_this_iteration = False
        for task_config in TASKS_CONFIG:
            col_name = task_config["col_name"]
            task_type = task_config["type"]

            if debug_mode: print(f"  [DEBUG] Attempting Task Type: '{task_type}', Col: '{col_name}'")

            if task_type == "gsm8k_average_2_lines":
                log_name = task_config["log_name"]
                metric = task_config["target_metric"]
                if line.startswith(f"|{log_name}") or (f"| {log_name}" in line and not line.startswith("| -")): 
                    if debug_mode: print(f"    [DEBUG] GSM8K: Initial line match for '{log_name}'.")
                    val1 = extract_value_from_row(line, log_name, metric, debug_mode, col_name)
                    if val1 is not None:
                        if debug_mode: print(f"      [DEBUG] GSM8K L1 Matched: Value {val1}")
                        if current_iteration_line_index + 1 < len(lines):
                            next_line_content = lines[current_iteration_line_index + 1].strip()
                            if debug_mode: print(f"      [DEBUG] GSM8K L2 Check on line {current_iteration_line_index + 2}: \"{next_line_content}\"")
                            val2 = extract_value_from_row(next_line_content, r"\s*", metric, debug_mode, col_name + " (L2)")
                            if val2 is not None and (re.match(r"\|\s*\|", next_line_content) or "strict-match" in next_line_content):
                                if debug_mode: print(f"      [DEBUG] GSM8K L2 Matched: Value {val2}")
                                results[current_model_key][col_name] = f"{(val1 + val2) / 2:.4f}"
                                i = current_iteration_line_index + 1 
                                line_processed_this_iteration = True
                            else:
                                results[current_model_key][col_name] = f"{val1:.4f}"
                                line_processed_this_iteration = True
                                if debug_mode: print(f"      [DEBUG] GSM8K L2 NO MATCH or not suitable continuation. Using L1 value: {val1:.4f}.")
                        else: 
                            results[current_model_key][col_name] = f"{val1:.4f}"
                            line_processed_this_iteration = True
                    elif debug_mode:
                        print(f"    [DEBUG] GSM8K: Failed to extract L1 value for '{log_name}'.")
            
            elif task_type == "ifeval_average_4_metrics":
                log_name = task_config["log_name"]
                is_first_ifeval_line = line.startswith(f"|{log_name}") or (f"| {log_name}" in line and not line.startswith("| -"))

                is_continuation_ifeval_line = False
                if not is_first_ifeval_line and temp_scores_per_model[current_model_key]["ifeval_metrics_collected"]:
                    if re.match(r"\|\s*\|", line): 
                        is_continuation_ifeval_line = True
                
                is_ifeval_line = is_first_ifeval_line or is_continuation_ifeval_line

                if is_ifeval_line:
                    if debug_mode: print(f"    [DEBUG] IFEVAL: Potential line for '{log_name}'. Collected so far: {len(temp_scores_per_model[current_model_key]['ifeval_metrics_collected'])}")
                    task_name_pattern_for_extract = log_name if is_first_ifeval_line else r"\s*"
                    
                    for metric_to_find in task_config["metrics_to_average"]:
                        val = extract_value_from_row(line, task_name_pattern_for_extract, metric_to_find, debug_mode, col_name + f" ({metric_to_find})")
                        if val is not None:
                            temp_scores_per_model[current_model_key]["ifeval_metrics_collected"].append(val)
                            if debug_mode: print(f"      [DEBUG] IFEVAL: Collected value {val} for metric '{metric_to_find}'. Total: {len(temp_scores_per_model[current_model_key]['ifeval_metrics_collected'])}")
                            line_processed_this_iteration = True 
                            break 
                    
                    if len(temp_scores_per_model[current_model_key]["ifeval_metrics_collected"]) == 4:
                        avg = sum(temp_scores_per_model[current_model_key]["ifeval_metrics_collected"]) / 4
                        results[current_model_key][col_name] = f"{avg:.4f}"
                        if debug_mode: print(f"    [DEBUG] IFEVAL: SUCCESS for '{col_name}', All 4 metrics collected. Avg: {avg:.4f}")
                        temp_scores_per_model[current_model_key]["ifeval_metrics_collected"] = [] 
            
            elif task_type == "truthfulqa_average_mc1_mc2":
                mc1_log_name = task_config["mc1_log_name"]
                mc2_log_name = task_config["mc2_log_name"]
                metric = task_config["target_metric"]
                temp_scores_per_model[current_model_key]["_truthfulqa_mc1_attempted"] = True # Mark that we are looking
                temp_scores_per_model[current_model_key]["_truthfulqa_mc2_attempted"] = True


                if line.startswith(f"|{mc1_log_name}") or (f"| {mc1_log_name}" in line and not line.startswith("| -")):
                    val = extract_value_from_row(line, mc1_log_name, metric, debug_mode, col_name + " (mc1)")
                    if val is not None:
                        temp_scores_per_model[current_model_key]["_truthfulqa_mc1_acc"] = val
                        if debug_mode: print(f"    [DEBUG] TRUTHFULQA: Collected mc1_acc: {val}")
                        line_processed_this_iteration = True
                elif line.startswith(f"|{mc2_log_name}") or (f"| {mc2_log_name}" in line and not line.startswith("| -")):
                    val = extract_value_from_row(line, mc2_log_name, metric, debug_mode, col_name + " (mc2)")
                    if val is not None:
                        temp_scores_per_model[current_model_key]["_truthfulqa_mc2_acc"] = val
                        if debug_mode: print(f"    [DEBUG] TRUTHFULQA: Collected mc2_acc: {val}")
                        line_processed_this_iteration = True
            
            # New handler for arc_challenge and hellaswag
            elif task_type == "metric_on_continuation":
                log_name = task_config["log_name"]
                target_metric = task_config["target_metric_on_continuation"]
                # Check if current line is the primary line for this task
                if line.startswith(f"|{log_name}") or (f"| {log_name}" in line and not line.startswith("| -")):
                    if debug_mode: print(f"    [DEBUG] {col_name.upper()}: Found primary line for '{log_name}'. Checking next line for '{target_metric}'.")
                    if current_iteration_line_index + 1 < len(lines):
                        next_line_content = lines[current_iteration_line_index + 1].strip()
                        if debug_mode: print(f"      [DEBUG] {col_name.upper()}: Next line ({current_iteration_line_index + 2}): \"{next_line_content}\"")
                        # Check if next line is a continuation and has the target metric
                        if re.match(r"\|\s*\|", next_line_content): # Empty first cell
                            val_target = extract_value_from_row(next_line_content, r"\s*", target_metric, debug_mode, col_name)
                            if val_target is not None:
                                results[current_model_key][col_name] = f"{val_target:.4f}"
                                i = current_iteration_line_index + 1 # Consumed both current and next line
                                line_processed_this_iteration = True
                                if debug_mode: print(f"    [DEBUG] {col_name.upper()}: SUCCESS. Found '{target_metric}' on continuation. Value: {val_target:.4f}")
                            elif debug_mode: print(f"      [DEBUG] {col_name.upper()}: Target metric '{target_metric}' NOT FOUND on continuation line.")
                        elif debug_mode: print(f"      [DEBUG] {col_name.upper()}: Next line not a valid continuation for metric.")
                    elif debug_mode: print(f"    [DEBUG] {col_name.upper()}: No next line to check.")

            elif task_type in ["direct", "direct_prefer_groups"]: # Generic direct handlers
                log_name = task_config["log_name"]
                metric = task_config["target_metric"]
                if line.startswith(f"|{log_name}") or (f"| {log_name}" in line and not line.startswith("| -")):
                    val = extract_value_from_row(line, log_name, metric, debug_mode, col_name)
                    if val is not None:
                        results[current_model_key][col_name] = f"{val:.4f}" 
                        if debug_mode: print(f"    [DEBUG] DIRECT ({task_type}): SUCCESS for '{col_name}', Value: {val:.4f}")
                        line_processed_this_iteration = True
            
            if line_processed_this_iteration:
                break 
        
        if debug_mode and not line_processed_this_iteration and line.startswith("|") and not identified_method_this_line:
            print(f"  [DEBUG] Unprocessed Table-like Line {current_iteration_line_index+1} (Model: '{current_model_key}'): \"{line}\"")
        
        i += 1 
    
    if current_model_key: 
        tqa_avg = finalize_truthfulqa(temp_scores_per_model[current_model_key], debug_mode)
        if tqa_avg is not None: results[current_model_key]["truthfulqa"] = tqa_avg
        
        ifeval_collected_metrics = temp_scores_per_model[current_model_key]["ifeval_metrics_collected"]
        if results[current_model_key]["ifeval"] == "n/a" and ifeval_collected_metrics: # If not already set by full collection
             if len(ifeval_collected_metrics) == 4: # Should have been set if 4
                avg = sum(ifeval_collected_metrics) / 4
                results[current_model_key]["ifeval"] = f"{avg:.4f}"
                if debug_mode: print(f"[DEBUG] IFEVAL: Finalized for model {current_model_key} at end of log. Avg: {avg:.4f}")
             elif debug_mode: 
                print(f"[DEBUG] IFEVAL incomplete for model {current_model_key} at end of log. Collected ({len(ifeval_collected_metrics)}/4): {ifeval_collected_metrics}")


    if debug_mode: print("\n--- Finished Log Parsing ---")
    
    model_col_width = len("model") 
    if ORDERED_METHOD_NAMES: 
        max_model_name_len = 0
        for name in ORDERED_METHOD_NAMES:
            if name: max_model_name_len = max(len(name), max_model_name_len)
        model_col_width = max(max_model_name_len, model_col_width)
    model_col_width += 2

    header_parts = [f"{'model':<{model_col_width}}"]
    for task_conf_entry in TASKS_CONFIG:
        header_parts.append(f"{task_conf_entry['col_name']:<10}")
    
    output_table_lines = [" ".join(header_parts)]
    output_table_lines.append("-" * len(output_table_lines[0]))

    for method_name in ORDERED_METHOD_NAMES:
        row_parts = [f"{method_name:<{model_col_width}}"]
        for task_conf_entry in TASKS_CONFIG:
            row_parts.append(f"{results[method_name].get(task_conf_entry['col_name'], 'n/a'):<10}")
        output_table_lines.append(" ".join(row_parts))

    return "\n".join(output_table_lines)

if __name__ == '__main__':
    log_file_to_parse = "UNA_Qwen_merge123.txt" 
    enable_debug = True 
    
    try:
        with open(log_file_to_parse, "r", encoding="utf-8") as f:
            file_content = f.read()
        
        print(f"--- Parsing log file: {log_file_to_parse} ---")
        if enable_debug:
            print("--- DEBUG MODE IS ON ---")
            
        table_output = generate_table_from_log(file_content, debug_mode=enable_debug)
        
        print("\n--- Generated Table ---")
        print(table_output)

    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"The file '{log_file_to_parse}' was not found.")
    except Exception as e:
        print(f"\n--- ERROR ---")
        import traceback
        print(f"An unexpected error occurred: {e}")
        print("Traceback:")
        traceback.print_exc()