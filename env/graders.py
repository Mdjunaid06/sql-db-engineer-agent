import re
import json
import os
from functools import lru_cache
from env.models import Action, DifficultyLevel
from env.tasks import task_manager

# ─────────────────────────────────────────────
#  HELPERS (unchanged from Round 1)
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())

def _safe_get(payload: dict, key: str, default=None):
    if not isinstance(payload, dict):
        return default
    return payload.get(key, default)

def _score_explanation(explanation: str) -> float:
    if not explanation or not isinstance(explanation, str):
        return 0.0
    explanation = explanation.strip()
    if len(explanation) < 10:  return 0.0
    if len(explanation) < 30:  return 0.05
    if len(explanation) < 80:  return 0.10
    return 0.15

def _score_confidence(confidence) -> float:
    try:
        c = float(confidence)
        if 0.0 <= c <= 1.0:
            return 0.05
    except (TypeError, ValueError):
        pass
    return 0.0

def _query_similarity(submitted: str, expected: str) -> float:
    s = _normalize(submitted)
    e = _normalize(expected)
    if s == e:
        return 1.0
    s_tokens = set(s.split())
    e_tokens = set(e.split())
    if not e_tokens:
        return 0.0
    overlap = len(s_tokens & e_tokens) / len(e_tokens)
    critical_keywords = _extract_critical_keywords(e)
    critical_found = sum(1 for kw in critical_keywords if kw in s)
    critical_score = critical_found / len(critical_keywords) if critical_keywords else 0.0
    return round((overlap * 0.4) + (critical_score * 0.6), 4)

def _extract_critical_keywords(query: str) -> list[str]:
    keywords = [
        "left join", "inner join", "right join",
        "group by", "order by", "having",
        "partition by", "coalesce", "distinct",
        "where", "on", "and", "or", "not",
        "count", "sum", "avg", "max", "min",
        "select", "from", "join"
    ]
    found = []
    q = query.lower()
    for kw in keywords:
        if kw in q:
            found.append(kw)
    return found

def _score_error_type(submitted_type: str, expected_type: str) -> float:
    if not submitted_type:
        return 0.0
    s = submitted_type.strip().lower()
    e = expected_type.strip().lower()
    if s == e:
        return 0.10
    related = {
        "performance": ["optimization", "slow", "index", "scan"],
        "logic":       ["semantic", "incorrect", "wrong"],
        "syntax":      ["parse", "grammar", "token"]
    }
    for canonical, aliases in related.items():
        if e == canonical and any(alias in s for alias in aliases):
            return 0.05
    return 0.0

def _score_error_location(submitted_location: str, expected_location: str) -> float:
    if not submitted_location or not expected_location:
        return 0.0
    s = submitted_location.strip().lower()
    e = expected_location.strip().lower()
    if s == e:
        return 0.15
    e_words = set(e.split())
    s_words = set(s.split())
    overlap = len(e_words & s_words) / len(e_words) if e_words else 0.0
    return round(overlap * 0.10, 4)


# ─────────────────────────────────────────────
#  ROUND 2 — SCENARIO LOADER
# ─────────────────────────────────────────────

# Cache for loaded scenarios — avoids re-reading JSON on every grader call
_scenario_cache: dict[str, dict] = {}
_cache_loaded = False

def _load_all_scenarios():
    """Load all Round 2 scenario JSONs into cache once at startup."""
    global _cache_loaded
    if _cache_loaded:
        return
    for fname in [
        "dataset/easy_scenarios.json",
        "dataset/medium_scenarios.json",
        "dataset/hard_scenarios.json",
    ]:
        try:
            with open(fname) as f:
                for s in json.load(f):
                    _scenario_cache[s["id"]] = s
        except FileNotFoundError:
            pass
        except Exception:
            pass
    _cache_loaded = True

def _get_scenario(task_id: str) -> dict | None:
    """Get a Round 2 scenario by ID. Returns None if not found."""
    _load_all_scenarios()
    return _scenario_cache.get(task_id)

def _is_scenario_task(task_id: str) -> bool:
    """
    Round 2 scenario IDs have format: easy_s001, medium_s002, hard_s003.
    Round 1 task IDs have format:     easy_001, medium_001, hard_001.
    Distinction: Round 2 has 's' before the number.
    """
    if not task_id:
        return False
    parts = task_id.split("_")
    # easy_s001 → ["easy", "s001"]  |  easy_001 → ["easy", "001"]
    return len(parts) >= 2 and parts[-1].startswith("s")


# ─────────────────────────────────────────────
#  ROUND 2 — DB ACTION GRADER
# ─────────────────────────────────────────────

def grade_db_action(action: Action, task_id: str) -> tuple[float, dict, str]:
    """
    Grades a Round 2 database engineering action.

    Scoring philosophy:
      - Does the action target valid tables/queries in THIS scenario?
      - For create_index: does it match the missing_index_hints?
      - For rewrite_query: is the SQL structurally better?
      - For submit_report: was a meaningful summary provided?
      - All terminal/non-terminal actions get meaningful differentiation.

    Returns (score 0.001-0.999, breakdown dict, feedback string).
    DETERMINISTIC: same input → same score always.
    """
    if action is None or action.payload is None:
        return 0.001, {"error": "null_action"}, "No action provided."

    scenario = _get_scenario(task_id)
    if scenario is None:
        # Unknown scenario — give a small score for valid action structure
        return 0.10, {"error": "scenario_not_found"}, f"Scenario '{task_id}' not in dataset."

    action_type = (
        action.action_type.value
        if hasattr(action.action_type, "value")
        else str(action.action_type)
    )
    payload = action.payload or {}

    valid_tables  = {t["name"] for t in scenario.get("tables", [])}
    valid_queries = {q["id"]   for q in scenario.get("slow_queries", [])}
    hints         = scenario.get("missing_index_hints", [])
    large_tables  = {
        t["name"] for t in scenario.get("tables", [])
        if t.get("rows", 0) > 100_000
    }

    score     = 0.0
    breakdown = {}
    feedback  = []

    # ── inspect_query ─────────────────────────────────────────────
    if action_type == "inspect_query":
        qid = str(payload.get("query_id", "")).strip()
        if qid in valid_queries:
            score = 0.40
            feedback.append(f"Inspecting valid slow query '{qid}'.")
            breakdown["query_valid"] = 0.40
        elif qid:
            score = 0.10
            feedback.append(f"Query '{qid}' not in scenario slow_queries.")
            breakdown["query_valid"] = 0.10
        else:
            score = 0.05
            feedback.append("No query_id provided in payload.")
            breakdown["query_valid"] = 0.05

    # ── analyze_indexes ───────────────────────────────────────────
    elif action_type == "analyze_indexes":
        table = str(payload.get("table", "")).strip()
        if table in valid_tables:
            score = 0.35
            feedback.append(f"Analyzing indexes on valid table '{table}'.")
            breakdown["table_valid"] = 0.35
        elif table:
            score = 0.08
            feedback.append(f"Table '{table}' not in scenario.")
            breakdown["table_valid"] = 0.08
        else:
            score = 0.05
            feedback.append("No table provided in payload.")
            breakdown["table_valid"] = 0.05

    # ── create_index ──────────────────────────────────────────────
    elif action_type == "create_index":
        table = str(payload.get("table", "")).strip()
        cols  = payload.get("columns", [])

        # Normalise columns: accept list or comma-string
        if isinstance(cols, str):
            cols = [c.strip() for c in cols.split(",") if c.strip()]
        elif not isinstance(cols, list):
            cols = []

        if table not in valid_tables:
            score = 0.05
            feedback.append(f"Table '{table}' not in scenario.")
            breakdown["table_valid"] = 0.05
        elif not cols:
            score = 0.10
            feedback.append("Table valid but no columns specified.")
            breakdown["columns_valid"] = 0.10
        else:
            # Score against missing_index_hints
            best_match = 0.0
            for hint in hints:
                if hint.get("table") == table:
                    hint_cols      = set(hint.get("columns", []))
                    submitted_cols = set(cols)
                    if hint_cols and submitted_cols:
                        overlap    = len(hint_cols & submitted_cols) / len(hint_cols)
                        best_match = max(best_match, overlap)

            if best_match >= 1.0:
                score = 0.85
                feedback.append(
                    f"Perfect index on {table}({', '.join(cols)}) — "
                    "matches missing_index_hints exactly."
                )
                breakdown["index_match"] = 0.85
            elif best_match >= 0.5:
                score = 0.55
                feedback.append(
                    f"Partial index match on {table} ({int(best_match*100)}% column overlap)."
                )
                breakdown["index_match"] = 0.55
            elif hints:
                # Table valid, hints exist but columns don't match
                score = 0.20
                feedback.append(
                    f"Table '{table}' is valid but columns {cols} don't match any hint."
                )
                breakdown["index_match"] = 0.20
            else:
                # No hints in scenario — any reasonable index gets credit
                score = 0.35
                feedback.append(f"Index on {table}({', '.join(cols)}) — no hints to verify against.")
                breakdown["index_match"] = 0.35

    # ── rewrite_query ─────────────────────────────────────────────
    elif action_type == "rewrite_query":
        qid     = str(payload.get("query_id", "")).strip()
        new_sql = str(payload.get("new_sql", "")).strip()

        base = 0.0
        if qid in valid_queries:
            base = 0.20
            feedback.append(f"Rewriting valid query '{qid}'.")
        elif qid:
            base = 0.05
            feedback.append(f"Query '{qid}' not in scenario.")
        else:
            base = 0.03
            feedback.append("No query_id provided.")

        sql_bonus = 0.0
        if new_sql and len(new_sql) > 15:
            lower = new_sql.lower()
            if "select *" not in lower:                         sql_bonus += 0.10
            if "join" in lower and "where" in lower:           sql_bonus += 0.10
            if "index" in lower or "force index" in lower:     sql_bonus += 0.08
            if "left join" in lower or "inner join" in lower:  sql_bonus += 0.05
            feedback.append("SQL provided and has structure.")
        else:
            feedback.append("No new_sql provided.")

        score = min(base + sql_bonus, 0.65)
        breakdown["rewrite_quality"] = round(score, 4)

    # ── partition_table ───────────────────────────────────────────
    elif action_type == "partition_table":
        table = str(payload.get("table", "")).strip()
        col   = str(payload.get("partition_column", "")).strip()

        if table in large_tables:
            score = 0.65
            feedback.append(f"Correct — '{table}' is large and benefits from partitioning.")
            breakdown["partition_benefit"] = 0.65
            if col:
                score = min(score + 0.10, 0.75)
                feedback.append(f"Partition column '{col}' specified.")
        elif table in valid_tables:
            score = 0.20
            feedback.append(f"Table '{table}' exists but may not need partitioning (check row count).")
            breakdown["partition_benefit"] = 0.20
        else:
            score = 0.05
            feedback.append(f"Table '{table}' not in scenario.")
            breakdown["partition_benefit"] = 0.05

    # ── analyze_statistics ────────────────────────────────────────
    elif action_type == "analyze_statistics":
        table = str(payload.get("table", "")).strip()
        if table in valid_tables:
            score = 0.30
            feedback.append(f"Analyzing statistics on valid table '{table}'.")
            breakdown["table_valid"] = 0.30
        else:
            score = 0.08
            feedback.append(f"Table '{table}' not in scenario.")
            breakdown["table_valid"] = 0.08

    # ── drop_index ────────────────────────────────────────────────
    elif action_type == "drop_index":
        table = str(payload.get("table", "")).strip()
        idx   = str(payload.get("index_name", "")).strip()
        if table in valid_tables and idx and idx != "PRIMARY":
            score = 0.25
            feedback.append(f"Dropping index '{idx}' on '{table}'.")
        elif idx == "PRIMARY":
            score = 0.001
            feedback.append("Cannot drop PRIMARY index.")
        else:
            score = 0.05
            feedback.append("Invalid table or index_name.")
        breakdown["drop_validity"] = score

    # ── add_column ────────────────────────────────────────────────
    elif action_type == "add_column":
        table = str(payload.get("table", "")).strip()
        col   = str(payload.get("column_name", "")).strip()
        if table in valid_tables and col:
            score = 0.25
            feedback.append(f"Adding column '{col}' to '{table}'.")
        else:
            score = 0.05
            feedback.append("Missing table or column_name.")
        breakdown["add_column"] = score

    # ── request_hint ──────────────────────────────────────────────
    elif action_type == "request_hint":
        # Hint requests are penalised in the environment reward but still valid actions
        score = 0.10
        feedback.append("Hint requested — valid but penalised in full episode reward.")
        breakdown["hint_penalty_note"] = 0.10

    # ── submit_report ─────────────────────────────────────────────
    elif action_type == "submit_report":
        summary = str(payload.get("summary", "")).strip()
        # Score on summary quality — episode score handled separately by /grader
        if len(summary) >= 100:
            score = 0.50
            feedback.append("Detailed report submitted.")
        elif len(summary) >= 30:
            score = 0.30
            feedback.append("Brief report submitted.")
        elif summary:
            score = 0.15
            feedback.append("Minimal report submitted.")
        else:
            score = 0.05
            feedback.append("Empty report — include a summary of actions taken.")
        breakdown["report_quality"] = score

    # ── unknown action ────────────────────────────────────────────
    else:
        score = 0.05
        feedback.append(f"Unknown action_type '{action_type}'.")
        breakdown["unknown_action"] = 0.05

    final_score = round(max(0.001, min(0.999, score)), 4)
    return final_score, breakdown, " ".join(feedback) or "Action processed."


# ─────────────────────────────────────────────
#  ROUND 1 GRADERS (unchanged)
# ─────────────────────────────────────────────

def grade_easy(action: Action, ground_truth: dict) -> tuple[float, dict, str]:
    if action is None or action.payload is None:
        return 0.001, {"error": "null_action"}, "No action provided."

    payload   = action.payload
    score     = 0.0
    breakdown = {}
    feedback_parts = []

    submitted_query = _safe_get(payload, "fixed_query", "") or _safe_get(payload, "optimized_query", "")
    expected_query  = ground_truth.get("fixed_query", "")
    similarity      = _query_similarity(submitted_query, expected_query)

    if similarity >= 1.0:
        fix_score = 0.50; feedback_parts.append("Correct fix applied.")
    elif similarity >= 0.75:
        fix_score = 0.30; feedback_parts.append("Fix is mostly correct but has minor differences.")
    elif similarity >= 0.50:
        fix_score = 0.15; feedback_parts.append("Fix is partially correct.")
    else:
        fix_score = 0.0;  feedback_parts.append("Fix is incorrect or not provided.")

    score += fix_score
    breakdown["fix_correctness"] = round(fix_score, 4)

    submitted_location = _safe_get(payload, "error_location", "")
    expected_location  = ground_truth.get("error_location", "")
    loc_score          = _score_error_location(str(submitted_location), expected_location)
    score             += loc_score
    breakdown["error_location"] = round(loc_score, 4)
    if loc_score > 0: feedback_parts.append("Correctly identified error location.")

    submitted_type = _safe_get(payload, "error_type", "")
    expected_type  = ground_truth.get("error_type", "syntax")
    type_score     = _score_error_type(str(submitted_type), expected_type)
    score         += type_score
    breakdown["error_type"] = round(type_score, 4)
    if type_score > 0: feedback_parts.append("Correctly identified error type.")

    explanation = _safe_get(payload, "explanation", "") or _safe_get(payload, "change_made", "")
    expl_score  = _score_explanation(str(explanation) if explanation else "")
    score      += expl_score
    breakdown["explanation"] = round(expl_score, 4)
    if expl_score > 0: feedback_parts.append("Explanation provided.")

    confidence = _safe_get(payload, "confidence", None)
    conf_score = _score_confidence(confidence)
    score     += conf_score
    breakdown["confidence"] = round(conf_score, 4)

    final_score = round(max(0.001, min(0.999, score)), 4)
    feedback    = " ".join(feedback_parts) if feedback_parts else "No valid response provided."
    return final_score, breakdown, feedback


def grade_medium(action: Action, ground_truth: dict) -> tuple[float, dict, str]:
    if action is None or action.payload is None:
        return 0.001, {"error": "null_action"}, "No action provided."

    payload        = action.payload
    score          = 0.0
    breakdown      = {}
    feedback_parts = []

    submitted_query = _safe_get(payload, "fixed_query", "") or _safe_get(payload, "optimized_query", "")
    expected_query  = ground_truth.get("fixed_query", "")
    similarity      = _query_similarity(submitted_query, expected_query)

    if similarity >= 1.0:
        fix_score = 0.40; feedback_parts.append("Correct fix applied.")
    elif similarity >= 0.80:
        fix_score = 0.28; feedback_parts.append("Fix is mostly correct.")
    elif similarity >= 0.60:
        fix_score = 0.16; feedback_parts.append("Fix is partially correct.")
    elif similarity >= 0.40:
        fix_score = 0.08; feedback_parts.append("Fix shows some understanding.")
    else:
        fix_score = 0.0;  feedback_parts.append("Fix is incorrect or missing.")

    score += fix_score
    breakdown["fix_correctness"] = round(fix_score, 4)

    explanation = str(_safe_get(payload, "explanation", "") or _safe_get(payload, "change_made", "") or "")
    error_type  = ground_truth.get("error_type", "logic")

    logic_keywords = {
        "logic":       ["join", "left join", "inner join", "having", "where", "group by",
                        "aggregate", "subquery", "correlation", "distinct", "count"],
        "performance": ["index", "scan", "n+1", "correlated", "cartesian", "window"]
    }
    keywords_to_check = logic_keywords.get(error_type, logic_keywords["logic"])
    expl_lower        = explanation.lower()
    keyword_hits      = sum(1 for kw in keywords_to_check if kw in expl_lower)
    logic_score       = min(keyword_hits * 0.05, 0.20)
    score            += logic_score
    breakdown["logic_flaw_identification"] = round(logic_score, 4)
    if logic_score > 0: feedback_parts.append("Shows understanding of the logic flaw.")

    submitted_location = _safe_get(payload, "error_location", "")
    expected_location  = ground_truth.get("error_location", "")
    loc_score          = _score_error_location(str(submitted_location), expected_location)
    score             += loc_score
    breakdown["error_location"] = round(loc_score, 4)

    expl_score = _score_explanation(explanation)
    score     += expl_score
    breakdown["explanation"] = round(expl_score, 4)

    confidence = _safe_get(payload, "confidence", None)
    conf_score = _score_confidence(confidence)
    score     += conf_score
    breakdown["confidence"] = round(conf_score, 4)

    impact = str(_safe_get(payload, "impact", "") or "")
    if len(impact.strip()) > 20:
        score += 0.05
        breakdown["impact_analysis"] = 0.05
        feedback_parts.append("Impact analysis provided.")
    else:
        breakdown["impact_analysis"] = 0.0

    final_score = round(max(0.001, min(0.999, score)), 4)
    feedback    = " ".join(feedback_parts) if feedback_parts else "No valid response provided."
    return final_score, breakdown, feedback


def grade_hard(action: Action, ground_truth: dict) -> tuple[float, dict, str]:
    if action is None or action.payload is None:
        return 0.001, {"error": "null_action"}, "No action provided."

    payload        = action.payload
    score          = 0.0
    breakdown      = {}
    feedback_parts = []

    submitted_query = (
        _safe_get(payload, "optimized_query", "")
        or _safe_get(payload, "fixed_query", "")
        or ""
    )
    expected_query = ground_truth.get("fixed_query", "")
    similarity     = _query_similarity(submitted_query, expected_query)

    if similarity >= 1.0:
        fix_score = 0.30; feedback_parts.append("Perfectly optimized query.")
    elif similarity >= 0.85:
        fix_score = 0.22; feedback_parts.append("Query is mostly correct.")
    elif similarity >= 0.65:
        fix_score = 0.14; feedback_parts.append("Query shows correct approach but incomplete.")
    elif similarity >= 0.40:
        fix_score = 0.07; feedback_parts.append("Query partially addresses the issue.")
    else:
        fix_score = 0.0;  feedback_parts.append("Query does not address the performance issue.")

    score += fix_score
    breakdown["query_correctness"] = round(fix_score, 4)

    explanation   = str(_safe_get(payload, "explanation", "") or _safe_get(payload, "change_made", "") or "")
    optimization  = str(_safe_get(payload, "optimization_type", "") or "")
    combined_text = (explanation + " " + optimization).lower()
    perf_issue    = ground_truth.get("performance_issue", {})
    issue_type    = perf_issue.get("type", "").lower()

    performance_concept_map = {
        "n+1":               ["n+1", "correlated subquery", "subquery per row", "multiple queries", "join instead"],
        "full table scan":   ["full table scan", "index not used", "function on column", "sargable", "range scan", "seek"],
        "cartesian product": ["cartesian", "cross join", "missing join condition", "implicit join", "comma join"],
        "select *":          ["select *", "over-fetch", "covering index", "column projection", "unnecessary columns"],
        "window function":   ["window function", "partition by", "row_number", "subquery filter", "where clause window"]
    }

    concept_score = 0.0
    for concept, keywords in performance_concept_map.items():
        if any(concept_part in issue_type for concept_part in concept.split()):
            hits = sum(1 for kw in keywords if kw in combined_text)
            concept_score = min(hits * 0.06, 0.30)
            break

    score += concept_score
    breakdown["performance_concept"] = round(concept_score, 4)
    if concept_score > 0: feedback_parts.append("Demonstrates understanding of the performance issue.")

    expl_score = _score_explanation(explanation)
    if len(explanation.strip()) > 150:
        expl_score = min(expl_score + 0.05, 0.15)
    score += expl_score
    breakdown["explanation_depth"] = round(expl_score, 4)

    root_cause = str(_safe_get(payload, "root_cause", "") or "")
    if len(root_cause.strip()) > 30:
        score += 0.10
        breakdown["root_cause_analysis"] = 0.10
        feedback_parts.append("Root cause analysis provided.")
    else:
        breakdown["root_cause_analysis"] = 0.0

    improvement = str(_safe_get(payload, "expected_improvement", "") or "")
    if len(improvement.strip()) > 20:
        score += 0.10
        breakdown["expected_improvement"] = 0.10
        feedback_parts.append("Performance improvement estimate provided.")
    else:
        breakdown["expected_improvement"] = 0.0

    confidence = _safe_get(payload, "confidence", None)
    conf_score = _score_confidence(confidence)
    score     += conf_score
    breakdown["confidence"] = round(conf_score, 4)

    final_score = round(max(0.001, min(0.999, score)), 4)
    feedback    = " ".join(feedback_parts) if feedback_parts else "Performance issue not identified."
    return final_score, breakdown, feedback


# ─────────────────────────────────────────────
#  MAIN GRADER DISPATCHER
# ─────────────────────────────────────────────

def grade(action: Action, task_id: str) -> tuple[float, dict, str]:
    """
    Main grader entry point.

    ROUTING:
      Round 2 scenario IDs (easy_s001, medium_s002, hard_s003)
        → grade_db_action()   ← NEW: scores DB engineering actions

      Round 1 task IDs (easy_001, medium_001, hard_001)
        → grade_easy/medium/hard()  ← unchanged

    ALWAYS returns (float, dict, str). NEVER crashes.
    Score always strictly between 0.001 and 0.999.
    """
    if action is None:
        return 0.001, {"error": "null_action"}, "No action provided."

    # ── Round 2: DB engineering scenario ─────────────────────────
    if _is_scenario_task(task_id):
        return grade_db_action(action, task_id)

    # ── Round 1: SQL debugging task ───────────────────────────────
    ground_truth = task_manager.get_ground_truth(task_id)
    if ground_truth is None:
        return 0.001, {"error": "unknown_task"}, f"Task '{task_id}' not found."

    difficulty = task_id.split("_")[0]

    try:
        if difficulty == "easy":
            return grade_easy(action, ground_truth)
        elif difficulty == "medium":
            return grade_medium(action, ground_truth)
        elif difficulty == "hard":
            return grade_hard(action, ground_truth)
        else:
            return 0.001, {"error": "unknown_difficulty"}, f"Unknown difficulty: {difficulty}"
    except Exception as e:
        return 0.001, {"error": str(e)}, f"Grader error: {str(e)}"
