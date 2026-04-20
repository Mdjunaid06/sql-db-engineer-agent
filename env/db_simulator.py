"""
env/db_simulator.py — SQL Database Engineer Agent
Simulates a production database responding to optimization actions.
Core mechanism: index coverage reduces query execution time by up to 85-90%.
"""

import math
import random
from typing import Optional


class DatabaseSimulator:
    """
    Simulates a production database that degrades over time.
    The agent applies optimization actions and sees performance scores change.

    Performance score: 0-100 (100 = all queries running at target speed).
    The agent's goal: get performance_score >= target_score.
    """

    def __init__(self, scenario: dict):
        self.scenario     = scenario
        self.tables       = {t["name"]: dict(t) for t in scenario["tables"]}
        self.queries      = [dict(q) for q in scenario["slow_queries"]]
        self.indexes      = {
            name: list(t.get("indexes", ["PRIMARY"]))
            for name, t in self.tables.items()
        }
        self.stats_fresh  = {name: False for name in self.tables}
        self.partitioned  = {name: False for name in self.tables}
        self.baseline     = self._compute_score()
        self.history      = [self.baseline]
        self.best_score   = self.baseline
        self.target_score = scenario.get("target_score", 85.0)

    # ─────────────────────────────────────────────
    #  PUBLIC ACTIONS
    # ─────────────────────────────────────────────

    def apply_action(self, action_type: str, payload: dict) -> dict:
        """
        Apply an optimization action to the database.
        Returns delta showing performance change.
        """
        old_score = self._compute_score()
        affected  = []

        if action_type == "create_index":
            table = payload.get("table", "")
            cols  = payload.get("columns", [])
            if isinstance(cols, str):
                cols = [c.strip() for c in cols.split(",")]
            idx_name = "idx_" + "_".join(cols)
            if table in self.indexes and idx_name not in self.indexes[table]:
                self.indexes[table].append(idx_name)
                affected = self._queries_benefiting_from_index(table, cols)
            else:
                # Duplicate index — no benefit
                return {
                    "old_score": old_score, "new_score": old_score,
                    "delta": 0.0, "affected_queries": [],
                    "improved": False, "message": "Index already exists or table not found."
                }

        elif action_type == "rewrite_query":
            qid     = payload.get("query_id", "")
            new_sql = payload.get("new_sql", "")
            for q in self.queries:
                if q["id"] == qid:
                    improvement = self._estimate_rewrite(new_sql, q)
                    q["avg_ms"]  = max(10, int(q["avg_ms"] * (1 - improvement)))
                    affected     = [qid]
                    break

        elif action_type == "partition_table":
            table = payload.get("table", "")
            if table in self.tables and not self.partitioned.get(table):
                self.partitioned[table] = True
                affected = [q["id"] for q in self.queries if table in q.get("sql", "")]

        elif action_type == "analyze_statistics":
            table = payload.get("table", "")
            if table in self.tables:
                self.stats_fresh[table] = True
                affected = [q["id"] for q in self.queries if table in q.get("sql", "")]

        elif action_type == "drop_index":
            table    = payload.get("table", "")
            idx_name = payload.get("index_name", "")
            if idx_name in self.indexes.get(table, []) and idx_name != "PRIMARY":
                self.indexes[table].remove(idx_name)

        elif action_type == "add_column":
            table   = payload.get("table", "")
            col     = payload.get("column", "")
            purpose = payload.get("purpose", "")
            if table in self.tables:
                if "extra_columns" not in self.tables[table]:
                    self.tables[table]["extra_columns"] = []
                self.tables[table]["extra_columns"].append(col)
                # Denormalization can help JOINy queries
                affected = [
                    q["id"] for q in self.queries
                    if "join" in q.get("sql", "").lower() and table in q.get("sql", "")
                ]

        new_score = self._compute_score()
        self.history.append(new_score)
        if new_score > self.best_score:
            self.best_score = new_score

        return {
            "old_score":        round(old_score, 2),
            "new_score":        round(new_score, 2),
            "delta":            round(new_score - old_score, 2),
            "affected_queries": affected,
            "improved":         new_score > old_score,
        }

    def inspect_query(self, query_id: str) -> dict:
        """
        EXPLAIN a slow query — reveals scan type, rows examined, cost.
        This is the agent's primary investigation tool.
        """
        for q in self.queries:
            if q["id"] == query_id:
                has_index    = self._check_query_index_coverage(q) > 0.1
                is_partition = self.partitioned.get(q.get("main_table", ""), False)
                rows_examined = 50 if has_index else q.get("rows_examined",
                    self.tables.get(q.get("main_table", ""), {}).get("rows", 50000))

                return {
                    "query_id":         query_id,
                    "sql":              q["sql"],
                    "avg_ms":           q["avg_ms"],
                    "scan_type":        "INDEX RANGE SCAN" if has_index else "FULL TABLE SCAN",
                    "rows_examined":    rows_examined,
                    "partitioned":      is_partition,
                    "optimization_hint": (
                        "Query is using index efficiently."
                        if has_index
                        else "No index covering WHERE columns. Consider adding composite index."
                    ),
                    "main_table":       q.get("main_table", "unknown"),
                }
        return {"error": f"Query '{query_id}' not found"}

    def analyze_indexes(self, table: str) -> dict:
        """
        Show all indexes on a table + usage stats + missing index hints.
        """
        if table not in self.tables:
            return {"error": f"Table '{table}' not found"}

        existing   = self.indexes.get(table, [])
        hints      = [
            h for h in self.scenario.get("missing_index_hints", [])
            if h.get("table") == table
        ]
        used_by    = []
        for q in self.queries:
            cov = self._check_query_index_coverage(q)
            if table in q.get("sql", "") and cov > 0.1:
                used_by.append(q["id"])

        return {
            "table":           table,
            "row_count":       self.tables[table].get("rows", 0),
            "existing_indexes": existing,
            "indexes_used_by": used_by,
            "missing_hints":   hints,
            "stats_fresh":     self.stats_fresh.get(table, False),
            "partitioned":     self.partitioned.get(table, False),
            "size_mb":         self.tables[table].get("size_mb", 0),
        }

    # ─────────────────────────────────────────────
    #  STATE
    # ─────────────────────────────────────────────

    def get_current_state(self) -> dict:
        """Returns the full current DB state for the Observation."""
        return {
            "performance_score": round(self._compute_score(), 2),
            "baseline_score":    round(self.baseline, 2),
            "target_score":      self.target_score,
            "tables":            list(self.tables.values()),
            "slow_queries":      self.queries,
            "indexes":           self.indexes,
            "history":           self.history,
            "best_score":        round(self.best_score, 2),
        }

    def get_performance_score(self) -> float:
        return round(self._compute_score(), 2)

    def is_target_reached(self) -> bool:
        return self._compute_score() >= self.target_score

    # ─────────────────────────────────────────────
    #  INTERNAL SCORING ENGINE
    # ─────────────────────────────────────────────

    def _compute_score(self) -> float:
        """
        Core scoring: calculates performance score 0-100.
        Higher = better. Based on how fast queries run given current indexes.
        """
        if not self.queries:
            return 0.0

        scores = []
        for q in self.queries:
            table       = q.get("main_table", "")
            coverage    = self._check_query_index_coverage(q)
            part_bonus  = 0.30 if self.partitioned.get(table, False) else 0.0
            stats_bonus = 0.05 if self.stats_fresh.get(table, False) else 0.0
            total_reduction = min(coverage * 0.85 + part_bonus + stats_bonus, 0.97)
            effective_ms    = q["avg_ms"] * (1 - total_reduction)
            # Score formula: 100ms = score 99, 1000ms = score 90, 8500ms = ~14
            score = max(0.0, 100.0 - (effective_ms / 100.0))
            scores.append(score)

        return round(sum(scores) / len(scores), 2)

    def _check_query_index_coverage(self, query: dict) -> float:
        """
        Returns 0.0-1.0 representing how well indexes cover this query's WHERE clause.
        0.0 = full table scan, 1.0 = perfect index coverage.
        """
        sql = query.get("sql", "").lower()
        for table, indexes in self.indexes.items():
            if table not in sql:
                continue
            for idx in indexes:
                if idx == "PRIMARY":
                    # Primary key only helps if query filters by primary key
                    if "where id=" in sql or "where id =" in sql:
                        return 0.95
                    continue
                # Extract columns from index name (idx_col1_col2)
                cols = idx.replace("idx_", "").split("_")
                matches = sum(1 for c in cols if c in sql)
                if matches >= 2:
                    return 0.90  # Composite index — excellent coverage
                if matches == 1:
                    return 0.60  # Single column — partial coverage
        return 0.0

    def _queries_benefiting_from_index(self, table: str, cols: list) -> list:
        """Returns query IDs that would benefit from an index on given table/columns."""
        benefiting = []
        for q in self.queries:
            sql = q.get("sql", "").lower()
            if table in sql and any(c.lower() in sql for c in cols):
                benefiting.append(q["id"])
        return benefiting

    def _estimate_rewrite(self, new_sql: str, query: dict) -> float:
        """
        Estimates improvement factor from a query rewrite (0.0 to 0.70).
        Checks for common optimization patterns.
        """
        new_lower = new_sql.lower()
        old_lower = query.get("sql", "").lower()
        improvement = 0.0

        # Remove SELECT * → specific columns
        if "select *" not in new_lower and "select *" in old_lower:
            improvement += 0.20

        # Add LIMIT clause
        if "limit " in new_lower and "limit " not in old_lower:
            improvement += 0.15

        # Use EXISTS instead of IN subquery
        if "exists" in new_lower and "in (select" in old_lower:
            improvement += 0.25

        # Use INNER JOIN instead of implicit cross join
        if "inner join" in new_lower and "," in old_lower and "join" not in old_lower:
            improvement += 0.30

        # Add WHERE clause that was missing
        if "where" in new_lower and "where" not in old_lower:
            improvement += 0.35

        # Use COALESCE / ISNULL
        if "coalesce" in new_lower:
            improvement += 0.05

        return min(improvement, 0.70)
