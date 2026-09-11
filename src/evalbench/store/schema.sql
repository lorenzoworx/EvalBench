PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    suite_name TEXT NOT NULL,
    suite_version TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'interrupted')
    ),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    completed_cases INTEGER NOT NULL DEFAULT 0 CHECK (completed_cases >= 0),
    total_cases INTEGER NOT NULL CHECK (total_cases >= 0),
    accuracy REAL CHECK (accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 1)),
    ci_low REAL CHECK (ci_low IS NULL OR (ci_low >= 0 AND ci_low <= 1)),
    ci_high REAL CHECK (ci_high IS NULL OR (ci_high >= 0 AND ci_high <= 1)),
    error TEXT
);

CREATE TABLE IF NOT EXISTS case_results (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    case_id TEXT NOT NULL,
    category TEXT NOT NULL,
    case_snapshot TEXT NOT NULL,
    generation TEXT NOT NULL,
    passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
    PRIMARY KEY (run_id, case_id)
);

CREATE TABLE IF NOT EXISTS judgments (
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    grader_index INTEGER NOT NULL CHECK (grader_index >= 0),
    grader_type TEXT NOT NULL,
    passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
    score REAL NOT NULL,
    rationale TEXT NOT NULL,
    is_primary INTEGER NOT NULL CHECK (is_primary IN (0, 1)),
    PRIMARY KEY (run_id, case_id, grader_index),
    FOREIGN KEY (run_id, case_id)
        REFERENCES case_results(run_id, case_id) ON DELETE CASCADE
);
