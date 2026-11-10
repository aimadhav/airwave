CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'personal'
);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS speakers (
    speaker_id TEXT PRIMARY KEY,
    display_name TEXT,
    voice_embedding TEXT,
    id_confidence REAL NOT NULL DEFAULT 0.5
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
    project_id TEXT REFERENCES projects(project_id),
    audio_path TEXT,
    recorded_at TEXT NOT NULL,
    title TEXT
);

CREATE TABLE IF NOT EXISTS utterances (
    utterance_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    speaker_id TEXT NOT NULL,
    speaker_name TEXT NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    raw_text TEXT NOT NULL,
    identity_evidence TEXT
);

CREATE TABLE IF NOT EXISTS content_units (
    content_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
    project_id TEXT REFERENCES projects(project_id),
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    decay_rate REAL,
    last_reaffirmed_at TEXT NOT NULL,
    source_utterance_ids TEXT NOT NULL DEFAULT '[]',
    source_content_ids TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id TEXT PRIMARY KEY,
    from_node TEXT NOT NULL REFERENCES content_units(content_id),
    to_node TEXT NOT NULL REFERENCES content_units(content_id),
    relation_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    reasoning TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS change_log (
    log_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    field TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    confidence REAL,
    reasoning TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_summaries (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
    title TEXT NOT NULL,
    overview TEXT NOT NULL,
    discussion_highlights TEXT NOT NULL DEFAULT '[]',
    decisions_summary TEXT NOT NULL DEFAULT '[]',
    action_items_summary TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS project_summaries (
    project_id TEXT PRIMARY KEY REFERENCES projects(project_id),
    title TEXT NOT NULL,
    status_overview TEXT NOT NULL,
    latest_milestones TEXT NOT NULL DEFAULT '[]',
    open_action_items TEXT NOT NULL DEFAULT '[]',
    session_ids TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_content_units_workspace ON content_units(workspace_id);
CREATE INDEX IF NOT EXISTS idx_content_units_status ON content_units(status);
CREATE INDEX IF NOT EXISTS idx_utterances_session ON utterances(session_id);
