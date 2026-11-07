-- Sevi store schema v2 (B spec §4.1). Scope: panel_responses omitted (human panel cut).
CREATE TABLE IF NOT EXISTS schema_version (v INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
  session_id           TEXT PRIMARY KEY,
  record_json          TEXT NOT NULL,          -- SessionRecord JSON with "trials": []
  state                TEXT NOT NULL,
  retention            TEXT NOT NULL,
  participant_type     TEXT NOT NULL,
  created_at           TEXT NOT NULL,          -- ISO-8601 UTC 'Z'
  started_at           TEXT,
  delete_after         TEXT,                   -- NULL for STUDY_KEEP
  report_token_sha256  TEXT UNIQUE,
  stripped             INTEGER NOT NULL DEFAULT 0,
  updated_at           TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS sessions_due ON sessions(retention, delete_after);
CREATE TABLE IF NOT EXISTS trials (
  session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  seq             INTEGER NOT NULL,
  presentation_id INTEGER NOT NULL,
  record_json     TEXT NOT NULL,
  PRIMARY KEY (session_id, seq));
CREATE TABLE IF NOT EXISTS twilio_status (
  call_sid TEXT NOT NULL, call_status TEXT NOT NULL, seq INTEGER NOT NULL, received_at TEXT NOT NULL,
  PRIMARY KEY (call_sid, call_status, seq));
CREATE TABLE IF NOT EXISTS idempotency (
  scope TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, expires_at TEXT NOT NULL,
  PRIMARY KEY (scope, key));
CREATE TABLE IF NOT EXISTS allowlist (
  participant_id TEXT PRIMARY KEY, number_hmac TEXT UNIQUE NOT NULL, entry_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ledger_events (
  event_id TEXT PRIMARY KEY, key_id TEXT NOT NULL, bucket TEXT NOT NULL, purpose TEXT NOT NULL,
  window_id TEXT NOT NULL, origin TEXT NOT NULL, at TEXT NOT NULL, event_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ledger_k ON ledger_events(key_id, bucket, at);
CREATE TABLE IF NOT EXISTS trial_minutes (
  call_sha256 TEXT PRIMARY KEY, minutes INTEGER NOT NULL, added_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ip_hits (ip_hmac TEXT NOT NULL, t REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ip_hits_k ON ip_hits(ip_hmac, t);
INSERT INTO schema_version(v) SELECT 2 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
