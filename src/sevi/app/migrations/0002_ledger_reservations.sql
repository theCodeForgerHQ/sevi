-- Hard per-key credit budget (BUILD-BRIEF credit safety; adapters/ledger.py): a reservation is
-- written before every billed Gnani call and replaced by a ledger_events row after it. Rows left
-- by a crash keep counting as spent until an operator clears them.
CREATE TABLE IF NOT EXISTS ledger_reservations (
  reservation_id TEXT PRIMARY KEY, key_id TEXT NOT NULL, purpose TEXT NOT NULL,
  credits REAL NOT NULL, at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ledger_reservations_k ON ledger_reservations(key_id);
