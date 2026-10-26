# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""SQLite storage for the annotation tool.

Everything the annotators produce lands here. datasets/dialfork/ is never
touched. One connection per request; WAL so concurrent
annotators do not block each other.
"""
import json
import os
import secrets
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS annotators (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    token        TEXT NOT NULL UNIQUE,
    created_at   REAL NOT NULL,
    consented_at REAL,
    note         TEXT,
    chocolates   INTEGER NOT NULL DEFAULT 10  -- allowance, paid pro rata over what they hold
);
CREATE TABLE IF NOT EXISTS items (
    item_id    TEXT PRIMARY KEY,  -- registered the first time a batch draws it
    task       TEXT NOT NULL,
    item_json  TEXT NOT NULL,     -- metadata incl. its foil, fixed from then on
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    name       TEXT PRIMARY KEY,
    task       TEXT NOT NULL,
    items_json TEXT NOT NULL,
    seed       INTEGER NOT NULL,  -- the draw's rng seed, for the record
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS assignments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    annotator_id INTEGER NOT NULL REFERENCES annotators(id),
    batch        TEXT NOT NULL,
    order_json   TEXT NOT NULL,   -- server-side presentation order, seeded
    created_at   REAL NOT NULL,
    UNIQUE (annotator_id, batch)
);
CREATE TABLE IF NOT EXISTS responses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_id INTEGER NOT NULL REFERENCES assignments(id),
    annotator_id  INTEGER NOT NULL REFERENCES annotators(id),
    item_id       TEXT NOT NULL,
    step          INTEGER NOT NULL,
    payload_json  TEXT NOT NULL,
    ms            INTEGER,
    submitted_at  REAL NOT NULL,
    UNIQUE (assignment_id, item_id, step)
);
CREATE TABLE IF NOT EXISTS task_intros (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    annotator_id INTEGER NOT NULL REFERENCES annotators(id),
    task         TEXT NOT NULL,
    action       TEXT NOT NULL,   -- 'read' | 'skipped' | 'reviewed' (reopened later)
    ms           INTEGER,         -- time spent on the intro screen
    at           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_resp_item ON responses(item_id);
CREATE INDEX IF NOT EXISTS idx_resp_ann  ON responses(annotator_id);
"""


def connect(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


SCHEMA_VERSION = "3"
DEFAULT_CHOCOLATES = 10


class OldDatabase(Exception):
    pass


def init(path):
    """Create or open the DB. Refuses a DB from the frozen-sample version (v1),
    whose batches and item ids mean something different."""
    con = connect(path)
    cols = {r["name"] for r in con.execute("PRAGMA table_info(batches)")}
    if cols and "task" not in cols:
        con.close()
        raise OldDatabase(
            "%s was made by the frozen-sample version of the tool (item ids like "
            "i0001, batches b00..b11). Batches are now drawn per task, so it cannot "
            "be reused: move it aside and start the server again." % path)
    ann_cols = {r["name"] for r in con.execute("PRAGMA table_info(annotators)")}
    with con:
        if ann_cols and "chocolates" not in ann_cols:
            # v2 -> v3: chocolates moved from a global per-batch rate to a
            # per-person allowance
            con.execute("ALTER TABLE annotators ADD COLUMN chocolates INTEGER NOT NULL "
                        "DEFAULT %d" % DEFAULT_CHOCOLATES)
            con.execute("DELETE FROM meta WHERE key='chocolates_per_batch'")
        con.executescript(SCHEMA)
        # the old URL admin token is gone: the panel is behind a password now
        con.execute("DELETE FROM meta WHERE key='admin_token'")
        con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema',?)",
                    (SCHEMA_VERSION,))
    from . import adminauth
    adminauth.ensure(con)
    con.close()


def meta_get(con, key):
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def meta_set(con, key, value):
    with con:
        con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))


# --- items ------------------------------------------------------------------

def all_items(con):
    return {r["item_id"]: json.loads(r["item_json"])
            for r in con.execute("SELECT item_id, item_json FROM items")}


def study_items(con):
    """The conversations that are part of the study: those in some batch. Items
    stay registered when their batch is deleted (so a redraw reuses the same id
    and foil); they drop out of the study with it."""
    held = batch_counts(con)
    return {i: it for i, it in all_items(con).items() if i in held}


def register_items(con, items):
    """Gives each new item its opaque id and stores it."""
    from .sampling import new_item_id
    with con:
        for it in items:
            it.setdefault("item_id", new_item_id())
            con.execute("INSERT OR IGNORE INTO items(item_id,task,item_json,created_at) "
                        "VALUES(?,?,?,?)",
                        (it["item_id"], it["task"], json.dumps(it), time.time()))


# --- batches ----------------------------------------------------------------

def _batch_row(r):
    return {"name": r["name"], "task": r["task"], "items": json.loads(r["items_json"]),
            "seed": r["seed"], "created_at": r["created_at"]}


def all_batches(con):
    """name -> {name, task, items, seed, created_at}"""
    return {r["name"]: _batch_row(r)
            for r in con.execute("SELECT * FROM batches ORDER BY task, name")}


def batch(con, name):
    r = con.execute("SELECT * FROM batches WHERE name=?", (name,)).fetchone()
    return _batch_row(r) if r else None


def next_batch_name(con, task):
    """<task>-01, <task>-02, ... -- the task is in the name."""
    used = {r["name"] for r in con.execute("SELECT name FROM batches WHERE task=?", (task,))}
    n = 1
    while ("%s-%02d" % (task, n)) in used:
        n += 1
    return "%s-%02d" % (task, n)


def create_batch(con, name, task, items, seed):
    with con:
        con.execute("INSERT INTO batches(name,task,items_json,seed,created_at) "
                    "VALUES(?,?,?,?,?)", (name, task, json.dumps(list(items)), seed,
                                          time.time()))
    return batch(con, name)


def batch_counts(con):
    """item_id -> how many batches contain it."""
    out = {}
    for r in con.execute("SELECT items_json FROM batches"):
        for i in json.loads(r["items_json"]):
            out[i] = out.get(i, 0) + 1
    return out


def delete_batch(con, name):
    """Removes the batch, everyone's assignment of it, and answers given under it."""
    with con:
        rows = con.execute("SELECT id FROM assignments WHERE batch=?", (name,)).fetchall()
        for r in rows:
            con.execute("DELETE FROM responses WHERE assignment_id=?", (r["id"],))
        con.execute("DELETE FROM assignments WHERE batch=?", (name,))
        con.execute("DELETE FROM batches WHERE name=?", (name,))
    return len(rows)


def batch_answer_count(con, name):
    r = con.execute(
        "SELECT COUNT(DISTINCT r.item_id) AS n FROM responses r "
        "JOIN assignments a ON a.id=r.assignment_id WHERE a.batch=?", (name,)).fetchone()
    return r["n"] if r else 0


def rename_annotator(con, annotator_id, name):
    with con:
        con.execute("UPDATE annotators SET name=? WHERE id=?", (name, annotator_id))


def delete_annotator(con, annotator_id):
    """Removes the person, their assignments, and every answer they gave."""
    with con:
        n = con.execute("SELECT COUNT(*) AS n FROM responses WHERE annotator_id=?",
                        (annotator_id,)).fetchone()["n"]
        con.execute("DELETE FROM responses WHERE annotator_id=?", (annotator_id,))
        con.execute("DELETE FROM assignments WHERE annotator_id=?", (annotator_id,))
        con.execute("DELETE FROM task_intros WHERE annotator_id=?", (annotator_id,))
        con.execute("DELETE FROM annotators WHERE id=?", (annotator_id,))
    return n


def record_intro(con, annotator_id, task, action, ms):
    with con:
        con.execute("INSERT INTO task_intros(annotator_id,task,action,ms,at) "
                    "VALUES(?,?,?,?,?)", (annotator_id, task, action, ms, time.time()))


def intros_for(con, annotator_id):
    """task -> {action, ms} of the FIRST time this person passed the task's intro
    (read or skipped). Later 'reviewed' visits are logged but do not change it."""
    out = {}
    for r in con.execute("SELECT task, action, ms FROM task_intros WHERE annotator_id=? "
                         "AND action IN ('read','skipped') ORDER BY id", (annotator_id,)):
        out.setdefault(r["task"], {"action": r["action"], "ms": r["ms"]})
    return out


def annotator_answer_count(con, annotator_id):
    return con.execute("SELECT COUNT(DISTINCT item_id) AS n FROM responses "
                       "WHERE annotator_id=?", (annotator_id,)).fetchone()["n"]


# --- annotators -------------------------------------------------------------

def create_annotator(con, name, note=None, chocolates=DEFAULT_CHOCOLATES):
    tok = secrets.token_urlsafe(16)
    with con:
        cur = con.execute(
            "INSERT INTO annotators(name,token,created_at,note,chocolates) VALUES(?,?,?,?,?)",
            (name, tok, time.time(), note, chocolates))
    return cur.lastrowid, tok


def set_chocolates(con, annotator_id, n):
    with con:
        con.execute("UPDATE annotators SET chocolates=? WHERE id=?", (n, annotator_id))


def annotator_by_token(con, token):
    return con.execute("SELECT * FROM annotators WHERE token=?", (token,)).fetchone()


def list_annotators(con):
    return con.execute("SELECT * FROM annotators ORDER BY id").fetchall()


def set_consent(con, annotator_id):
    with con:
        con.execute("UPDATE annotators SET consented_at=? WHERE id=? AND consented_at IS NULL",
                    (time.time(), annotator_id))


# --- assignments ------------------------------------------------------------

def assign_batch(con, annotator_id, batch, order):
    with con:
        con.execute(
            "INSERT OR IGNORE INTO assignments(annotator_id,batch,order_json,created_at) "
            "VALUES(?,?,?,?)",
            (annotator_id, batch, json.dumps(order), time.time()))
    return con.execute("SELECT * FROM assignments WHERE annotator_id=? AND batch=?",
                       (annotator_id, batch)).fetchone()


def assignment_by(con, annotator_id, batch):
    return con.execute("SELECT * FROM assignments WHERE annotator_id=? AND batch=?",
                       (annotator_id, batch)).fetchone()


def responses_count(con, assignment_id):
    return con.execute("SELECT COUNT(*) AS n FROM responses WHERE assignment_id=?",
                       (assignment_id,)).fetchone()["n"]


def items_answered_in(con, assignment_id):
    return con.execute(
        "SELECT COUNT(DISTINCT item_id) AS n FROM responses WHERE assignment_id=?",
        (assignment_id,)).fetchone()["n"]


def delete_assignment(con, assignment_id):
    """Drops the assignment and any answers recorded under it (FK would block)."""
    with con:
        con.execute("DELETE FROM responses WHERE assignment_id=?", (assignment_id,))
        con.execute("DELETE FROM assignments WHERE id=?", (assignment_id,))


def assignments_for(con, annotator_id):
    return con.execute("SELECT * FROM assignments WHERE annotator_id=? ORDER BY id",
                       (annotator_id,)).fetchall()


def all_assignments(con):
    return con.execute(
        "SELECT a.*, n.name AS annotator_name FROM assignments a "
        "JOIN annotators n ON n.id=a.annotator_id ORDER BY a.id").fetchall()


# --- responses --------------------------------------------------------------

def save_response(con, assignment_id, annotator_id, item_id, step, payload, ms):
    with con:
        con.execute(
            "INSERT OR REPLACE INTO responses"
            "(assignment_id,annotator_id,item_id,step,payload_json,ms,submitted_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (assignment_id, annotator_id, item_id, step, json.dumps(payload), ms, time.time()))


def responses_for_assignment(con, assignment_id):
    rows = con.execute(
        "SELECT item_id, step, payload_json, ms, submitted_at FROM responses "
        "WHERE assignment_id=?", (assignment_id,)).fetchall()
    out = {}
    for r in rows:
        out.setdefault(r["item_id"], {})[r["step"]] = {
            "payload": json.loads(r["payload_json"]),
            "ms": r["ms"], "submitted_at": r["submitted_at"]}
    return out


def all_responses(con):
    return con.execute(
        "SELECT r.*, n.name AS annotator_name FROM responses r "
        "JOIN annotators n ON n.id=r.annotator_id ORDER BY r.id").fetchall()


def items_done_by(con, annotator_id):
    """Items this annotator has fully completed, across ALL their assignments."""
    return {r["item_id"] for r in con.execute(
        "SELECT item_id FROM responses WHERE annotator_id=? AND step=3", (annotator_id,))}


def items_assigned_to(con, annotator_id):
    out = set()
    for a in assignments_for(con, annotator_id):
        out |= set(json.loads(a["order_json"]))
    return out


def annotation_counts(con):
    """item_id -> how many distinct annotators have completed it."""
    return {r["item_id"]: r["n"] for r in con.execute(
        "SELECT item_id, COUNT(DISTINCT annotator_id) AS n FROM responses "
        "WHERE step=3 GROUP BY item_id")}


def progress_counts(con):
    """annotator_id -> (items with step3 done, total assigned items)."""
    done = {}
    for r in con.execute(
            "SELECT annotator_id, COUNT(*) AS n FROM responses WHERE step=3 "
            "GROUP BY annotator_id"):
        done[r["annotator_id"]] = r["n"]
    return done
