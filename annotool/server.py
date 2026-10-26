# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Annotation server: stdlib http.server + sqlite3, no third-party deps.

  python3 -m annotool.server --root datasets/dialfork --port 8080

Blinding is enforced HERE, not in the browser. /api/item returns only what the
requested step is allowed to show, and a step is only served once the previous
one is submitted:

  step 1  task + workflow steps + transcript               (no cue, no fork info)
  step 2  + the cued instruction to look for           (still no fork turn)
  step 3  + fork turn, injected text, WIC substitute

If the cue appears in a step-1 payload or the fork turn in a step-2 payload,
the sequencing is decorative; test_server.py asserts it on the response bodies.
"""
import argparse
import json
import mimetypes
import os
import posixpath
import random
import re
import sys
import threading
from collections import defaultdict
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import adminauth, db, sampling
from .dialogs import (HALLUCINATION_TYPES, TYPE_ABBREV, cue_sentence, injection_event,
                      list_parent_ids, list_tasks, load_dialog, load_domain_tasks,
                      load_variant_pools, parent_path,
                      substitute_workflow, system_turn_indices)
from .workflow import TASK_DESCRIPTIONS, TASK_TOPICS, build_flow

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
TEMPLATES = os.path.join(HERE, "templates")
SAFE_PREFIX = re.compile(r"(/[A-Za-z0-9_-]+)+")
ROOT_URL = re.compile(rb'((?:href|src)=")/(?!/)')  # root-absolute, not //host


class Store(object):
    """Process-wide: read-only dataset access plus the registry of drawn items.

    Nothing is frozen. Items enter the registry (and the DB) the first time a
    batch draws them; the registry only ever grows while the server runs.
    """

    def __init__(self, root, db_path, star_dir, task2_root="predictions/task2",
                 dataset_py="generation/generate.py"):
        self.root = os.path.abspath(root)
        self.db_path = db_path
        self.star_dir = star_dir
        self.task2_root = task2_root
        self.pools = load_variant_pools(dataset_py)
        self.domain_tasks = load_domain_tasks(dataset_py)
        self.tasks = list_tasks(self.root)
        con = db.connect(db_path)
        self.items = db.all_items(con)
        self.by_ref = {it["ref"]: it for it in self.items.values()}
        cached = db.meta_get(con, "task_stats")
        con.close()
        self.task_stats = {t["task"]: t for t in json.loads(cached)} if cached else {}
        if set(self.task_stats) != set(self.tasks):
            self.task_stats = {}  # the dataset changed under the cache: recompute
        self._cache = {}
        self._lock = threading.Lock()
        self.draw_lock = threading.Lock()

    def register(self, con, items):
        db.register_items(con, items)
        with self._lock:
            for it in items:
                self.items[it["item_id"]] = it
                self.by_ref[it["ref"]] = it

    def compute_task_stats(self):
        """Sources and mean length per task, for choosing tasks. Reads every
        original once (~30 s on NFS), so it runs in the background and is cached."""
        stats = [sampling.task_stats(self.root, t) for t in self.tasks]
        con = db.connect(self.db_path)
        db.meta_set(con, "task_stats", json.dumps(stats))
        con.close()
        self.task_stats = {t["task"]: t for t in stats}

    def dialog(self, item):
        with self._lock:
            d = self._cache.get(item["item_id"])
        if d is None:
            d = load_dialog(os.path.join(self.root, item["path"]))
            with self._lock:
                self._cache[item["item_id"]] = d
        return d

    def intro_for(self, task):
        """The workflow of a task, read from any of its conformant originals
        (every dialogue of a task carries the same one)."""
        with self._lock:
            hit = self._cache.get(("intro", task))
        if hit is None:
            d = load_dialog(parent_path(self.root, task, list_parent_ids(self.root, task)[0]))
            topic, emoji = TASK_TOPICS.get(task, (task.replace("_", " "), ""))
            hit = {"task": task, "task_title": task.replace("_", " "),
                   "topic": topic, "emoji": emoji,
                   "task_description": TASK_DESCRIPTIONS.get(task, ""),
                   "flow": self.flow_for(task, d),
                   "has_flowchart": os.path.exists(
                       os.path.join(self.star_dir, task, "%s.jpg" % task))}
            with self._lock:
                self._cache[("intro", task)] = hit
        return hit

    def flow_for(self, task, dialog):
        meta = dialog["annotations"]["task"]
        return build_flow(task, meta["graph"], meta["replies"])


def finite(obj):
    """NaN/inf -> None, recursively. json.dumps writes them as bare NaN/Infinity,
    which is not JSON: JSON.parse in the browser fails on the whole response."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    return obj


def turns_payload(dialog):
    out, n = [], 0
    for i, t in enumerate(dialog.get("turns") or []):
        is_sys = t.get("speaker") == "System"
        if is_sys:
            n += 1
        out.append({"i": i, "speaker": t.get("speaker"),
                    "sys_n": n if is_sys else None,
                    "text": t.get("text") or ""})
    return out


def item_payload(store, item, step):
    """Step-gated. Adding a field here to the wrong branch breaks the study."""
    d = store.dialog(item)
    task = item["task"]
    topic, emoji = TASK_TOPICS.get(task, (task.replace("_", " "), ""))
    payload = {
        "item_id": item["item_id"],
        "step": step,
        "task": task,
        "task_title": task.replace("_", " "),
        "topic": topic, "emoji": emoji,
        "task_description": TASK_DESCRIPTIONS.get(task, ""),
        "flow": store.flow_for(task, d),
        "user_goal": (d.get("personas", {}).get("User", {}) or {}).get("circumstances", ""),
        "turns": turns_payload(d),
        "n_system_turns": len(system_turn_indices(d)),
        "has_flowchart": os.path.exists(
            os.path.join(store.star_dir, task, "%s.jpg" % task)),
    }
    if step >= 2:
        payload["cue"] = cue_sentence(item["cue_type"], item["cue_variant"], store.pools)
    if step >= 3:
        payload["reveal"] = reveal_payload(store, item, d)
    return payload


def reveal_payload(store, item, dialog):
    ev = injection_event(dialog)
    out = {
        "kind": item["kind"],
        "is_foil": item["is_foil"],
        "fork_assistant_turn": item["fork_assistant_turn"],
        "segment": item["segment"],
        "true_type": item["hallucination_type"],
        "true_type_abbrev": TYPE_ABBREV.get(item["hallucination_type"] or ""),
        "true_variant": item["variant_id"],
        "injected_text": (ev or {}).get("content"),
        "true_cue": None,
        "substitute": None,
    }
    if item["hallucination_type"]:
        out["true_cue"] = cue_sentence(item["hallucination_type"],
                                       item["variant_id"], store.pools)
    sub = substitute_workflow(dialog) if item["hallucination_type"] == \
        "workflow_identity_corruption" else None
    if sub:
        name, graph, replies = sub
        out["substitute"] = {
            "name": name,
            "description": TASK_DESCRIPTIONS.get(name, ""),
            "flow": build_flow(name, graph, replies),
        }
    return out


# --- per-annotator flow -----------------------------------------------------

def next_step_for(done, item_id):
    """The only step this annotator may be served for this item (1..3), or None."""
    steps = done.get(item_id, {})
    for s in (1, 2, 3):
        if s not in steps:
            return s
    return None


def assignment_order(items, seed):
    """Presentation order is per (annotator, batch) and seeded, so it is
    reconstructible but different for each person holding the same batch.

    One mini-block at a time: a batch lists its items mini-block by mini-block
    (sampling.batch_cells), and each run of MINI is kept together, its items and
    the runs themselves shuffled. Someone who stops after any number of runs has
    judged a sample balanced on type and on fork position."""
    rng = random.Random(seed)
    items = list(items)
    runs = [items[i:i + sampling.MINI] for i in range(0, len(items), sampling.MINI)]
    rng.shuffle(runs)
    for r in runs:
        rng.shuffle(r)
    return [i for r in runs for i in r]


# A real reward handed out at the end: each person has an allowance of
# chocolates (set per annotator, typically by the time they offered), paid out
# pro rata over ALL the conversations they hold. 0 hides them for that person.
# It is a rate, not a ledger: changing an allowance, or giving someone another
# batch, changes what they have already earned too.

def chocolates(assigns, allowance):
    """Derived from progress, never stored: it cannot drift and nothing earns it
    twice. Exact fractions, so finishing everything pays exactly the allowance.
    Whole chocolates earned, how much of the next one (0..1), and the allowance."""
    total = sum(a["total"] for a in assigns)
    done = sum(a["done"] for a in assigns)
    earned = Fraction(allowance * done, total) if total else Fraction(0)
    whole = int(earned)
    return {"count": whole, "next": float(earned - whole), "total": allowance}


def state_for(store, con, ann):
    rows = db.assignments_for(con, ann["id"])
    # Defensive: should an annotator ever hold two batches sharing an item,
    # progress is stored per assignment, so without this the same item would be
    # served twice and the second submit would 409 forever.
    done_anywhere = db.items_done_by(con, ann["id"])
    task_of = {name: b["task"] for name, b in db.all_batches(con).items()}
    # finish a task before switching: all batches of one task are done back to
    # back, tasks in the order they were first given (a second block of a task
    # handed out later joins its task instead of queueing behind everything)
    first = {}
    for a in rows:
        first.setdefault(task_of.get(a["batch"]), len(first))
    rows = sorted(rows, key=lambda a: (first[task_of.get(a["batch"])], a["id"]))
    assigns, current = [], None
    for a in rows:
        order = json.loads(a["order_json"])
        done = db.responses_for_assignment(con, a["id"])
        complete = sum(1 for i in order
                       if len(done.get(i, {})) >= 3 or i in done_anywhere)
        nxt = None
        for item_id in order:
            if item_id in done_anywhere and len(done.get(item_id, {})) < 3:
                continue          # already answered under another assignment
            s = next_step_for(done, item_id)
            if s is not None:
                nxt = {"item_id": item_id, "step": s}
                break
        assigns.append({"batch": a["batch"], "task": task_of.get(a["batch"]),
                        "total": len(order), "done": complete, "next": nxt})
        if current is None and nxt is not None:
            current = {"batch": a["batch"], "assignment_id": a["id"],
                       "task": task_of.get(a["batch"])}
            current.update(nxt)
    # the annotator's work as a sequence of TASKS (in the order they were given),
    # each with its conversations -- what the progress header shows
    intros = db.intros_for(con, ann["id"])
    tasks = []
    for x in assigns:
        t = next((y for y in tasks if y["task"] == x["task"]), None)
        if t is None:
            t = {"task": x["task"], "title": (x["task"] or "").replace("_", " "),
                 "emoji": TASK_TOPICS.get(x["task"], ("", ""))[1],
                 "done": 0, "total": 0, "intro": intros.get(x["task"])}
            tasks.append(t)
        t["done"] += x["done"]
        t["total"] += x["total"]
    if current is not None:
        # first conversation of a task this person has not been introduced to yet
        current["need_intro"] = current["task"] not in intros
    return {
        "annotator": {"name": ann["name"], "consented": bool(ann["consented_at"])},
        "assignments": assigns,
        "tasks": tasks,
        "chocolates": chocolates(assigns, ann["chocolates"]),
        "current": current,
    }


def find_assignment(con, ann, item_id):
    """The assignment this annotator is actually working this item under.

    Prefer one that already holds partial responses for it, so a part-answered
    item is not silently re-started under a different assignment.
    """
    holding = [a for a in db.assignments_for(con, ann["id"])
               if item_id in json.loads(a["order_json"])]
    for a in holding:
        if item_id in db.responses_for_assignment(con, a["id"]):
            return a
    return holding[0] if holding else None


# --- HTTP -------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "dialfork-annotool/1.0"
    store = None

    def log_message(self, fmt, *args):
        if os.environ.get("ANNOTOOL_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers
    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}):
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(finite(obj)))

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except ValueError:
            return {}

    def _prefix(self):
        """The path prefix a reverse proxy serves us under (X-Forwarded-Prefix,
        e.g. /criteria/forkbench), or "" when accessed directly. The proxy strips
        it before forwarding, so routing never sees it; only the URLs handed to
        the browser need it. Anything but a plain path is ignored: it goes into HTML."""
        v = (self.headers.get("X-Forwarded-Prefix") or "").strip().rstrip("/")
        return v if SAFE_PREFIX.fullmatch(v) else ""

    def _page(self, name):
        path = os.path.join(TEMPLATES, name)
        if not os.path.exists(path):
            return self._err(404, "missing template %s" % name)
        with open(path, "rb") as fh:
            html = fh.read()
        prefix = self._prefix().encode()
        if prefix:  # without one the page goes out exactly as written
            html = ROOT_URL.sub(lambda m: m.group(1) + prefix + b"/", html)
            html = html.replace(b"<script", b'<script>window.BASE = "%s";</script>\n<script'
                                % prefix, 1)
        self._send(200, html, "text/html; charset=utf-8")

    def _auth(self, con, body):
        tok = body.get("token") or ""
        ann = db.annotator_by_token(con, tok) if tok else None
        return ann

    def _session(self):
        from http.cookies import SimpleCookie
        c = SimpleCookie(self.headers.get("Cookie") or "")
        return c[adminauth.COOKIE].value if adminauth.COOKIE in c else ""

    def _admin_ok(self, con):
        return adminauth.valid(con, self._session())

    def _query(self):
        from urllib.parse import parse_qs, urlparse
        return parse_qs(urlparse(self.path).query)

    # -- routing
    def do_GET(self):
        from urllib.parse import urlparse
        p = urlparse(self.path).path
        if p == "/":
            return self._send(200, "<h1>DialFork annotation</h1>"
                                   "<p>Use the personal link you were sent.</p>",
                              "text/html; charset=utf-8")
        if p.startswith("/a/"):
            return self._page("annotate.html")
        if p == "/api/admin/export":
            return self._export()
        if p == "/admin":
            con = db.connect(self.store.db_path)
            try:
                ok = self._admin_ok(con)
            finally:
                con.close()
            return self._page("admin.html" if ok else "login.html")
        if p.startswith("/static/"):
            return self._static(p[len("/static/"):])
        if p.startswith("/flow/"):
            return self._flowchart(p[len("/flow/"):])
        return self._err(404, "not found")

    def _static(self, rel):
        rel = posixpath.normpath("/" + rel).lstrip("/")
        path = os.path.join(STATIC, rel)
        if not os.path.abspath(path).startswith(STATIC) or not os.path.exists(path):
            return self._err(404, "not found")
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as fh:
            self._send(200, fh.read(), ctype)

    def _flowchart(self, rel):
        task = re.sub(r"[^a-zA-Z0-9_]", "", rel.replace(".jpg", ""))
        path = os.path.join(self.store.star_dir, task, "%s.jpg" % task)
        if not os.path.exists(path):
            return self._err(404, "no flowchart")
        with open(path, "rb") as fh:
            blob = fh.read()
        # 2 of STAR's 24 ".jpg" flowcharts are actually PNGs
        # (restaurant_book, ride_book), so sniff rather than trust the extension.
        ctype = "image/png" if blob[:4] == b"\x89PNG" else "image/jpeg"
        self._send(200, blob, ctype)

    def do_POST(self):
        from urllib.parse import urlparse
        p = urlparse(self.path).path
        body = self._body()
        con = db.connect(self.store.db_path)
        try:
            if p == "/api/admin/login":
                return self._login(con, body)
            if p.startswith("/api/admin/"):
                if not self._admin_ok(con):
                    return self._json({"error": "please log in", "login": True}, 401)
                action = p[len("/api/admin/"):]
                if action == "logout":
                    return self._send(200, json.dumps({"ok": True}),
                                      extra=[("Set-Cookie", adminauth.cookie_header("", 0))])
                if action == "password":
                    return self._password(con, body)
                # the public default opens nothing but the change-password form
                if adminauth.using_default(con):
                    return self._json({"error": "change the default password first",
                                       "must_change": True}, 403)
                return self._admin(action, con, body)
            ann = self._auth(con, body)
            if ann is None:
                return self._err(403, "unknown or missing token")
            if p == "/api/state":
                return self._json(state_for(self.store, con, ann))
            if p == "/api/consent":
                db.set_consent(con, ann["id"])
                return self._json(state_for(self.store, con, ann))
            if p == "/api/intro":
                return self._intro(con, ann, body)
            if p == "/api/intro_done":
                task = body.get("task") or ""
                action = body.get("action")
                if action not in ("read", "skipped", "reviewed"):
                    return self._err(400, "bad action")
                if task not in {x["task"] for x in state_for(self.store, con, ann)["tasks"]}:
                    return self._err(403, "not one of your tasks")
                db.record_intro(con, ann["id"], task, action, body.get("ms"))
                return self._json(state_for(self.store, con, ann))
            if p == "/api/item":
                return self._item(con, ann, body)
            if p == "/api/answer":
                return self._answer(con, ann, body)
            return self._err(404, "not found")
        finally:
            con.close()

    def _item(self, con, ann, body):
        item_id = body.get("item_id")
        item = self.store.items.get(item_id)
        if item is None:
            return self._err(404, "unknown item")
        a = find_assignment(con, ann, item_id)
        if a is None:
            return self._err(403, "item not assigned to you")
        done = db.responses_for_assignment(con, a["id"])
        step = next_step_for(done, item_id)
        if step is None:
            return self._json({"done": True})
        # The client may ask, but the server decides: never serve a step whose
        # predecessor is unanswered.
        out = item_payload(self.store, item, step)
        if step == 3:
            # the annotator's own earlier answers, for the neutral side-by-side
            # in the reveal (they are final by now; step 3 cannot change them)
            s2 = (done.get(item_id, {}).get(2) or {}).get("payload") or {}
            out["yours"] = {"appears": s2.get("appears"), "onset": s2.get("onset_sys_turn")}
        return self._json(out)

    def _intro(self, con, ann, body):
        """A task's intro: its description and steps, examples included. Only
        the workflow -- nothing about any conversation."""
        task = body.get("task") or ""
        if task not in {x["task"] for x in state_for(self.store, con, ann)["tasks"]}:
            return self._err(403, "not one of your tasks")
        return self._json(self.store.intro_for(task))

    def _answer(self, con, ann, body):
        item_id = body.get("item_id")
        item = self.store.items.get(item_id)
        if item is None:
            return self._err(404, "unknown item")
        a = find_assignment(con, ann, item_id)
        if a is None:
            return self._err(403, "item not assigned to you")
        done = db.responses_for_assignment(con, a["id"])
        expected = next_step_for(done, item_id)
        step = int(body.get("step") or 0)
        if expected is None:
            return self._err(409, "already complete")
        if step != expected:
            return self._err(409, "expected step %d, got %d" % (expected, step))
        ok, msg = validate(step, body.get("payload") or {}, item, self.store)
        if not ok:
            return self._err(400, msg)
        db.save_response(con, a["id"], ann["id"], item_id, step,
                         body.get("payload") or {}, body.get("ms"))
        return self._json(state_for(self.store, con, ann))

    def _admin(self, action, con, body):
        if action == "stats":
            return self._json(admin_stats(self.store, con))
        if action == "report":
            return self._report(con, body)

        # ---- annotators
        if action == "annotator_add":
            name = (body.get("name") or "").strip()
            if not name:
                return self._err(400, "a name is required")
            n = parse_chocolates(body.get("chocolates", db.DEFAULT_CHOCOLATES))
            if n is None:
                return self._err(400, "chocolates must be a whole number from 0 to 1000")
            aid, tok = db.create_annotator(con, name, body.get("note"), chocolates=n)
            out = admin_stats(self.store, con)
            out["created_annotator"] = {"id": aid, "name": name, "token": tok}
            return self._json(out)
        if action == "annotator_rename":
            ann = self._ann(con, body)
            if ann is None:
                return self._err(404, "unknown annotator")
            name = (body.get("name") or "").strip()
            if not name:
                return self._err(400, "a name is required")
            db.rename_annotator(con, ann["id"], name)
            return self._json(admin_stats(self.store, con))
        if action == "annotator_delete":
            ann = self._ann(con, body)
            if ann is None:
                return self._err(404, "unknown annotator")
            n = db.annotator_answer_count(con, ann["id"])
            if n and not body.get("force"):
                return self._err(409,
                    "%s has answered %d conversation(s). Deleting them removes "
                    "those answers permanently." % (ann["name"], n))
            db.delete_annotator(con, ann["id"])
            out = admin_stats(self.store, con)
            out["removed"] = {"what": ann["name"], "answers_deleted": n}
            return self._json(out)

        if action == "annotator_chocolates":
            ann = self._ann(con, body)
            if ann is None:
                return self._err(404, "unknown annotator")
            n = parse_chocolates(body.get("chocolates"))
            if n is None:
                return self._err(400, "chocolates must be a whole number from 0 to 1000")
            db.set_chocolates(con, ann["id"], n)
            return self._json(admin_stats(self.store, con))

        # ---- batches
        if action == "batch_add":
            task = body.get("task") or ""
            if task not in self.store.tasks:
                return self._err(400, "pick a task")
            try:
                size = int(body.get("size", sampling.BLOCK))
            except (TypeError, ValueError):
                size = 0
            if not (sampling.MINI <= size <= MAX_BATCH and size % sampling.MINI == 0):
                return self._err(400, "a batch is %d to %d conversations, in steps of %d"
                                 % (sampling.MINI, MAX_BATCH, sampling.MINI))
            ann = None
            if body.get("annotator_id"):
                ann = self._ann(con, body)
                if ann is None:
                    return self._err(404, "unknown annotator")
            try:
                name, picked = new_batch(self.store, con, task, size // sampling.MINI, ann)
            except sampling.DrawError as exc:
                return self._err(409, str(exc))
            out = admin_stats(self.store, con)
            out["created_batch"] = {"name": name, "size": len(picked),
                                    "for": ann["name"] if ann else None}
            return self._json(out)
        if action == "batch_delete":
            name = body.get("batch") or ""
            b = db.batch(con, name)
            if b is None:
                return self._err(404, "unknown batch")
            n = db.batch_answer_count(con, name)
            if n and not body.get("force"):
                return self._err(409,
                    "%s has %d answered conversation(s) in it. Deleting the batch "
                    "removes those answers permanently." % (name, n))
            db.delete_batch(con, name)
            out = admin_stats(self.store, con)
            out["removed"] = {"what": name, "answers_deleted": n}
            return self._json(out)

        # ---- looking inside a batch (admin sees everything: ground truth included)
        if action == "batch_view":
            b = db.batch(con, body.get("batch") or "")
            if b is None:
                return self._err(404, "unknown batch")
            counts = db.annotation_counts(con)
            rows = []
            for i in b["items"]:
                it = self.store.items.get(i)
                if it is None:
                    continue
                rows.append({"item_id": i, "ref": it["ref"], "kind": it["kind"],
                             "type": TYPE_ABBREV.get(it["hallucination_type"] or ""),
                             "segment": it["segment"], "dialog_id": it["dialog_id"],
                             "is_foil": it["is_foil"],
                             "cue_type": TYPE_ABBREV.get(it["cue_type"] or ""),
                             "judged": counts.get(i, 0)})
            return self._json({"name": b["name"], "task": b["task"], "items": rows,
                               "held_by": [r["annotator_name"] for r in db.all_assignments(con)
                                           if r["batch"] == b["name"]]})
        if action == "item_view":
            item = self.store.items.get(body.get("item_id"))
            if item is None:
                return self._err(404, "unknown item")
            out = item_payload(self.store, item, 3)
            out["meta"] = {k: item[k] for k in ("ref", "kind", "segment", "fork_assistant_turn",
                                                "variant_id", "is_foil", "cue_variant",
                                                "dialog_id")}
            out["meta"]["type"] = TYPE_ABBREV.get(item["hallucination_type"] or "")
            out["meta"]["cue_type"] = TYPE_ABBREV.get(item["cue_type"] or "")
            out["answers"] = item_answers(con, item["item_id"])
            return self._json(out)

        # ---- who holds what
        if action == "assign":
            ann = self._ann(con, body)
            if ann is None:
                return self._err(404, "unknown annotator")
            b = db.batch(con, body.get("batch") or "")
            if b is None:
                return self._err(400, "unknown batch")
            clash = sources_of(self.store, db.items_assigned_to(con, ann["id"])) \
                & sources_of(self.store, b["items"])
            if clash:
                return self._err(409,
                    "%s already holds %d of the source dialogues in %s. Nobody may "
                    "see two versions of the same dialogue: they share its opening, "
                    "so comparing them would give the fork away."
                    % (ann["name"], len(clash), b["name"]))
            give(con, ann, b)
            return self._json(admin_stats(self.store, con))
        if action == "unassign":
            ann = self._ann(con, body)
            if ann is None:
                return self._err(404, "unknown annotator")
            a = db.assignment_by(con, ann["id"], body.get("batch") or "")
            if a is None:
                return self._err(404, "that annotator does not hold that batch")
            touched = db.items_answered_in(con, a["id"])
            if touched and not body.get("force"):
                return self._err(409,
                    "%s has already answered %d conversation(s) in %s. Taking it "
                    "back deletes those answers."
                    % (ann["name"], touched, a["batch"]))
            db.delete_assignment(con, a["id"])
            out = admin_stats(self.store, con)
            out["removed"] = {"what": "%s from %s" % (a["batch"], ann["name"]),
                              "answers_deleted": touched}
            return self._json(out)
        return self._err(404, "not found")

    def _export(self):
        """Download everything collected: answers.csv, or a .zip with the answers,
        the metric correlation, who holds which batch, and the report.

        GET, so the browser can save it: authenticated by the session cookie."""
        import io, time as _t, zipfile
        from . import analyze as A
        con = db.connect(self.store.db_path)
        try:
            if not self._admin_ok(con):
                return self._err(401, "please log in")
            if adminauth.using_default(con):
                return self._err(403, "change the default password first")
            q = self._query()
            what = (q.get("what") or ["csv"])[0]
            metrics = (q.get("metrics") or ["0"])[0] == "1"
            items = db.study_items(con)
            answers = A.load_answers(self.store.db_path)      # everyone, nobody excluded
            enc = "dialog2flow-joint-dse-base"
            cols, dist, corr = [], None, None
            if metrics:
                cols = [m for m, _ in A.task2_methods(self.store.task2_root, enc)]
                dist = A.metric_distances(items, list(answers), self.store.task2_root, enc)
            buf = io.StringIO()
            A.write_answers_csv(items, answers, buf, dist, cols)
            answers_csv = "\ufeff" + buf.getvalue()           # BOM: Excel reads UTF-8
            stamp = _t.strftime("%Y-%m-%d")
            if what != "zip":
                return self._send(200, answers_csv, "text/csv; charset=utf-8", extra=[(
                    "Content-Disposition",
                    'attachment; filename="dialfork-answers-%s.csv"' % stamp)])

            import csv
            if metrics:
                corr = A.metric_rows(items, answers, self.store.task2_root, None, enc)
            files = {"answers.csv": answers_csv}
            b = io.StringIO()
            w = csv.writer(b)
            w.writerow(["batch", "task", "annotator", "conversations", "completed"])
            done = {}
            for r in con.execute("SELECT a.batch, n.name, COUNT(DISTINCT r.item_id) AS n "
                                 "FROM assignments a JOIN annotators n ON n.id=a.annotator_id "
                                 "LEFT JOIN responses r ON r.assignment_id=a.id AND r.step=3 "
                                 "GROUP BY a.id"):
                done[(r["batch"], r["name"])] = r["n"]
            for name, bt in db.all_batches(con).items():
                holders = [k[1] for k in done if k[0] == name] or [""]
                for h in holders:
                    w.writerow([name, bt["task"], h, len(bt["items"]), done.get((name, h), "")])
            files["batches.csv"] = "\ufeff" + b.getvalue()
            if metrics:
                m = io.StringIO()
                w = csv.writer(m)
                w.writerow(["method", "encoder", "n_forks", "spearman_rho", "ci95_low",
                            "ci95_high"])
                for row in corr or []:
                    w.writerow([row[0], row[1], row[2]] +
                               ["" if v != v else round(v, 4) for v in row[3:]])
                files["metric_correlation.csv"] = "\ufeff" + m.getvalue()
            files["report.md"] = A.analyze(items, answers,
                                           self.store.task2_root if metrics else None,
                                           None, enc, corr=corr)
            files["README.txt"] = (
                "DialFork human annotation export, %s\n\n"
                "answers.csv             one row per (conversation, annotator); every answer,\n"
                "                        nobody excluded%s\n"
                "batches.csv             which batch each annotator holds, and how far they got\n"
                "%s"
                "report.md               the full report, as in the admin panel\n"
                % (stamp, "; metric_* columns = each Task-2 metric's distance\n"
                   "                        from the conformant original (forks only)"
                   if metrics else "",
                   "metric_correlation.csv  Spearman rho, human deviation vs each metric\n"
                   if metrics else ""))
            z = io.BytesIO()
            with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, text in files.items():
                    zf.writestr("dialfork-export-%s/%s" % (stamp, fname), text)
            return self._send(200, z.getvalue(), "application/zip", extra=[(
                "Content-Disposition",
                'attachment; filename="dialfork-export-%s.zip"' % stamp)])
        finally:
            con.close()

    def _login(self, con, body):
        addr = self.client_address[0]
        if adminauth.locked_out(addr):
            return self._err(429, "too many wrong passwords -- wait a few minutes")
        value = adminauth.login(con, addr, body.get("password"))
        if value is None:
            return self._err(401, "wrong password")
        return self._send(200, json.dumps({"ok": True,
                                           "must_change": adminauth.using_default(con)}),
                          extra=[("Set-Cookie", adminauth.cookie_header(value))])

    def _password(self, con, body):
        if not adminauth.login(con, self.client_address[0], body.get("current")):
            return self._err(403, "the current password is not right")
        try:
            adminauth.set_password(con, body.get("new") or "")
        except ValueError as exc:
            return self._err(400, str(exc))
        # the secret rotated, so every other session is out; this one continues
        return self._send(200, json.dumps({"ok": True}),
                          extra=[("Set-Cookie",
                                  adminauth.cookie_header(adminauth.new_session(con)))])

    def _ann(self, con, body):
        return con.execute("SELECT * FROM annotators WHERE id=?",
                           (body.get("annotator_id"),)).fetchone()

    def _report(self, con, body):
        from . import analyze as A
        items = db.study_items(con)
        answers = A.load_answers(self.store.db_path, body.get("exclude") or [])
        data = A.summary_data(items, answers)
        encoder = body.get("encoder") or "dialog2flow-joint-dse-base"
        root = self.store.task2_root if body.get("metrics") else None
        corr = A.metric_rows(items, answers, root, None, encoder) if root else None
        data["metric_correlation"] = None if root is None else [
            {"method": m, "encoder": e, "n": n, "rho": rho,
             "lo": None if lo != lo else lo, "hi": None if hi != hi else hi}
            for m, e, n, rho, lo, hi in (corr or [])]
        text = A.analyze(items, answers, root, None, encoder, corr=corr)
        return self._json({"summary": data, "markdown": text,
                           "excluded": body.get("exclude") or []})

def item_answers(con, item_id):
    """Everyone's answers to one conversation, for the admin viewer."""
    by = {}
    for r in con.execute(
            "SELECT r.step, r.payload_json, n.name FROM responses r "
            "JOIN annotators n ON n.id=r.annotator_id WHERE r.item_id=? ORDER BY r.id",
            (item_id,)):
        by.setdefault(r["name"], {})[r["step"]] = json.loads(r["payload_json"])
    out = []
    for name, st in by.items():
        out.append({"annotator": name,
                    "slider": st.get(1, {}).get("slider"),
                    "appears": st.get(2, {}).get("appears"),
                    "onset": st.get(2, {}).get("onset_sys_turn"),
                    "prefix_ok": st.get(3, {}).get("prefix_ok"),
                    "note": st.get(3, {}).get("note"),
                    "complete": 3 in st})
    return out


def sources_of(store, item_ids):
    """The source dialogues behind these items -- the unit nobody may see twice."""
    return {store.items[i]["source_id"] for i in item_ids if i in store.items}


def give(con, ann, b):
    order = assignment_order(b["items"], "%s|%s" % (ann["id"], b["name"]))
    db.assign_batch(con, ann["id"], b["name"], order)


MAX_BATCH = 100


def new_batch(store, con, task, minis, ann=None, seed=None):
    """Draw a batch of `minis` mini-blocks of `task`, record it, and -- when it is
    for a named person -- give it to them. For a person, the draw avoids every
    source they hold and continues quartets others started; for anyone, it takes
    sources nobody holds yet. Returns (name, items); raises sampling.DrawError.
    `seed` is for scripted, reproducible draws; by default it is random."""
    with store.draw_lock:
        if seed is None:
            seed = random.SystemRandom().randrange(2 ** 31)
        avoid = sources_of(store, db.items_assigned_to(con, ann["id"])) if ann else set()
        picked, new = sampling.draw_batch(
            store.root, task, minis, store.by_ref, db.batch_counts(con),
            random.Random(seed), store.pools, store.domain_tasks,
            avoid_sources=avoid, complete_quartets=ann is not None)
        store.register(con, new)
        name = db.next_batch_name(con, task)
        db.create_batch(con, name, task, [i["item_id"] for i in picked], seed)
    if ann:
        give(con, ann, db.batch(con, name))
    return name, picked


def parse_chocolates(v):
    """A whole number 0..1000 (as a number or a string of digits), or None."""
    if isinstance(v, bool) or isinstance(v, float) and not v.is_integer():
        return None
    try:
        n = int(str(v).strip()) if isinstance(v, str) else int(v)
    except (TypeError, ValueError):
        return None
    return n if 0 <= n <= 1000 else None


def validate(step, payload, item, store):
    if step == 1:
        v = payload.get("slider")
        if not isinstance(v, (int, float)) or not (0 <= v <= 100):
            return False, "slider must be 0-100"
    elif step == 2:
        if payload.get("appears") not in ("yes", "partly", "no"):
            return False, "appears must be yes/partly/no"
        onset = payload.get("onset_sys_turn")
        if payload["appears"] in ("yes", "partly"):
            n = len(system_turn_indices(store.dialog(item)))
            if not isinstance(onset, int) or not (1 <= onset <= n):
                return False, "onset must be a system turn number"
        elif onset is not None:
            return False, "onset must be empty when the behaviour does not appear"
    elif step == 3:
        # "n/a" exactly when there is nothing before the fork to judge
        if sampling.has_prefix(item):
            if payload.get("prefix_ok") not in ("yes", "mostly", "no"):
                return False, "prefix_ok must be yes/mostly/no"
        elif payload.get("prefix_ok") != "n/a":
            return False, "prefix_ok must be n/a: there is nothing before the fork"
    else:
        return False, "bad step"
    return True, ""


def composition(store, ids):
    """What a batch is made of, for the admin panel."""
    its = [store.items[i] for i in ids if i in store.items]
    forks = [i for i in its if i["kind"] == "fork"]
    return {"conformant": sum(1 for i in its if i["kind"] == "conformant"),
            "segments": [sum(1 for i in forks if i["segment"] == k) for k in (1, 2, 3, 4)],
            "types": {TYPE_ABBREV[h]: sum(1 for i in forks if i["hallucination_type"] == h)
                      for h in HALLUCINATION_TYPES},
            "sources": len({i["source_id"] for i in its}),
            "one_per_source": len({i["source_id"] for i in its}) == len(its)}


def admin_stats(store, con):
    counts = db.annotation_counts(con)
    done_counts = db.progress_counts(con)
    defined = db.all_batches(con)

    held = {}
    for r in db.all_assignments(con):
        held.setdefault(r["batch"], []).append(r["annotator_name"])

    owns = {}
    anns = []
    for a in db.list_annotators(con):
        assigned = db.items_assigned_to(con, a["id"])
        owns[a["name"]] = sources_of(store, assigned)
        mine = [x["batch"] for x in db.assignments_for(con, a["id"])]
        # listed in queue order: grouped by task, tasks in the order first given
        first = {}
        for b in mine:
            first.setdefault(defined.get(b, {}).get("task"), len(first))
        mine.sort(key=lambda b: (first[defined.get(b, {}).get("task")], b))
        anns.append({"id": a["id"], "name": a["name"], "token": a["token"],
                     "consented": bool(a["consented_at"]),
                     "assigned": len(assigned), "done": done_counts.get(a["id"], 0),
                     "answers": db.annotator_answer_count(con, a["id"]),
                     "chocolates": state_for(store, con, a)["chocolates"]["count"],
                     "allowance": a["chocolates"],
                     "batches": mine,
                     "intros": db.intros_for(con, a["id"]),
                     "tasks": sorted({defined[b]["task"] for b in mine if b in defined})})

    batches = []
    for name, b in defined.items():
        srcs = sources_of(store, b["items"])
        batches.append({
            "name": name,
            "task": b["task"],
            "emoji": TASK_TOPICS.get(b["task"], ("", ""))[1],
            "size": len(b["items"]),
            "taken_by": held.get(name, []),
            "eligible": sorted(n for n, o in owns.items() if not (o & srcs)),
            # who is left out and why: they already hold n of these source dialogues
            "blocked": [{"name": n, "overlap": len(o & srcs)}
                        for n, o in sorted(owns.items()) if o & srcs],
            "composition": composition(store, b["items"]),
            "judged": sum(1 for i in b["items"] if counts.get(i, 0) > 0),
            "answers": db.batch_answer_count(con, name),
        })

    in_batches = set(db.batch_counts(con))
    cover = {"0": 0, "1": 0, "2+": 0}
    for i in in_batches:
        n = counts.get(i, 0)
        cover["0" if n == 0 else ("1" if n == 1 else "2+")] += 1

    per_task = {}
    for b in defined.values():
        per_task.setdefault(b["task"], set()).update(sources_of(store, b["items"]))
    tasks = []
    for t in store.tasks:
        st = store.task_stats.get(t) or {}
        tasks.append({"task": t, "sources": st.get("sources"),
                      "mean_system_turns": st.get("mean_system_turns"),
                      "sources_drawn": len(per_task.get(t, ())),
                      "batches": sum(1 for b in defined.values() if b["task"] == t)})

    times = [r["ms"] for r in db.all_responses(con) if r["ms"]]
    times.sort()
    return {"annotators": anns, "batches": batches, "coverage": cover,
            "n_items": len(in_batches), "tasks": tasks, "mini": sampling.MINI,
            "max_batch": MAX_BATCH,
            "median_seconds_per_step": (times[len(times) // 2] / 1000.0) if times else None,
            # the address annotators reach the server at (--public-url); links in
            # the panel are built from it, not from wherever the admin is browsing
            "public_url": getattr(store, "public_url", None),
            "responses": len(times),
            "chocolates_owed": sum(a["chocolates"] for a in anns),
            "chocolates_allowed": sum(a["allowance"] for a in anns)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/dialfork",
                    help="the DialFork dataset root (opened read-only)")
    ap.add_argument("--db", default="annotool/data/annot.sqlite")
    ap.add_argument("--dataset-py", default="generation/generate.py",
                    help="source of the variant sentences annotators read")
    ap.add_argument("--star", default="datasets/STAR/tasks",
                    help="for the optional collapsed flowchart images")
    ap.add_argument("--task2", default="predictions/task2",
                    help="Task 2 distances for the optional metric correlation: a "
                         "results/task2 folder, a folder of prediction files or a single "
                         "prediction file (default: predictions/task2, shipped with the repo)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--public-url", default=None,
                    help="the address annotators reach this server at, e.g. "
                         "https://annotate.example.org -- personal links are built from it")
    ap.add_argument("--set-admin-password", action="store_true",
                    help="ask for a new admin password in the terminal, store it, and exit "
                         "(for when it is forgotten)")
    args = ap.parse_args(argv)

    try:
        db.init(args.db)
    except db.OldDatabase as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.set_admin_password:
        import getpass
        new = getpass.getpass("new admin password: ")
        if new != getpass.getpass("again: "):
            print("the two did not match", file=sys.stderr)
            return 2
        con = db.connect(args.db)
        try:
            adminauth.set_password(con, new)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        finally:
            con.close()
        print("admin password changed for %s; every admin session was logged out" % args.db)
        return 0
    Handler.store = Store(args.root, args.db, args.star, args.task2, args.dataset_py)
    Handler.store.public_url = (args.public_url or "").rstrip("/") or None
    if not Handler.store.task_stats:
        threading.Thread(target=Handler.store.compute_task_stats, daemon=True).start()
    try:
        srv = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        print("cannot listen on port %d: %s" % (args.port, exc), file=sys.stderr)
        print("Something else is already using it -- stop that, or pass a "
              "different --port.", file=sys.stderr)
        return 2
    # flush: stdout is block-buffered when redirected, so without this nothing
    # reaches a log file until the process exits.
    print("annotool serving %s (%d tasks, %d conversations drawn so far)"
          % (Handler.store.root, len(Handler.store.tasks), len(Handler.store.items)),
          flush=True)
    print("  annotators : %s/a/<token>" % (Handler.store.public_url or
                                           "http://<host>:%d" % args.port), flush=True)
    con = db.connect(args.db)
    default = adminauth.using_default(con)
    con.close()
    print("  admin      : http://<host>:%d/admin%s" % (args.port,
          "   (first login: '%s' -- you will be asked to change it)"
          % adminauth.DEFAULT_PASSWORD if default else ""), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
