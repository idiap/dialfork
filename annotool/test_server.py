# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""End-to-end server test, including the blinding assertions.

Starts a server on a scratch DB, drives the API, checks that the payloads do
not leak ground truth ahead of the step that is allowed to show it, then exits.

  python3 -m annotool.test_server
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from . import db
from .sampling import has_prefix
from .server import Handler, Store

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


# The admin session is a cookie: `post`/`get` share one browser-like jar (the
# logged-in admin); `bare` is a client with no cookies at all.
import http.cookiejar
JAR = http.cookiejar.CookieJar()
ADMIN_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(JAR))
BARE_OPENER = urllib.request.build_opener()


def _strict(text):
    """Parse like the browser does: Python accepts NaN/Infinity, JSON.parse does not."""
    def refuse(c):
        raise ValueError("not JSON: %s (JSON.parse in the browser fails here)" % c)
    return json.loads(text, parse_constant=refuse)


def _post(opener, base, path, body):
    req = urllib.request.Request(base + path, method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with opener.open(req) as r:
            return r.status, _strict(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, _strict(e.read().decode())


def post(base, path, body):
    return _post(ADMIN_OPENER, base, path, body)


def bare(base, path, body):
    return _post(BARE_OPENER, base, path, body)


def get(base, path, opener=None):
    try:
        with (opener or ADMIN_OPENER).open(base + path) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main(argv=None):
    argv = list(argv or sys.argv[1:])
    root = argv[0] if argv else "datasets/dialfork"
    tmp = tempfile.mkdtemp(prefix="annotool-test-")
    dbp = os.path.join(tmp, "t.sqlite")
    admin = db.init(dbp)
    Handler.store = Store(root, dbp, "datasets/STAR/tasks", "predictions/task2")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = "http://127.0.0.1:%d" % srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        run(base, admin, dbp)
    finally:
        srv.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n%s  (%d failures)" % ("ALL CHECKS PASSED" if not FAILS else "FAILURES", len(FAILS)))
    return 1 if FAILS else 0


def admin_login(base, dbp):
    """The admin panel sits behind a password; the default only opens the form
    to change it."""
    from . import adminauth
    code, _ = bare(base, "/api/admin/stats", {})
    check(code == 401, "without a session the admin API is refused")
    code, body = get(base, "/admin", BARE_OPENER)
    check(code == 200 and b"Admin login" in body, "/admin shows the login page first")
    code, body = get(base, "/admin?t=anything", BARE_OPENER)
    check(b"Admin login" in body, "the old ?t= token in the URL opens nothing")
    code, _ = post(base, "/api/admin/login", {"password": "wrong-password"})
    check(code == 401, "a wrong password is refused")
    code, j = post(base, "/api/admin/login", {"password": adminauth.DEFAULT_PASSWORD})
    check(code == 200 and j["must_change"] is True, "the default password logs in once...")
    code, j = post(base, "/api/admin/stats", {})
    check(code == 403 and j.get("must_change"),
          "...but opens nothing except the change-password form")
    code, _ = post(base, "/api/admin/password", {"current": adminauth.DEFAULT_PASSWORD,
                                                 "new": "short"})
    check(code == 400, "a too-short new password is refused")
    code, _ = post(base, "/api/admin/password", {"current": adminauth.DEFAULT_PASSWORD,
                                                 "new": adminauth.DEFAULT_PASSWORD})
    check(code == 400, "keeping the public default is refused")
    # a second admin browser, logged in before the change, must be logged out by it
    other = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    _post(other, base, "/api/admin/login", {"password": adminauth.DEFAULT_PASSWORD})
    code, _ = post(base, "/api/admin/password", {"current": adminauth.DEFAULT_PASSWORD,
                                                 "new": "a-good-new-password"})
    check(code == 200, "a proper new password is accepted")
    code, _ = post(base, "/api/admin/stats", {})
    check(code == 200, "and this session goes straight on")
    code, _ = _post(other, base, "/api/admin/stats", {})
    check(code == 401, "while every other admin session is logged out")
    code, _ = post(base, "/api/admin/login", {"password": adminauth.DEFAULT_PASSWORD})
    check(code == 401, "the default no longer works")
    code, body = get(base, "/admin")
    check(code == 200 and b"Admin login" not in body, "logged in, /admin is the panel")
    con = db.connect(dbp)
    stored = db.meta_get(con, "admin_pw")
    con.close()
    check("a-good-new-password" not in stored and stored.startswith("pbkdf2_sha256$"),
          "the password is stored hashed and salted, never in clear")
    now = __import__("time").time()
    adminauth._fails["203.0.113.9"] = [now] * adminauth.MAX_FAILS
    check(adminauth.locked_out("203.0.113.9"),
          "too many wrong passwords lock the address out for a while")


def page(base, path, prefix=None, opener=None):
    req = urllib.request.Request(base + path)
    if prefix is not None:
        req.add_header("X-Forwarded-Prefix", prefix)
    with (opener or ADMIN_OPENER).open(req) as r:
        return r.read()


def proxy_prefix(base, opener):
    """Behind a reverse proxy that strips /criteria/forkbench, the pages point
    the browser back under it; accessed directly they are served as written."""
    import re
    from .server import TEMPLATES
    pre = "/criteria/forkbench"
    for path, tpl in [("/a/whatever", "annotate.html"),
                      ("/admin", "admin.html" if opener is ADMIN_OPENER else "login.html")]:
        with open(os.path.join(TEMPLATES, tpl), "rb") as fh:
            raw = fh.read()
        check(page(base, path, opener=opener) == raw,
              "%s without X-Forwarded-Prefix is the template unchanged" % tpl)
        html = page(base, path, pre, opener).decode()
        urls = re.findall(r'(?:href|src)="(/[^"]*)"', html)
        check(urls and all(u.startswith(pre + "/static/") for u in urls),
              "%s behind the proxy: every href/src is under the prefix" % tpl)
        check(('window.BASE = "%s";' % pre) in html
              and html.index("window.BASE") < html.index("<script src="),
              "%s sets window.BASE before the other scripts" % tpl)
        check(page(base, path, pre + "/", opener).decode() == html,
              "%s: a trailing slash on the prefix is dropped" % tpl)
        for bad in ['/x"></script><script>alert(1)</script>', "/a<b", "/a'b", "x/y",
                    "//evil.example", "/a/../b"]:
            check(page(base, path, bad, opener) == raw,
                  "%s: an unsafe prefix %r is ignored" % (tpl, bad))


def run(base, admin, dbp):
    store = Handler.store
    print("\n[4a] auth and admin")
    code, _ = post(base, "/api/state", {"token": "nope"})
    check(code == 403, "an unknown token is rejected")
    proxy_prefix(base, BARE_OPENER)
    admin_login(base, dbp)
    proxy_prefix(base, ADMIN_OPENER)
    _, r1 = post(base, "/api/admin/annotator_add", {"admin_token": admin, "name": "Ann One"})
    a1 = r1["created_annotator"]
    _, r2 = post(base, "/api/admin/annotator_add", {"admin_token": admin, "name": "Ann Two"})
    a2 = r2["created_annotator"]
    check("token" in a1 and a1["token"] != a2["token"], "annotators get distinct tokens")
    _, j = post(base, "/api/admin/stats", {"admin_token": admin})
    check(j["batches"] == [] and j["n_items"] == 0,
          "a fresh database starts with no batches and nothing drawn")
    code, j = post(base, "/api/admin/batch_add", {"admin_token": admin})
    check(code == 400, "a batch needs a task")
    code, j = post(base, "/api/admin/batch_add",
                   {"admin_token": admin, "task": "hotel_book", "size": 20})
    B0 = j["created_batch"]["name"]
    check(code == 200 and B0 == "hotel_book-01" and j["created_batch"]["size"] == 20,
          "a batch is drawn on demand, per task, 20 conversations by default (%s)" % B0)
    for a in (a1, a2):
        post(base, "/api/admin/assign",
             {"admin_token": admin, "annotator_id": a["id"], "batch": B0})
    t1, t2 = a1["token"], a2["token"]

    print("\n[4b] consent gate")
    _, s = post(base, "/api/state", {"token": t1})
    check(s["annotator"]["consented"] is False, "a new annotator has not consented")
    post(base, "/api/consent", {"token": t1})
    post(base, "/api/consent", {"token": t2})
    _, s1 = post(base, "/api/state", {"token": t1})
    check(s1["annotator"]["consented"] is True and s1["current"], "consent unlocks work")

    print("\n[4b2] learning the task")
    check(s1["current"]["need_intro"] is True and s1["tasks"][0]["task"] == "hotel_book",
          "the first conversation of a new task asks for its intro first")
    code, intro = post(base, "/api/intro", {"token": t1, "task": "hotel_book"})
    check(code == 200 and intro["flow"]["steps"] and "turns" not in intro,
          "the intro carries the task's steps and nothing about any conversation")
    code, _ = post(base, "/api/intro", {"token": t1, "task": "weather"})
    check(code == 403, "an intro for a task you do not hold is refused")
    code, s1 = post(base, "/api/intro_done",
                    {"token": t1, "task": "hotel_book", "action": "read", "ms": 42000})
    check(code == 200 and s1["current"]["need_intro"] is False and
          s1["tasks"][0]["intro"]["action"] == "read",
          "reading it is recorded, and the conversations follow")
    code, _ = post(base, "/api/intro_done",
                   {"token": t1, "task": "hotel_book", "action": "cheat"})
    check(code == 400, "only read / skipped / reviewed are accepted")
    post(base, "/api/intro_done", {"token": t2, "task": "hotel_book", "action": "skipped"})
    _, s2x = post(base, "/api/state", {"token": t2})
    check(s2x["tasks"][0]["intro"]["action"] == "skipped", "skipping is recorded too")

    print("\n[4c] presentation order is per-annotator")
    _, s2 = post(base, "/api/state", {"token": t2})
    con = db.connect(dbp)
    o1 = json.loads(db.assignments_for(con, a1["id"])[0]["order_json"])
    o2 = json.loads(db.assignments_for(con, a2["id"])[0]["order_json"])
    con.close()
    check(sorted(o1) == sorted(o2) and o1 != o2,
          "same batch, different randomised order for the two annotators")

    print("\n[4d] blinding: the payload must not leak ahead of the step")
    item_id = s1["current"]["item_id"]
    meta = store.items[item_id]
    _, p1 = post(base, "/api/item", {"token": t1, "item_id": item_id})
    check(p1["step"] == 1, "first serve is step 1")
    blob1 = json.dumps(p1)
    check("cue" not in p1, "step 1 carries no cue field")
    check("reveal" not in p1, "step 1 carries no reveal field")
    check(meta["cue_variant"] not in blob1, "step-1 payload does not contain the cue variant id")
    check("fork_assistant_turn" not in blob1 and "fork_turn_index" not in blob1,
          "step-1 payload does not mention the fork turn (fork_assistant_turn, fork_turn_index)")
    _, st1 = post(base, "/api/state", {"token": t1})
    check(meta["ref"] not in blob1 and meta["ref"] not in json.dumps(st1),
          "neither the item nor the state payload carries the descriptive ref")
    check("/" not in item_id and not any(
              ("/%s/" % x) in item_id for x in ("WIC", "SSPC", "GR", "CSC", "conformant")),
          "the item id the browser sees encodes neither type nor fork position")

    code, _ = post(base, "/api/answer", {"token": t1, "item_id": item_id, "step": 3,
                                         "payload": {"prefix_ok": "yes"}})
    check(code == 409, "cannot jump straight to step 3")
    code, _ = post(base, "/api/answer", {"token": t1, "item_id": item_id, "step": 1,
                                         "payload": {"slider": 999}})
    check(code == 400, "an out-of-range slider is rejected")

    post(base, "/api/answer", {"token": t1, "item_id": item_id, "step": 1,
                               "payload": {"slider": 62}, "ms": 5000})
    _, p2 = post(base, "/api/item", {"token": t1, "item_id": item_id})
    blob2 = json.dumps(p2)
    check(p2["step"] == 2 and p2.get("cue"), "step 2 carries the cue")
    check("reveal" not in p2, "step 2 carries no reveal field")
    check("fork_assistant_turn" not in blob2 and "fork_turn_index" not in blob2,
          "step-2 payload does not mention the fork turn (fork_assistant_turn, fork_turn_index)")
    if meta["fork_assistant_turn"]:
        check(('"%d"' % meta["fork_assistant_turn"]) not in blob2, "step-2 payload does not leak the fork turn")

    code, _ = post(base, "/api/answer", {"token": t1, "item_id": item_id, "step": 2,
                                         "payload": {"appears": "yes"}})
    check(code == 400, "step 2 requires an onset turn when the behaviour appears")
    post(base, "/api/answer", {"token": t1, "item_id": item_id, "step": 2,
                               "payload": {"appears": "yes", "onset_sys_turn": 2}, "ms": 9000})
    _, p3 = post(base, "/api/item", {"token": t1, "item_id": item_id})
    check(p3["step"] == 3 and "reveal" in p3, "step 3 carries the reveal")
    check("yours" not in p1 and "yours" not in p2 and
          p3.get("yours") == {"appears": "yes", "onset": 2},
          "step 3 (only) echoes the annotator's own step-2 answer, for the side-by-side")
    check(p3["reveal"]["fork_assistant_turn"] == meta["fork_assistant_turn"],
          "step 3 discloses the true fork turn (banding happens only here)")

    # the prefix question applies only when there is something before the fork
    right, wrong = ("yes", "n/a") if has_prefix(meta) else ("n/a", "yes")
    code, _ = post(base, "/api/answer", {"token": t1, "item_id": item_id, "step": 3,
                                         "payload": {"prefix_ok": wrong}})
    check(code == 400, "step 3 insists on n/a exactly when there is nothing before the "
          "fork (conformant, or injected before turn 1) -- here %r is refused" % wrong)
    k1 = [i for i in store.items.values() if i["kind"] == "fork" and i["fork_assistant_turn"] == 1]
    check(k1 and all(not has_prefix(i) for i in k1) and
          all(has_prefix(i) for i in store.items.values()
              if i["kind"] == "fork" and i["fork_assistant_turn"] > 1),
          "every segment-1 fork (injected before turn 1) has no prefix to judge")

    print("\n[4e] persistence and resume")
    post(base, "/api/answer", {"token": t1, "item_id": item_id, "step": 3,
                               "payload": {"prefix_ok": right, "note": "n"}, "ms": 3000})
    _, s = post(base, "/api/state", {"token": t1})
    check(s["assignments"][0]["done"] == 1, "a completed item counts towards progress")
    check(s["current"]["item_id"] != item_id, "the next item is served after completion")
    code, _ = post(base, "/api/answer", {"token": t1, "item_id": item_id, "step": 3,
                                         "payload": {"prefix_ok": "no"}})
    check(code == 409, "a completed item cannot be re-answered")

    print("\n[4e2] chocolates")
    c = s["chocolates"]
    check(c == {"count": 0, "next": 0.5, "total": 10},
          "by default a person's allowance is 10: one of 20 conversations earns half")
    code, j = post(base, "/api/admin/annotator_chocolates",
                   {"admin_token": admin, "annotator_id": a1["id"], "chocolates": 20})
    _, s = post(base, "/api/state", {"token": t1})
    check(code == 200 and s["chocolates"] == {"count": 1, "next": 0, "total": 20},
          "the admin sets each person's allowance, and it applies to what was already earned")
    _, s2 = post(base, "/api/state", {"token": t2})
    check(s2["chocolates"]["total"] == 10, "... for that person only")
    mine = [a for a in j["annotators"] if a["id"] == a1["id"]][0]
    check(mine["chocolates"] == 1 and mine["allowance"] == 20,
          "the admin panel shows what each person has earned and their allowance")
    check(j["chocolates_allowed"] == 30 and j["chocolates_owed"] == 1,
          "and the totals: owed if everyone finishes, earned so far")
    for bad in (-1, "ten", None, 1001, 2.5, True):
        code, _ = post(base, "/api/admin/annotator_chocolates",
                       {"admin_token": admin, "annotator_id": a1["id"], "chocolates": bad})
        check(code == 400, "an allowance of %r is refused" % (bad,))
    code, _ = bare(base, "/api/admin/annotator_chocolates",
                   {"annotator_id": a1["id"], "chocolates": 99})
    check(code in (401, 403), "only the admin can change an allowance")
    post(base, "/api/admin/annotator_chocolates",
         {"admin_token": admin, "annotator_id": a1["id"], "chocolates": 10})
    code, j = post(base, "/api/admin/annotator_add",
                   {"admin_token": admin, "name": "Choc Test", "chocolates": 5})
    ct = j["created_annotator"]
    check(code == 200 and [a["allowance"] for a in j["annotators"]
                           if a["id"] == ct["id"]] == [5],
          "an allowance can be set when the person is added")
    post(base, "/api/admin/annotator_delete", {"admin_token": admin, "annotator_id": ct["id"]})
    from .server import chocolates
    three = [{"done": 5, "total": 5}, {"done": 10, "total": 15}]
    check(chocolates(three, 15) == {"count": 11, "next": 0.25, "total": 15},
          "the allowance is paid over everything a person holds: 15 of 20 done -> 11.25")
    check(chocolates([{"done": 7, "total": 7}, {"done": 13, "total": 13}], 15)["count"] == 15,
          "finishing everything pays exactly the allowance (no rounding loss)")

    print("\n[4e3] batch sizes and presentation order")
    for bad in (0, 3, 7, 105, "x"):
        code, _ = post(base, "/api/admin/batch_add",
                       {"admin_token": admin, "task": "hotel_search", "size": bad})
        check(code == 400, "a batch of %r conversations is refused" % (bad,))
    code, j = post(base, "/api/admin/batch_add",
                   {"admin_token": admin, "task": "hotel_search", "size": 15,
                    "annotator_id": a2["id"]})
    small = j["created_batch"]["name"]
    check(code == 200 and j["created_batch"]["size"] == 15 and
          j["created_batch"]["for"] == "Ann Two", "a batch of 15, given to a person")
    con = db.connect(dbp)
    order = [json.loads(r["order_json"]) for r in db.all_assignments(con)
             if r["batch"] == small][0]
    batch_items = db.batch(con, small)["items"]
    con.close()
    runs_ok = all(set(order[i:i + 5]) in [set(batch_items[m:m + 5]) for m in (0, 5, 10)]
                  for i in (0, 5, 10))
    check(runs_ok, "it is presented one mini-block at a time")
    ok = True
    for i in (0, 5, 10):
        run = [store.items[x] for x in order[i:i + 5]]
        forks = [x for x in run if x["kind"] == "fork"]
        ok &= (len(forks) == 4 and len({x["hallucination_type"] for x in forks}) == 4
               and len({x["segment"] for x in forks}) == 4)
    check(ok, "and every run of 5 is 4 forks (each type once, each k once) + 1 conformant")
    post(base, "/api/admin/batch_delete", {"admin_token": admin, "batch": small})

    _, j = post(base, "/api/admin/batch_add", {"admin_token": admin, "task": "party_rsvp"})
    other = [i for i in store.items if i not in o1][0]
    code, _ = post(base, "/api/item", {"token": t1, "item_id": other})
    check(code == 403, "an item outside your batch is refused")

    print("\n[4g] batch management")
    _, _r = post(base, "/api/admin/annotator_add", {"admin_token": admin, "name": "Ann Three"})
    a3 = _r["created_annotator"]
    code, j = post(base, "/api/admin/assign",
                   {"admin_token": admin, "annotator_id": a3["id"], "batch": B0})
    check(code == 200, "a batch can go to a third person")
    # a batch for a named person: drawn around what they hold, assigned at once
    code, j = post(base, "/api/admin/batch_add",
                   {"admin_token": admin, "task": "hotel_book", "annotator_id": a3["id"]})
    mine = j.get("created_batch") or {}
    check(code == 200 and mine.get("for") == "Ann Three",
          "a batch can be drawn for a named person and is given to them")
    con2 = db.connect(dbp)
    srcs = lambda ids: {store.items[i]["source_id"] for i in ids}
    check(not (srcs(db.batch(con2, mine["name"])["items"]) & srcs(db.batch(con2, B0)["items"])),
          "it avoids every source dialogue that person already holds")
    # a newcomer's batch continues quartets B0 started, so it shares B0's
    # sources -- and whoever holds B0 must then be refused it
    _, _r = post(base, "/api/admin/annotator_add", {"admin_token": admin, "name": "Ann New"})
    anew = _r["created_annotator"]
    _, j = post(base, "/api/admin/batch_add",
                {"admin_token": admin, "task": "hotel_book", "annotator_id": anew["id"]})
    nb = j["created_batch"]["name"]
    con2 = db.connect(dbp)
    shared = srcs(db.batch(con2, nb)["items"]) & srcs(db.batch(con2, B0)["items"])
    con2.close()
    check(shared, "a batch for a newcomer continues quartets others started")
    code, j = post(base, "/api/admin/assign",
                   {"admin_token": admin, "annotator_id": a1["id"], "batch": nb})
    check(code == 409 and "source dialogues" in (j.get("error") or ""),
          "so it is refused for someone who already holds one of its source dialogues")
    _, st = post(base, "/api/admin/stats", {"admin_token": admin})
    row = [b for b in st["batches"] if b["name"] == nb][0]
    check("Ann One" in [x["name"] for x in row["blocked"]] and
          "Ann One" not in row["eligible"],
          "and the panel lists them as not offered, with the overlap")
    check(all(set(b.keys()) >= {"task", "eligible", "blocked"} for b in st["batches"]),
          "batch rows carry their task and who is / is not eligible")

    code, j = post(base, "/api/admin/batch_add",
                   {"admin_token": admin, "task": "ride_status"})
    check(code == 409 and "source dialogue" in (j.get("error") or ""),
          "a task too small for a batch is refused with the reason")

    _, j = post(base, "/api/admin/batch_add",
                {"admin_token": admin, "task": "party_plan", "size": 20})
    made = j.get("created_batch")
    check(made and made["name"] == "party_plan-01", "names count per task")
    code, j = post(base, "/api/admin/batch_delete",
                   {"admin_token": admin, "batch": made["name"]})
    check(code == 200, "a batch with no answers in it can be deleted")
    code, _ = post(base, "/api/admin/batch_delete",
                   {"admin_token": admin, "batch": made["name"]})
    check(code == 404, "deleting it twice is refused")

    print("\n[4l] batch viewer")
    code, v = post(base, "/api/admin/batch_view", {"admin_token": admin, "batch": B0})
    check(code == 200 and len(v["items"]) == 20 and v["task"] == "hotel_book",
          "the admin can list what is inside a batch")
    code, iv = post(base, "/api/admin/item_view",
                    {"admin_token": admin, "item_id": v["items"][0]["item_id"]})
    check(code == 200 and "reveal" in iv and "cue" in iv and "answers" in iv,
          "and open one conversation with its ground truth and answers")
    code, _ = bare(base, "/api/admin/batch_view", {"token": t1, "batch": B0})
    check(code == 401, "an annotator's token cannot open the batch viewer")
    code, _ = bare(base, "/api/admin/item_view", {"token": t1, "item_id": item_id})
    check(code == 401, "nor read a conversation's ground truth through it")

    print("\n[4j] unassigning")
    _, _r = post(base, "/api/admin/annotator_add", {"admin_token": admin, "name": "Ann Five"})
    a5 = _r["created_annotator"]
    _, j = post(base, "/api/admin/batch_add",
                {"admin_token": admin, "task": "weather"})
    free = j["created_batch"]["name"]
    post(base, "/api/admin/assign",
         {"admin_token": admin, "annotator_id": a5["id"], "batch": free})
    code, j = post(base, "/api/admin/unassign",
                   {"admin_token": admin, "annotator_id": a5["id"], "batch": free})
    check(code == 200 and free in (j.get("removed", {}).get("what") or ""),
          "an untouched batch can be taken back")
    con3 = db.connect(dbp)
    check(free not in [x["batch"] for x in db.assignments_for(con3, a5["id"])],
          "the assignment is gone afterwards")
    con3.close()
    code, _ = post(base, "/api/admin/unassign",
                   {"admin_token": admin, "annotator_id": a5["id"], "batch": free})
    check(code == 404, "unassigning something they do not hold is refused")

    # a1 has answered items in its batch: removing it must warn before deleting
    b_of_a1 = B0
    code, j = post(base, "/api/admin/unassign",
                   {"admin_token": admin, "annotator_id": a1["id"], "batch": b_of_a1})
    check(code == 409 and "deletes those answers" in (j.get("error") or ""),
          "removing a batch with answers in it warns first")
    con3 = db.connect(dbp)
    before = len(db.items_done_by(con3, a1["id"]))
    con3.close()
    code, j = post(base, "/api/admin/unassign",
                   {"admin_token": admin, "annotator_id": a1["id"],
                    "batch": b_of_a1, "force": True})
    check(code == 200 and j["removed"]["answers_deleted"] >= 1,
          "forcing it through removes the batch and reports what was deleted")
    con3 = db.connect(dbp)
    check(len(db.items_done_by(con3, a1["id"])) < before or before == 0,
          "the answers really are gone")
    con3.close()

    print("\n[4h] the double-serve regression")
    # a1 finished one item earlier; give a1 a batch containing it and check it is
    # not queued a second time
    con2 = db.connect(dbp)
    done_before = db.items_done_by(con2, a1["id"])
    con2.close()
    _, s1b = post(base, "/api/state", {"token": t1})
    queued = [x["next"]["item_id"] for x in s1b["assignments"] if x["next"]]
    check(not (set(queued) & done_before),
          "no item already completed by this annotator is queued again")

    print("\n[4i] report endpoint")
    _, rep = post(base, "/api/admin/report", {"admin_token": admin, "metrics": False})
    check("summary" in rep and "markdown" in rep, "the report endpoint returns both views")
    con6 = db.connect(dbp)
    in_study = len(db.study_items(con6))
    con6.close()
    check(rep["summary"]["n_items"] == in_study < len(store.items),
          "the report counts the conversations in a batch, not those of deleted batches")
    check("# DialFork human annotation results" in rep["markdown"],
          "the markdown report renders with partial data")
    from .server import finite
    check(finite({"a": [float("nan"), 1.5, {"b": float("inf")}]}) == {"a": [None, 1.5, {"b": None}]},
          "NaN/inf in any response become null, so the browser can parse it")

    print("\n[4k] annotator lifecycle")
    _, _r = post(base, "/api/admin/annotator_add",
                 {"admin_token": admin, "name": "Temp Person"})
    tmp = _r["created_annotator"]
    code, j = post(base, "/api/admin/annotator_rename",
                   {"admin_token": admin, "annotator_id": tmp["id"], "name": "Renamed"})
    check(code == 200 and any(a["name"] == "Renamed" for a in j["annotators"]),
          "an annotator can be renamed")
    code, _ = post(base, "/api/admin/annotator_rename",
                   {"admin_token": admin, "annotator_id": tmp["id"], "name": "  "})
    check(code == 400, "renaming to blank is refused")
    code, j = post(base, "/api/admin/annotator_delete",
                   {"admin_token": admin, "annotator_id": tmp["id"]})
    check(code == 200 and not any(a["id"] == tmp["id"] for a in j["annotators"]),
          "an annotator with no answers can be deleted outright")
    # a self-contained person who has actually answered something
    _, _r = post(base, "/api/admin/annotator_add",
                 {"admin_token": admin, "name": "Worker"})
    w = _r["created_annotator"]
    _, j = post(base, "/api/admin/batch_add", {"admin_token": admin, "task": "trivia"})
    wb = j["created_batch"]["name"]
    post(base, "/api/admin/assign",
         {"admin_token": admin, "annotator_id": w["id"], "batch": wb})
    post(base, "/api/consent", {"token": w["token"]})
    _, st = post(base, "/api/state", {"token": w["token"]})
    wi = st["current"]["item_id"]
    for step, pl in ((1, {"slider": 40}),
                     (2, {"appears": "no", "onset_sys_turn": None}),
                     (3, {"prefix_ok": "yes" if has_prefix(store.items[wi]) else "n/a"})):
        post(base, "/api/answer", {"token": w["token"], "item_id": wi,
                                   "step": step, "payload": pl, "ms": 1000})
    code, j = post(base, "/api/admin/annotator_delete",
                   {"admin_token": admin, "annotator_id": w["id"]})
    check(code == 409 and "answered" in (j.get("error") or ""),
          "deleting an annotator who has answered warns first")
    code, j = post(base, "/api/admin/annotator_delete",
                   {"admin_token": admin, "annotator_id": w["id"], "force": True})
    check(code == 200 and j["removed"]["answers_deleted"] >= 1,
          "forcing it through reports how many answers were destroyed")
    con4 = db.connect(dbp)
    check(db.annotator_by_token(con4, w["token"]) is None,
          "the annotator's login link stops working afterwards")
    con4.close()

    print("\n[4o] a task is finished before the next one starts")
    _, _r = post(base, "/api/admin/annotator_add", {"name": "Orderly"})
    od = _r["created_annotator"]
    for t in ("weather", "trivia", "weather"):          # a 2nd weather block, given later
        post(base, "/api/admin/batch_add", {"task": t, "annotator_id": od["id"]})
    _, st = post(base, "/api/state", {"token": od["token"]})
    check([a["task"] for a in st["assignments"]] == ["weather", "weather", "trivia"] and
          [(t["task"], t["total"]) for t in st["tasks"]] == [("weather", 40), ("trivia", 20)],
          "a later block of a task joins that task in the queue, not the end")

    print("\n[4n] export")
    import csv as _csv, io as _io, zipfile as _zip
    code, _ = get(base, "/api/admin/export?what=csv", BARE_OPENER)
    check(code == 401, "the export needs an admin session")
    code, body = get(base, "/api/admin/export?what=csv")
    rows = list(_csv.reader(_io.StringIO(body.decode("utf-8-sig"))))
    con5 = db.connect(dbp)
    n_answered = con5.execute("SELECT COUNT(*) AS n FROM (SELECT DISTINCT item_id, "
                              "annotator_id FROM responses WHERE step=3)").fetchone()["n"]
    con5.close()
    check(code == 200 and rows[0][:3] == ["item_id", "ref", "annotator"] and
          len(rows) - 1 == n_answered,
          "answers.csv has one row per completed (conversation, annotator) (%d)" % n_answered)
    code, body = get(base, "/api/admin/export?what=zip")
    names = sorted(n.split("/", 1)[1] for n in _zip.ZipFile(_io.BytesIO(body)).namelist())
    check(code == 200 and names == ["README.txt", "answers.csv", "batches.csv", "report.md"],
          "the zip holds answers, batches, the report and a README (%s)" % names)
    code, body = get(base, "/api/admin/export?what=zip&metrics=1")
    z = _zip.ZipFile(_io.BytesIO(body))
    head = z.read([n for n in z.namelist() if n.endswith("answers.csv")][0]).decode(
        "utf-8-sig").splitlines()[0]
    has_metrics = os.path.isdir("predictions/task2")
    check(code == 200 and (not has_metrics or
          ("metric_flowscore" in head and
           any(n.endswith("metric_correlation.csv") for n in z.namelist()))),
          "with metrics: one metric_* column per Task-2 metric, plus metric_correlation.csv")

    print("\n[4m] logging out")
    other = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    code, _ = _post(other, base, "/api/admin/login", {"password": "a-good-new-password"})
    check(code == 200, "the new password logs in")
    _post(other, base, "/api/admin/logout", {})
    code, _ = _post(other, base, "/api/admin/stats", {})
    check(code == 401, "after logging out the session no longer works")

    print("\n[4f] static and pages")
    code, body = get(base, "/a/" + t1)
    check(code == 200 and b"DialFork annotation" in body, "the annotate page is served")
    code, _ = get(base, "/static/app.js")
    check(code == 200, "static assets are served")
    code, _ = get(base, "/static/../db.py")
    check(code != 200, "static path traversal is blocked")
    task = store.items[item_id]["task"]
    code, body = get(base, "/flow/%s.jpg" % task)
    check(code == 200 and (body[:2] == b"\xff\xd8" or body[:4] == b"\x89PNG"),
          "the STAR flowchart image is served (JPEG or the 2 mislabelled PNGs)")


if __name__ == "__main__":
    sys.exit(main())
