# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Assertions on the sampler and the dataset access it relies on.

  python3 -m annotool.tests [dataset root]

Draws batches into a scratch DB -- nothing is written anywhere else.
"""
import os
import random
import shutil
import sys
import tempfile
from collections import Counter

from . import db, sampling
from .dialogs import (HALLUCINATION_TYPES, SEGMENTS, TYPE_ABBREV, cue_sentence,
                      list_parent_ids, list_tasks, load_dialog, parent_path,
                      resolve_fork_index, substitute_workflow, was_injected)
from .server import Store
from .workflow import TASK_DESCRIPTIONS, validate_flow

FAILS = []


def check(cond, msg):
    if cond:
        print("  ok   %s" % msg)
    else:
        print("  FAIL %s" % msg)
        FAILS.append(msg)


def draw(store, con, task, minis=4, avoid=(), complete=False, seed=1):
    picked, new = sampling.draw_batch(
        store.root, task, minis, store.by_ref, db.batch_counts(con),
        random.Random(seed), store.pools, store.domain_tasks,
        avoid_sources=avoid, complete_quartets=complete)
    store.register(con, new)
    name = db.next_batch_name(con, task)
    db.create_batch(con, name, task, [i["item_id"] for i in picked], seed)
    return name, picked


def main(argv=None):
    argv = list(argv or sys.argv[1:])
    root = argv[0] if argv else "datasets/dialfork"
    tmp = tempfile.mkdtemp(prefix="annotool-tests-")
    try:
        dbp = os.path.join(tmp, "t.sqlite")
        db.init(dbp)
        store = Store(root, dbp, "datasets/STAR/tasks")
        con = db.connect(dbp)
        run(store, con, root)
        con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n%s  (%d failures)" % ("ALL CHECKS PASSED" if not FAILS else "FAILURES", len(FAILS)))
    return 1 if FAILS else 0


def run(store, con, root):
    print("\n[1] a batch is one task, balanced, one conversation per source dialogue")
    name, b1 = draw(store, con, "hotel_book", minis=8)
    check(name == "hotel_book-01", "the task is in the batch name (%s)" % name)
    check(len(b1) == 40 and {i["task"] for i in b1} == {"hotel_book"},
          "2 blocks = 40 conversations, all of the one task")
    check(len({i["source_id"] for i in b1}) == len(b1),
          "no two conversations share a source dialogue")
    cells = Counter((i["hallucination_type"], i["segment"]) for i in b1)
    check(all(cells[(h, k)] == 2 for h in HALLUCINATION_TYPES for k in SEGMENTS),
          "every (type x fork segment) cell appears once per block")
    check(sum(1 for i in b1 if i["kind"] == "conformant") == 8,
          "4 conformant per block = 20%")
    minis = [b1[m:m + sampling.MINI] for m in range(0, len(b1), sampling.MINI)]
    check(all(sum(1 for i in r if i["kind"] == "conformant") == 1 and
              len({i["hallucination_type"] for i in r if i["kind"] == "fork"}) == 4 and
              len({i["segment"] for i in r if i["kind"] == "fork"}) == 4 for r in minis),
          "items are listed mini-block by mini-block: each run of 5 is 4 forks, every "
          "type once and every k once, + 1 conformant")
    for m in (1, 2, 3, 5, 6, 7):
        rows = sampling.batch_cells(m, random.Random(m))
        types = Counter(c[0] for r in rows for c in r if c)
        ks = Counter(c[1] for r in rows for c in r if c)
        if not (len(rows) == m and set(types.values()) == {m} and set(ks.values()) == {m}
                and len(set(c for r in rows[:4] for c in r if c)) == 4 * min(m, 4)):
            check(False, "batch_cells(%d) is balanced" % m)
            break
    else:
        check(True, "any number of mini-blocks is balanced on type and on k, and no (type x k) "
              "cell repeats within a block of four")
    check(all(len(i["item_id"]) == 12 and i["ref"] not in i["item_id"] for i in b1),
          "item ids are opaque tokens; the descriptive ref stays server-side")

    print("\n[2] the items are what the files on disk say")
    bad = []
    for it in b1:
        d = load_dialog(os.path.join(store.root, it["path"]))
        a = d["annotations"]
        if it["kind"] == "fork":
            if (resolve_fork_index(d) is None or not was_injected(d)
                    or a["fork_assistant_turn"] != it["fork_assistant_turn"]
                    or a["hallucination_type"] != it["hallucination_type"]):
                bad.append(it["item_id"])
        elif a["hallucination_type"] is not None:
            bad.append(it["item_id"])
    check(not bad, "metadata matches the files; no never-injected fork (%d bad)" % len(bad))
    collide = [i for i in b1 if i["is_foil"] and i["cue_type"] == i["hallucination_type"]]
    check(not collide, "foil cues never collide with the true type")
    check(all(cue_sentence(i["cue_type"], i["cue_variant"], store.pools) for i in b1),
          "every item resolves a cue sentence")
    check(all(i["is_foil"] for i in b1 if i["kind"] == "conformant"),
          "conformant conversations are always foils")
    same = all(sampling.make_fork_item(store.root, i["task"], i["hallucination_type"],
                                       i["segment"], i["dialog_id"], store.pools,
                                       store.domain_tasks)
               == {k: v for k, v in i.items() if k != "item_id"}
               for i in b1 if i["kind"] == "fork")
    check(same, "an item's foil is deterministic -- rebuilding it gives the same record")

    print("\n[3] a batch for anyone takes fresh sources; one for a person completes quartets")
    _, b2 = draw(store, con, "hotel_book", seed=2)
    _, small = draw(store, con, "hotel_book", minis=1, seed=9)
    check(len(small) == 5 and not ({i["source_id"] for i in small} &
                                   {i["source_id"] for i in b1 + b2}),
          "a batch can be a single mini-block of 5")
    check(not ({i["source_id"] for i in b2} & {i["source_id"] for i in b1}),
          "a batch for anyone uses sources nobody holds, so everyone is eligible")
    held = {i["source_id"] for i in b1}
    _, b3 = draw(store, con, "hotel_book", avoid=held, complete=True, seed=3)
    check(not ({i["source_id"] for i in b3} & held),
          "a batch for a person never uses a source that person holds")
    started = {i["quartet_id"] for i in b1 + b2 if i["quartet_id"]}
    check(sum(1 for i in b3 if i["quartet_id"] in started) >= 12,
          "... and continues quartets other people started (%d of 16 forks)"
          % sum(1 for i in b3 if i["quartet_id"] in started))
    reused = {i["item_id"] for i in b3} & {i["item_id"] for i in b1 + b2 + small}
    check(not reused, "while unseen conversations remain, none is drawn twice")

    print("\n[4] tasks that cannot supply a batch say so")
    try:
        draw(store, con, "ride_status")
        check(False, "ride_status (9 sources) is refused a batch of 20")
    except sampling.DrawError as exc:
        check("9 source" in str(exc), "ride_status (9 sources) is refused a batch of 20, "
              "with the reason")
    _, rs = draw(store, con, "ride_status", minis=1)
    check(len(rs) == 5, "... but can supply a mini-block")

    print("\n[5] workflow steps, descriptions, WIC substitutes")
    problems, missing = [], []
    for task in list_tasks(root):
        d = load_dialog(parent_path(root, task, list_parent_ids(root, task)[0]))
        problems += validate_flow(task, d["annotations"]["task"]["graph"],
                                  d["annotations"]["task"]["replies"])
        if task not in TASK_DESCRIPTIONS:
            missing.append(task)
    check(not problems, "every task's step description matches its graph: every reply "
          "covered, every edge and branch where the graph puts it (%s)" % (problems[:3] or "-"))
    check(not missing, "every task has a one-line description (%s)" % (missing or "-"))
    wic = [i for i in store.items.values()
           if i["hallucination_type"] == "workflow_identity_corruption"]
    bad_wic = [i for i in wic
               if not substitute_workflow(load_dialog(os.path.join(store.root, i["path"])))]
    check(wic and not bad_wic, "every drawn WIC fork yields its substitute workflow "
          "from the injection event (%d/%d bad)" % (len(bad_wic), len(wic)))
    sub_problems = []
    for i in wic:
        name, g, r = substitute_workflow(load_dialog(os.path.join(store.root, i["path"])))
        sub_problems += validate_flow(name, g, r)
    check(not sub_problems, "and each substitute's step description matches it too (%s)"
          % (sub_problems[:3] or "-"))
    # the two substitutes that are not DialFork tasks only exist inside WIC
    # injections, so find them there
    import glob
    from .workflow import parse_wic_injection
    extra, wanted = {}, {"trivia_score_reporting", "weather_alerts"}
    for task in list_tasks(root):
        for f in sorted(glob.glob(os.path.join(
                root, task, "workflow_identity_corruption", "n_hallucinations_4",
                "fork_at_segment_1", "*.json")))[:60]:
            d = load_dialog(f)
            ev = [e for e in d.get("events") or [] if e.get("action") == "instruct-persist"]
            if ev:
                name, g, r = parse_wic_injection(ev[0]["content"])
                if name in wanted and name not in extra:
                    extra[name] = validate_flow(name, g, r)
        if set(extra) == wanted:
            break
    check(set(extra) == wanted and not any(extra.values()),
          "the substitute-only workflows (%s) are described and match too %s"
          % (", ".join(sorted(wanted)), [p for v in extra.values() for p in v][:3] or ""))

    print("\n[6] the registry survives a restart")
    fresh = Store(root, store.db_path, "datasets/STAR/tasks")
    check(set(fresh.items) == set(store.items),
          "items drawn so far are reloaded from the DB (%d)" % len(fresh.items))


if __name__ == "__main__":
    sys.exit(main())
