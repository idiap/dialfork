# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""DEV TOOL -- fills a scratch database with synthetic annotations.

Not part of the study. Its only purpose is to exercise analyze.py end to end so
the tables can be checked for shape before real annotators start, and to sanity
check that a plausible annotator would produce the expected signs.

  python3 -m annotool.simulate --db /tmp/sim.sqlite --annotators 4

Batches are drawn the way the admin page draws them: each annotator gets one or
two tasks, one batch per task; then a few batches also go to a second person
working the same task, which is how conversations end up judged twice.
"""
import argparse
import json
import os
import random
import sys

from . import db, sampling
from .dialogs import load_dialog, system_turn_indices
from .server import Store, give, sources_of

# A plausible annotator: conformance falls as the fork moves earlier.
SLIDER_MEAN = {None: 86, 4: 66, 3: 52, 2: 36, 1: 21}
DEFAULT_TASKS = "hotel_book,party_plan,doctor_schedule,weather"


def simulate(root, db_path, n_annotators, seed, skill, tasks, null=False, n_double=3,
             tasks_per_person=2, answer_existing=False):
    rng = random.Random(seed)
    db.init(db_path)
    store = Store(root, db_path, "datasets/STAR/tasks")
    con = db.connect(db_path)

    # people already in the DB (e.g. a demo's test users) keep their batches;
    # with answer_existing their unanswered conversations get simulated too
    existing = list(db.list_annotators(con)) if answer_existing else []
    people = []
    for n in range(n_annotators):
        aid, _ = db.create_annotator(con, "Sim %d" % (n + 1))
        db.set_consent(con, aid)
        people.append(db.annotator_by_token(
            con, con.execute("SELECT token FROM annotators WHERE id=?",
                             (aid,)).fetchone()["token"]))
    # one or two tasks each, rotating so every task gets people
    made = []
    for n, ann in enumerate(people):
        for j in range(tasks_per_person):
            task = tasks[(n + j) % len(tasks)]
            avoid = sources_of(store, db.items_assigned_to(con, ann["id"]))
            picked, new = sampling.draw_batch(
                store.root, task, sampling.BLOCK // sampling.MINI, store.by_ref,
                db.batch_counts(con),
                random.Random(rng.random()), store.pools, store.domain_tasks,
                avoid_sources=avoid, complete_quartets=True)
            store.register(con, new)
            name = db.next_batch_name(con, task)
            b = db.create_batch(con, name, task, [i["item_id"] for i in picked], 0)
            give(con, ann, b)
            made.append(b)
    # hand a few batches to a SECOND person, for agreement
    for b in made[:n_double]:
        srcs = sources_of(store, b["items"])
        holders = {r["annotator_id"] for r in db.all_assignments(con)
                   if r["batch"] == b["name"]}
        free = [a for a in people if a["id"] not in holders and not (
            sources_of(store, db.items_assigned_to(con, a["id"])) & srcs)]
        if free:
            give(con, rng.choice(free), b)

    noise = 22.0 * (1.0 - skill)
    items = store.items
    for ann in existing + people:
        aid = ann["id"]
        db.set_consent(con, aid)
        done = db.items_done_by(con, aid)
        for asg in db.assignments_for(con, aid):
            for item_id in json.loads(asg["order_json"]):
                if item_id in done:
                    continue
                done.add(item_id)
                it = items[item_id]
                # step 1: conformance slider
                mu = 50 if null else SLIDER_MEAN[it["segment"]]
                v = int(max(0, min(100, rng.gauss(mu, noise))))
                db.save_response(con, asg["id"], aid, item_id, 1, {"slider": v},
                                 int(rng.uniform(25000, 70000)))
                # step 2: does the cued behaviour appear?
                if it["is_foil"]:
                    p = [0.03, 0.09, 0.88]
                else:
                    hit = 0.90 - (0.10 if it["segment"] == 4 else 0.0)
                    p = [hit * skill, (1 - hit * skill) * 0.6, (1 - hit * skill) * 0.4]
                appears = rng.choices(["yes", "partly", "no"], weights=p)[0]
                onset = None
                if appears != "no":
                    if it["fork_assistant_turn"]:
                        n_sys = len(system_turn_indices(load_dialog(
                            os.path.join(store.root, it["path"]))))
                        onset = it["fork_assistant_turn"] + int(round(rng.gauss(0, 1.1 * (2 - skill))))
                        onset = max(1, min(n_sys, onset))
                    else:
                        onset = max(1, int(rng.uniform(1, 6)))
                db.save_response(con, asg["id"], aid, item_id, 2,
                                 {"appears": appears, "onset_sys_turn": onset},
                                 int(rng.uniform(30000, 90000)))
                # step 3
                pre = ("n/a" if not sampling.has_prefix(it)
                       else rng.choices(["yes", "mostly", "no"], weights=[0.83, 0.14, 0.03])[0])
                note = ""
                if pre == "no" or (appears == "no" and not it["is_foil"]):
                    note = rng.choice([
                        "the assistant looks fine here, I could not see the described behaviour",
                        "it drifts earlier than the highlighted point",
                        "hard to tell the difference from the normal procedure"])
                db.save_response(con, asg["id"], aid, item_id, 3,
                                 {"prefix_ok": pre, "note": note},
                                 int(rng.uniform(15000, 45000)))
    con.close()
    return len(people)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/dialfork")
    ap.add_argument("--db", required=True, help="scratch DB path -- never the real one")
    ap.add_argument("--annotators", type=int, default=4)
    ap.add_argument("--tasks", default=DEFAULT_TASKS, help="comma-separated")
    ap.add_argument("--tasks-per-person", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skill", type=float, default=0.92,
                    help="1.0 = ideal annotator, 0.0 = pure noise")
    ap.add_argument("--double", type=int, default=3,
                    help="how many batches also go to a second annotator")
    ap.add_argument("--answer-existing", action="store_true",
                    help="also answer the batches of annotators already in the DB")
    ap.add_argument("--null", action="store_true",
                    help="slider independent of fork segment; the ordering tau-b "
                         "must come out near 0 or the statistic is meaningless")
    args = ap.parse_args(argv)
    if args.db.endswith("annot.sqlite"):
        print("refusing to write synthetic data into the real database", file=sys.stderr)
        return 2
    n = simulate(args.root, args.db, args.annotators, args.seed, args.skill,
                 args.tasks.split(","), args.null, args.double, args.tasks_per_person,
                 args.answer_existing)
    print("simulated %d annotators into %s" % (n, args.db))
    return 0


if __name__ == "__main__":
    sys.exit(main())
