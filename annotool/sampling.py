# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Draw conversations into batches, on demand. Nothing is frozen.

A BATCH is a work unit for one person: conversations of ONE task, in
mini-blocks of 5. A mini-block is the smallest balanced unit:

     4 forks  = every hallucination type once AND every fork segment k once
                (one row of a 4x4 Latin square of type x k)
  +  1 conformant original                                     (= 20%)

Four rows of one square make a BLOCK of 20 -- each (type x k) exactly once,
plus 4 conformant -- so a 20-item batch is as balanced as it ever was, and a
5-, 10- or 15-item one is still balanced on type and on k. Batches keep their
items grouped by mini-block and are presented one mini-block at a time
(server.assignment_order), so someone who stops part-way leaves a balanced
sample behind, not a random subset.

Every conversation in a batch comes from a DIFFERENT source dialogue.

Why one task: an annotator who keeps judging against the same workflow gets
familiar with it and faster at spotting deviations from it.

Why distinct source dialogues: forks of one source share its opening, so an
annotator who saw two of them could diff them -- the onset question would
become text comparison and the slider a comparison between siblings, not a
judgement against the workflow. The same rule holds per
ANNOTATOR across all their batches; server.py enforces it at assignment.

The study therefore needs no fixed sample, annotator count or batch count: the
fork-order result is computed within each annotator (one person, one task, all
four fork positions mixed), so one annotator on a task is already a complete
measurement. Complete quartets -- one source's four forks judged by four
different people -- are a secondary check that accrues when several people work
the same task.

What is fixed is the RULE, and every draw is recorded: items are registered in
the DB the first time they are drawn (with their foil, which is deterministic
per item), and each batch stores its item list and seed.
"""
import hashlib
import os
import random
import secrets

from .dialogs import (HALLUCINATION_TYPES, SEGMENTS, TYPE_ABBREV, fork_path,
                      list_parent_ids, load_dialog, parent_path, resolve_fork_index,
                      system_turn_indices, was_injected)

MINI = 5        # one Latin-square row: 4 forks (each type once, each k once) + 1 conformant
BLOCK = 20      # four rows of one square: each (type x k) once + 4 conformant
FOIL_FRACTION = 0.20
FOIL_SEED = "forkbench-foil-v1"   # changing this changes every future foil


# --- identities -------------------------------------------------------------
#
# Every item has two names. `ref` is descriptive (task/type/k/dialog) and never
# leaves the server. `item_id` is an opaque random token and is what the browser
# sees: a descriptive id in the step-1 payload would hand the annotator the
# hallucination type and fork position before they judge.

def new_item_id():
    return secrets.token_hex(6)


def fork_id(task, htype, k, dialog_id):
    return "%s/%s/k%d/%s" % (task, TYPE_ABBREV[htype], k, dialog_id)


def conformant_id(task, dialog_id):
    return "%s/conformant/%s" % (task, dialog_id)


def has_prefix(item):
    """Whether step 3's "before the highlighted part" question applies: only a fork
    with at least one assistant turn before the injection. Conformant items have
    no fork, and every segment-1 fork is injected before assistant turn 1."""
    return item.get("kind") == "fork" and (item.get("fork_assistant_turn") or 0) > 1


def source_id(task, dialog_id):
    """The sibling unit: every version of one source dialogue shares its opening."""
    return "%s/%s" % (task, dialog_id)


def quartet_id(task, htype, dialog_id):
    return "%s/%s/%s" % (task, TYPE_ABBREV[htype], dialog_id)


# --- foils ------------------------------------------------------------------

def choose_foil(true_type, task, rng, pools, domain_tasks):
    """A variant from a DIFFERENT hallucination type than the true one.

    WIC foils draw a substitute from the same domain, mirroring how a real WIC
    fork is built, so the foil is plausible rather than absurd.
    """
    options = [h for h in HALLUCINATION_TYPES if h != true_type]
    htype = rng.choice(options)
    if htype == "workflow_identity_corruption":
        siblings = []
        for members in domain_tasks.values():
            if task in members:
                siblings = [m for m in members if m != task]
        if not siblings:
            htype = rng.choice([h for h in options if h != "workflow_identity_corruption"])
        else:
            return htype, rng.choice(siblings)
    variants = sorted(v for (h, v) in pools if h == htype)
    return htype, rng.choice(variants)


def _item_rng(item_id):
    h = hashlib.sha256(("%s|%s" % (FOIL_SEED, item_id)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# --- item construction --------------------------------------------------------

def make_fork_item(root, task, htype, k, dialog_id, pools, domain_tasks):
    """The item record, or None if the file is missing or was never injected
    (one of the 209 forks whose injection never fired)."""
    path = fork_path(root, task, htype, k, dialog_id)
    if not os.path.exists(path):
        return None
    d = load_dialog(path)
    fidx = resolve_fork_index(d)
    if fidx is None or not was_injected(d):
        return None
    ann = d["annotations"]
    ref = fork_id(task, htype, k, dialog_id)
    rng = _item_rng(ref)
    is_foil = rng.random() < FOIL_FRACTION
    if is_foil:
        cue_type, cue_variant = choose_foil(htype, task, rng, pools, domain_tasks)
    else:
        cue_type, cue_variant = htype, ann["variant_id"]
    return {
        "ref": ref, "kind": "fork", "task": task, "dialog_id": dialog_id,
        "source_id": source_id(task, dialog_id),
        "quartet_id": quartet_id(task, htype, dialog_id),
        "hallucination_type": htype, "variant_id": ann["variant_id"],
        "segment": k, "fork_assistant_turn": ann["fork_assistant_turn"], "fork_turn_index": fidx,
        "cue_type": cue_type, "cue_variant": cue_variant, "is_foil": is_foil,
        "path": os.path.relpath(path, root),
    }


def make_conformant_item(root, task, dialog_id, pools, domain_tasks):
    """Conformant originals are always foils: there is no added behaviour to name."""
    path = parent_path(root, task, dialog_id)
    if not os.path.exists(path):
        return None
    ref = conformant_id(task, dialog_id)
    cue_type, cue_variant = choose_foil(None, task, _item_rng(ref), pools, domain_tasks)
    return {
        "ref": ref, "kind": "conformant", "task": task, "dialog_id": dialog_id,
        "source_id": source_id(task, dialog_id), "quartet_id": None,
        "hallucination_type": None, "variant_id": None,
        "segment": None, "fork_assistant_turn": None, "fork_turn_index": None,
        "cue_type": cue_type, "cue_variant": cue_variant, "is_foil": True,
        "path": os.path.relpath(path, root),
    }


# --- drawing ----------------------------------------------------------------

class DrawError(Exception):
    pass


def batch_cells(minis, rng):
    """The cells of `minis` mini-blocks, one list per mini-block: four (htype, k)
    fork cells forming a Latin-square row, then None for the conformant cell.
    Each run of four mini-blocks is one whole square (type and k labels and row
    order shuffled), so every four of them cover each (type x k) exactly once."""
    types, ks = list(HALLUCINATION_TYPES), list(SEGMENTS)
    n = len(types)
    assert n == len(ks) == MINI - 1, "a mini-block is one fork per type and per k"
    rows = []
    while len(rows) < minis:
        rng.shuffle(types)
        rng.shuffle(ks)
        shifts = list(range(n))
        rng.shuffle(shifts)
        rows += [[(types[i], ks[(i + s) % n]) for i in range(n)] + [None] for s in shifts]
    return rows[:minis]


def draw_batch(root, task, minis, known, in_batches, rng, pools, domain_tasks,
               avoid_sources=(), complete_quartets=False):
    """Pick 5*minis items of `task`, one per source dialogue, listed mini-block by
    mini-block (see batch_cells).

    known        ref -> item for everything already registered (reused as is, so
                 an item's foil and id never change between batches)
    in_batches   item_id -> how many batches already contain it
    avoid_sources  sources the intended annotator already holds

    Preference, per cell:
      1. an item in no batch yet (new coverage before second judgements);
      2. complete_quartets=False (batch for anyone): a source NOBODY holds yet,
         so every annotator is eligible for the batch;
         complete_quartets=True (batch for a named person): a source whose
         quartet other people have already started, so quartets complete;
      3. random (seeded).

    Returns (items, new_items); new items have no item_id until registered.
    Raises DrawError when the task cannot supply it.
    """
    def batches_of(ref):
        it = known.get(ref)
        return in_batches.get(it["item_id"], 0) if it else 0

    avoid = set(avoid_sources)
    parents = [p for p in list_parent_ids(root, task)
               if source_id(task, p) not in avoid]
    need = MINI * minis
    if len(parents) < need:
        raise DrawError(
            "%s has %d source dialogue(s)%s; a batch of %d needs %d distinct ones"
            % (task, len(parents),
               " not already held by that person" if avoid else "", need, need))

    # how much of each source / quartet is already drawn anywhere
    src_drawn, q_drawn = {}, {}
    for it in known.values():
        if it["task"] != task or not in_batches.get(it["item_id"]):
            continue
        src_drawn[it["source_id"]] = src_drawn.get(it["source_id"], 0) + 1
        if it["quartet_id"]:
            q_drawn[it["quartet_id"]] = q_drawn.get(it["quartet_id"], 0) + 1

    rows = batch_cells(minis, rng)
    cells = [(r, c) for r, row in enumerate(rows) for c in row]
    rng.shuffle(cells)
    # conformant cells last: their only constraint is the source, so they
    # should not take a source a scarce fork cell needed
    cells.sort(key=lambda rc: rc[1] is None)
    used, picked, new = set(), [], []
    row_of = []
    tie = {p: rng.random() for p in parents}
    bad = set()

    for row, cell in cells:
        def key(p):
            if cell is None:
                iid = conformant_id(task, p)
                started = src_drawn.get(source_id(task, p), 0)
            else:
                iid = fork_id(task, cell[0], cell[1], p)
                started = q_drawn.get(quartet_id(task, cell[0], p), 0)
            fresh_src = src_drawn.get(source_id(task, p), 0)
            pref = -started if complete_quartets else fresh_src
            return (batches_of(iid), pref, tie[p])

        for p in sorted((p for p in parents if p not in used), key=key):
            iid = (conformant_id(task, p) if cell is None
                   else fork_id(task, cell[0], cell[1], p))
            if iid in bad:
                continue
            it = known.get(iid)
            if it is None:
                it = (make_conformant_item(root, task, p, pools, domain_tasks)
                      if cell is None else
                      make_fork_item(root, task, cell[0], cell[1], p, pools, domain_tasks))
                if it is None:
                    bad.add(iid)
                    continue
                new.append(it)
            used.add(p)
            picked.append(it)
            row_of.append(row)
            break
        else:
            raise DrawError(
                "%s ran out of usable source dialogues for %s; try a smaller batch"
                % (task, "a conformant slot" if cell is None
                   else "%s k=%d" % (TYPE_ABBREV[cell[0]], cell[1])))
    # back into mini-block order: the draw order above is only for scarcity
    picked = [it for _, it in sorted(zip(row_of, picked), key=lambda x: x[0])]
    return picked, new


def task_stats(root, task):
    """Number of source dialogues and mean System turns of the conformant originals."""
    ids = list_parent_ids(root, task)
    turns = []
    for p in ids:
        try:
            turns.append(len(system_turn_indices(load_dialog(parent_path(root, task, p)))))
        except (OSError, ValueError):
            pass
    return {"task": task, "sources": len(ids),
            "mean_system_turns": (sum(turns) / float(len(turns))) if turns else None}
