# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Turn a STAR workflow (graph + replies) into the steps annotators read.

Annotators are not experts and must never be shown a DOT graph. The step labels
are written once per workflow in flows.py; validate_flow() checks each against
the dataset's graph, and the example sentences are always read from the data.

Standalone:  python3 -m annotool.workflow --all
"""
import json
import re

# --- 26 one-line task descriptions -------------------------------------------
# 24 DialFork tasks + the 2 WIC-only substitutes. STAR ships only a slug
# ("book_doctor_appointment"), so these are hand-written.
TASK_DESCRIPTIONS = {
    "apartment_schedule": "Book a viewing appointment for an apartment.",
    "apartment_search": "Find apartments that match what the user is looking for.",
    "bank_balance": "Look up the balance of the user's bank account.",
    "bank_fraud_report": "Take a report of fraud on the user's bank account.",
    "doctor_followup": "Give the user their doctor's follow-up instructions.",
    "doctor_schedule": "Book a doctor's appointment for the user.",
    "hotel_book": "Book a hotel room for the user.",
    "hotel_search": "Find hotels that match the user's criteria.",
    "hotel_service_request": "Take a service request from a hotel guest.",
    "meeting_schedule": "Schedule a meeting between the user and someone else.",
    "party_plan": "Organise and book a party for the user.",
    "party_rsvp": "Take the user's RSVP for a party.",
    "plane_book": "Book a flight for the user.",
    "plane_search": "Find flights that match the user's criteria.",
    "restaurant_book": "Book a table at a restaurant for the user.",
    "restaurant_search": "Find restaurants that match the user's criteria.",
    "ride_book": "Book a ride for the user.",
    "ride_change": "Change a ride the user has already booked.",
    "ride_status": "Tell the user how long until their booked ride arrives.",
    "spaceship_access_codes": "Verify a crew member's spaceship access code.",
    "spaceship_life_support": "Walk a crew member through a life-support lock procedure.",
    "trip_directions": "Give the user step-by-step directions for a trip.",
    "trivia": "Run a trivia quiz with the user.",
    "weather": "Tell the user the weather forecast.",
    # WIC-only substitutes: these never appear as DialFork tasks, only as the
    # task a workflow_identity_corruption fork is told to switch to.
    "trivia_score_reporting": "Review the user's previous quiz scores with them.",
    "weather_alerts": "Check weather alerts and offer to subscribe the user to them.",
}

# What each task is about, as a phrase that completes "It's all about ..." in the
# task intro, plus an emoji for the header and the intro. Friendly framing only.
TASK_TOPICS = {
    "apartment_schedule": ("booking a viewing of an apartment", "\U0001F3E0"),
    "apartment_search": ("finding an apartment to rent", "\U0001F3E1"),
    "bank_balance": ("checking a bank balance", "\U0001F3E6"),
    "bank_fraud_report": ("reporting fraud on a bank account", "\U0001F6A8"),
    "doctor_followup": ("passing on a doctor's instructions", "\U0001FA7A"),
    "doctor_schedule": ("booking a doctor's appointment", "\U0001F4C5"),
    "hotel_book": ("booking a hotel room", "\U0001F3E8"),
    "hotel_search": ("finding a hotel", "\U0001F50E"),
    "hotel_service_request": ("handling a hotel guest's service request", "\U0001F6CE\uFE0F"),
    "meeting_schedule": ("scheduling a meeting", "\U0001F4C6"),
    "party_plan": ("planning and booking a party", "\U0001F389"),
    "party_rsvp": ("RSVPing to a party", "\U0001F48C"),
    "plane_book": ("booking a flight", "\u2708\uFE0F"),
    "plane_search": ("finding a flight", "\U0001F6EB"),
    "restaurant_book": ("booking a table at a restaurant", "\U0001F37D\uFE0F"),
    "restaurant_search": ("finding a restaurant", "\U0001F35D"),
    "ride_book": ("booking a ride", "\U0001F695"),
    "ride_change": ("changing a booked ride", "\U0001F504"),
    "ride_status": ("checking when a ride will arrive", "\u23F1\uFE0F"),
    "spaceship_access_codes": ("checking a spaceship access code", "\U0001F680"),
    "spaceship_life_support": ("fixing a spaceship's life support", "\U0001F6F8"),
    "trip_directions": ("giving directions for a trip", "\U0001F5FA\uFE0F"),
    "trivia": ("playing a trivia quiz", "\U0001F9E0"),
    "weather": ("telling the weather forecast", "\u2600\uFE0F"),
    "trivia_score_reporting": ("reviewing past quiz scores", "\U0001F3C6"),
    "weather_alerts": ("checking weather alerts", "\u26C8\uFE0F"),
}


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?::[^}]*)?\}")
# STAR's other placeholder dialect: bare UPPER_SNAKE tokens (DOCTOR_NAME, TIME).
_SLOT = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*\b")
_NOT_SLOTS = {"ID", "PIN", "ASAP", "I", "OK", "A"}


def clean_reply(text):
    """An example sentence as annotators read it: placeholders become <slot name>."""
    if not isinstance(text, str):
        text = str(text)
    text = _PLACEHOLDER.sub(lambda m: "\u2039%s\u203a" % m.group(1).replace("_", " ").lower(),
                            text)
    text = _SLOT.sub(lambda m: m.group(0) if m.group(0) in _NOT_SLOTS or len(m.group(0)) < 3
                     else "\u2039%s\u203a" % m.group(0).replace("_", " ").lower(), text)
    return " ".join(text.split())


def build_flow(task, graph, replies):
    """The workflow as steps (flows.py), with example sentences from `replies`.

    -> {steps: [{n, do, internal, examples, outcomes: [{if, steps, goto_n}]}],
        optional: [{do, examples}]}
    Main steps are numbered; a goto names the step number it returns to.
    """
    from .flows import FLOWS
    replies = dict(replies or {})
    spec = FLOWS.get(task)
    if spec is None:     # a workflow nobody has described yet: examples only
        return {"steps": [], "optional": [{"do": "(no step description)", "internal": False,
                                           "examples": [clean_reply(v)], "outcomes": []}
                                          for v in replies.values()]}
    number = {}
    for i, st in enumerate(spec["steps"], 1):
        if st["id"]:
            number[st["id"]] = i

    def step(st, n=None):
        out = {"do": st["do"], "internal": st["internal"],
               "examples": [clean_reply(replies[x]) for x in st["nodes"] if x in replies],
               "outcomes": [{"if": o["if"], "steps": [step(x) for x in o["steps"]],
                             "goto_n": number.get(o["goto"])} for o in st["outcomes"]]}
        if n is not None:
            out["n"] = n
        return out

    return {"steps": [step(st, i) for i, st in enumerate(spec["steps"], 1)],
            "optional": [step(st) for st in spec["optional"]]}


def validate_flow(task, graph, replies):
    """Problems (list of str) between flows.py's description and the real graph.

    Checks that every reply is covered, every node named exists, every branch
    condition is attached where the graph sends it, and every edge out of a
    described step is represented, by the next step or by a goto.
    """
    from .flows import FLOWS
    spec = FLOWS.get(task)
    if spec is None:
        return ["no flow description for %s" % task]
    graph, replies = dict(graph or {}), dict(replies or {})
    targets = set(graph.values())
    known = set(graph) | targets | set(replies)
    conds = {k for k in graph if k not in targets and k != "hello" and k not in replies
             and not k.startswith("query")}
    ids = {st["id"]: st for st in spec["steps"] if st["id"]}
    problems, covered, used_conds = [], set(), set()

    def first(steps):
        return steps[0]["nodes"][0] if steps else None

    def check_list(steps, goto, where):
        for i, st in enumerate(steps):
            covered.update(st["nodes"])
            for x in st["nodes"]:
                if x not in known:
                    problems.append("%s: node %r is not in the workflow" % (where, x))
            a = st["nodes"][0]
            nxt = steps[i + 1]["nodes"][0] if i + 1 < len(steps) else None
            if a in graph and a not in conds:
                b = graph[a]
                back = ids[goto]["nodes"][0] if goto in ids else None
                if not (b == nxt or (nxt is None and b == back)):
                    problems.append("%s: %s -> %s is not what follows %r"
                                    % (where, a, b, st["do"]))
            elif nxt is not None:
                problems.append("%s: %r is followed by %r but the graph has no %s -> %s"
                                % (where, st["do"], steps[i + 1]["do"], a, nxt))
            for o in st["outcomes"]:
                if o["goto"] and o["goto"] not in ids:
                    problems.append("%s: unknown goto %r" % (where, o["goto"]))
                for w in o["when"]:
                    used_conds.add(w)
                    if w not in graph:
                        problems.append("%s: condition %r is not in the graph" % (where, w))
                        continue
                    lands = first(o["steps"]) or (ids[o["goto"]]["nodes"][0]
                                                  if o["goto"] in ids else None)
                    if graph[w] != lands:
                        problems.append("%s: %r leads to %s in the graph, not %s"
                                        % (where, w, graph[w], lands))
                check_list(o["steps"], o["goto"], "%s / %s" % (where, o["if"]))

    check_list(spec["steps"], None, task)
    for st in spec["optional"]:
        covered.update(st["nodes"])
    for k in sorted(set(replies) - covered):
        problems.append("%s: reply %r is not covered by any step" % (task, k))
    for c in sorted(conds - used_conds):
        problems.append("%s: condition %r is not attached to any step" % (task, c))
    return problems


def render_text(flow, title=None, description=None):
    """Plain-text rendering, for CLI inspection."""
    out = [x for x in (title, description) if x]
    out.append("")

    def emit(steps, depth):
        pad = "    " * depth
        for st in steps:
            head = ("%2d. " % st["n"]) if "n" in st else "  - "
            out.append(pad + head + st["do"] + ("  [system]" if st["internal"] else ""))
            for e in st["examples"]:
                out.append(pad + '        e.g. "%s"' % e)
            for o in st["outcomes"]:
                out.append(pad + "      " + o["if"] + ":")
                emit(o["steps"], depth + 2)
                if o["goto_n"]:
                    out.append(pad + "            -> back to step %d" % o["goto_n"])
    emit(flow["steps"], 0)
    if flow["optional"]:
        out.append("")
        out.append("  At any point it may also:")
        emit(flow["optional"], 1)
    return "\n".join(out)


# --- WIC: the substitute workflow lives in the injection event ---------------
# Verified over all 24 distinct variant_ids: every instruct-persist content
# carries the full DOT plus its response examples, including for the two
# substitutes that have no STAR directory. Do NOT read STAR for this.
_DOT_RE = re.compile(r"```dot\s*(.*?)```", re.S)
_JSON_RE = re.compile(r"```json\s*(.*?)```", re.S)
_EDGE_RE = re.compile(r"^\s*([A-Za-z_][\w]*)\s*->\s*([A-Za-z_][\w]*)\s*;?\s*$")


def parse_wic_injection(content):
    """Extract (task_name, graph, replies) of the substitute workflow."""
    name = None
    m = re.search(r"\*\*New task:\*\*\s*(\S+)", content or "")
    if m:
        name = m.group(1).strip()
    graph = {}
    m = _DOT_RE.search(content or "")
    if m:
        for line in m.group(1).splitlines():
            e = _EDGE_RE.match(line)
            if e:
                graph[e.group(1)] = e.group(2)
    replies = {}
    m = _JSON_RE.search(content or "")
    if m:
        try:
            replies = json.loads(m.group(1))
        except ValueError:
            replies = {}
    return name, graph, replies


if __name__ == "__main__":
    import argparse
    import glob
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="datasets/dialfork")
    ap.add_argument("--task", help="one task; default is all of them")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    tasks = [args.task] if args.task else sorted(os.listdir(args.root))
    for t in tasks:
        files = sorted(glob.glob(os.path.join(args.root, t, "originals", "*.json")))
        if not files:
            continue
        meta = json.load(open(files[0]))["annotations"]["task"]
        for problem in validate_flow(t, meta["graph"], meta["replies"]):
            print("!! " + problem)
        fl = build_flow(t, meta["graph"], meta["replies"])
        print("=" * 78)
        print(render_text(fl, title="TASK: %s" % t,
                          description=TASK_DESCRIPTIONS.get(t, "!! NO DESCRIPTION !!")))
        print()
