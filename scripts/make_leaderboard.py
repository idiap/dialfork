# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Write the leaderboard tables of docs/leaderboard.md from leaderboard/<release>.json.

    python scripts/make_leaderboard.py            # rewrite the tables
    python scripts/make_leaderboard.py --check    # fail if docs/leaderboard.md is out of date
    python scripts/make_leaderboard.py --verify   # re-score every row's prediction file

    # add a submission: scores its prediction file, adds its row, rewrites the tables
    python scripts/make_leaderboard.py --add predictions/task3/my-method.csv.gz \
        --method "My Method" --link "Doe et al., 2027" https://... \
        --regime streaming --learns-from "train conformant"

leaderboard/<release>.json is the single source of the leaderboard: docs/leaderboard.md and
the Hugging Face Space (leaderboard/space/) both read it. Each row has its method, source
("paper", "community" or "control"), link, prediction file and scores, the latter exactly as
`evaluate.py --json` writes them. Each table in the docs sits between the markers
<!-- leaderboard:ID --> and <!-- /leaderboard:ID -->; the text around them is left as is.
Rows are sorted by their main score, best first, with the controls last.

--verify and --add need the data (python scripts/download_data.py --no-star). --verify checks
that every row with a `file` scores what the JSON says, at the precision the tables show.
--add takes the task from the file's folder (predictions/taskN/); Task 2 rows also need
--encoder, and Task 3 rows --regime and --learns-from.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "leaderboard.md")
TYPES = ("WIC", "SSPC", "GR", "CSC")
SOURCE = {"paper": "📄", "community": "🌍", "control": ""}
REGIMES = ("streaming", "non-streaming", "non-streaming+workflow")


def tau(v):
    return ("%.3f" % v).replace("-", "−")


# per task: extra text columns, then the score columns as (header, getter, formatter); the last is the main score
COLUMNS = {
    1: ([], [(t, lambda s, t=t: s["by_type"][t], tau) for t in TYPES]
        + [("**Avg.**", lambda s: s["avg"], tau)]),
    2: ([("encoder", "encoder")], [(t, lambda s, t=t: s["by_type"][t], tau) for t in TYPES]
        + [("**Avg.**", lambda s: s["avg"], tau)]),
    3: ([("learns from", "learns_from")],
        [("MAE", lambda s: s["hall_mae"], "{:.2f}".format),
         ("Cov.", lambda s: s["hall_cov"], "{:.2f}".format),
         ("False alarms", lambda s: s["conf_cov"], "{:.2f}%".format),
         ("**Score**", lambda s: s["score"], "{:.2f}".format)]),
}


def boards(lb):
    """(marker id, task, rows) for every table."""
    for task in ("task1", "task2", "task3"):
        for r in lb[task]:
            if r["source"] not in SOURCE:
                sys.exit("%s, %s: unknown source %r" % (task, r["method"], r["source"]))
            if task == "task3" and r.get("regime") not in REGIMES:
                sys.exit("task3, %s: regime must be one of %s" % (r["method"], ", ".join(REGIMES)))
    yield "task1", 1, lb["task1"]
    yield "task2", 2, lb["task2"]
    for regime in REGIMES:
        yield "task3-" + regime, 3, [r for r in lb["task3"] if r["regime"] == regime]


def table(task, rows):
    texts, scores = COLUMNS[task]
    main = scores[-1][1]
    methods = sorted((r for r in rows if r["source"] != "control"), key=lambda r: -main(r["scores"]))
    controls = sorted((r for r in rows if r["source"] == "control"), key=lambda r: -main(r["scores"]))
    out = ["| | method | " + " | ".join([h for h, _ in texts] + [h for h, _, _ in scores]) + " | link |",
           "|---|---|" + "---|" * len(texts) + "---:|" * len(scores) + "---|"]
    for i, r in enumerate(methods + controls):
        control = r["source"] == "control"
        cells = [r["method"] + (" (control)" if control else "")]
        cells += [r.get(key) or "–" for _, key in texts]
        cells += [fmt(get(r["scores"])) for _, get, fmt in scores]
        if control:  # italic, except the placeholder dashes
            cells = [c if c == "–" else "*%s*" % c for c in cells]
        elif i == 0:
            cells[-1] = "**%s**" % cells[-1]
        link = "[%s](%s)" % (r["link"]["text"], r["link"]["url"]) if r.get("link") else ""
        out.append("|" + "|".join(" %s " % c if c else " " for c in [SOURCE[r["source"]]] + cells + [link]) + "|")
    return "\n".join(out)


def render(doc, lb):
    for bid, task, rows in boards(lb):
        pattern = re.compile(r"(<!-- leaderboard:%s -->\n\n).*?(\n\n<!-- /leaderboard:%s -->)"
                             % (re.escape(bid), re.escape(bid)), re.S)
        if not pattern.search(doc):
            sys.exit("docs/leaderboard.md has no <!-- leaderboard:%s --> markers" % bid)
        doc = pattern.sub(lambda m: m.group(1) + table(task, rows) + m.group(2), doc)
    return doc


def score(task, release, path):
    """The scores of a prediction file, as `evaluate.py --json` writes them."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "scores.json")
        subprocess.run([sys.executable, os.path.join(ROOT, "evaluate.py"), "--task", str(task),
                        "--release", release, path, "--json", out],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        with open(out) as fh:
            return json.load(fh)


def verify(lb):
    bad = 0
    for bid, task, rows in boards(lb):
        for r in rows:
            if "file" not in r:
                continue
            got = score(task, lb["release"], r["file"])
            ok = all(fmt(get(got)) == fmt(get(r["scores"])) for _, get, fmt in COLUMNS[task][1])
            print(("  ok   " if ok else "  FAIL ") + "%s: %s (%s)" % (bid, r["method"], r["file"]))
            bad += not ok
    return bad


def add(ap, args, lb):
    m = re.match(r"predictions/task([123])/[^/]+$", os.path.relpath(os.path.abspath(args.add), ROOT).replace(os.sep, "/"))
    if not m:
        ap.error("--add: the file must be in predictions/task1/, task2/ or task3/")
    task = int(m.group(1))
    path = m.group(0)
    need = ["method", "link"] + {2: ["encoder"], 3: ["regime", "learns_from"]}.get(task, [])
    missing = ["--" + k.replace("_", "-") for k in need if not getattr(args, k)]
    if missing:
        ap.error("--add (Task %d) also needs %s" % (task, ", ".join(missing)))
    rows = lb["task%d" % task]
    if any(r.get("file") == path for r in rows):
        sys.exit("%s is already on the leaderboard" % path)
    r = {"method": args.method, "source": args.source}
    if task == 2:
        r["encoder"] = args.encoder
    if task == 3:
        r.update(regime=args.regime, learns_from=args.learns_from)
    r.update(link={"text": args.link[0], "url": args.link[1]}, file=path, scores=score(task, lb["release"], path))
    rows.append(r)
    with open(os.path.join(ROOT, "leaderboard", args.release + ".json"), "w", encoding="utf-8") as fh:
        json.dump(lb, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print("added %s to leaderboard/%s.json (task %d)" % (args.method, args.release, task))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default="v1")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if docs/leaderboard.md is out of date")
    mode.add_argument("--verify", action="store_true", help="re-score every row's prediction file")
    mode.add_argument("--add", metavar="FILE", help="add a row for this prediction file (predictions/taskN/...)")
    row = ap.add_argument_group("the row --add writes")
    row.add_argument("--method", help="the method's name, as the tables show it")
    row.add_argument("--link", nargs=2, metavar=("TEXT", "URL"), help="its paper or code")
    row.add_argument("--source", choices=sorted(SOURCE), default="community")
    row.add_argument("--encoder", help="Task 2: the encoder")
    row.add_argument("--regime", choices=REGIMES, help="Task 3: the regime")
    row.add_argument("--learns-from", help="Task 3: what it learned from, e.g. \"none (zero-shot)\"")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "leaderboard", args.release + ".json"), encoding="utf-8") as fh:
        lb = json.load(fh)
    if args.verify:
        bad = verify(lb)
        sys.exit("%d row(s) do not match their prediction file" % bad if bad else 0)
    if args.add:
        add(ap, args, lb)

    with open(DOC, encoding="utf-8") as fh:
        doc = fh.read()
    new = render(doc, lb)
    if args.check:
        if new != doc:
            sys.exit("docs/leaderboard.md is out of date: run python scripts/make_leaderboard.py")
        print("docs/leaderboard.md is up to date")
    elif new != doc:
        with open(DOC, "w", encoding="utf-8") as fh:
            fh.write(new)
        print("updated docs/leaderboard.md")


if __name__ == "__main__":
    main()
