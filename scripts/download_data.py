# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Download DialFork (and STAR) into the folder layout the experiment scripts expect.

    python scripts/download_data.py            # both, into datasets/
    python scripts/download_data.py --no-star  # DialFork only

--release picks the DialFork release (default: v1, the only one so far).

DialFork comes from the Hugging Face Hub (https://huggingface.co/datasets/DialFork/dialfork) and is written to
datasets/dialfork/<task>/originals/dialog_<id>.json and
datasets/dialfork/<task>/<hallucination_type>/n_hallucinations_4/fork_at_segment_<k>/dialog_<id>.json,
exactly the files the dataset was built as, plus datasets/dialfork/task3_splits.csv (the
official Task 3 train/test split; see scripts/make_splits.py). STAR (whose task workflows the code also
reads) comes from its original GitHub repository, at a pinned commit.

Requires: pip install huggingface_hub pyarrow
"""
import argparse
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile

REPO_ID = "DialFork/dialfork"
# release -> (Hub revision, folder under --output); evaluate.py has the same folders
RELEASES = {"v1": ("v1", "dialfork")}
STAR_COMMIT = "3058975092cdd09798e4a3a9e4ae135755824545"
STAR_URL = "https://github.com/RasaHQ/STAR/archive/%s.zip" % STAR_COMMIT
N_FORKS = 4


def dialog_path(root, r):
    name = "dialog_%d.json" % r["dialog_id"]
    if r["kind"] == "conformant":
        return os.path.join(root, r["task"], "originals", name)
    return os.path.join(root, r["task"], r["hallucination_type"], "n_hallucinations_%d" % N_FORKS,
                        "fork_at_segment_%d" % r["fork_position"], name)


def rebuild(src, root):
    """src: a local folder with data/*.parquet and extra/done_markers.json."""
    import pyarrow.parquet as pq
    shards = sorted(f for f in os.listdir(os.path.join(src, "data")) if f.endswith(".parquet"))
    n = 0
    for f in shards:
        cols = ["task", "dialog_id", "kind", "hallucination_type", "fork_position",
                "dialog_json"]
        for r in pq.read_table(os.path.join(src, "data", f), columns=cols).to_pylist():
            p = dialog_path(root, r)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="") as fh:
                fh.write(r["dialog_json"])
            n += 1
    with open(os.path.join(src, "extra", "done_markers.json"), encoding="utf-8") as fh:
        done = json.load(fh)
    for task, text in done.items():
        with open(os.path.join(root, task, "DONE.txt"), "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    shutil.copyfile(os.path.join(src, "extra", "task3_splits.csv"), os.path.join(root, "task3_splits.csv"))
    return n, len(done)


def get_dialfork(out, repo_id, revision, local=None):
    if local is None:
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id, repo_type="dataset", revision=revision,
                                  allow_patterns=["data/*", "extra/*"])
    n, tasks = rebuild(local, out)
    print("DialFork: %d dialogues over %d tasks -> %s" % (n, tasks, out))


def get_star(out):
    print("STAR: downloading %s ..." % STAR_URL)
    with urllib.request.urlopen(STAR_URL) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
    top = z.namelist()[0].split("/")[0]
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    z.extractall(tmp)
    shutil.rmtree(out, ignore_errors=True)
    shutil.move(os.path.join(tmp, top), out)
    shutil.rmtree(tmp, ignore_errors=True)
    print("STAR: commit %s -> %s" % (STAR_COMMIT[:7], out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", default="datasets",
                    help="parent folder (default: datasets/)")
    ap.add_argument("--release", choices=sorted(RELEASES), default="v1",
                    help="DialFork release (default: v1)")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--from-local", default=None,
                    help="rebuild from an already downloaded/exported copy instead of the Hub")
    ap.add_argument("--no-star", action="store_true", help="skip STAR")
    ap.add_argument("--no-dialfork", action="store_true", help="skip DialFork")
    args = ap.parse_args()
    if not args.no_dialfork:
        revision, folder = RELEASES[args.release]
        target = os.path.join(args.output, folder)
        if os.path.isdir(target) and os.listdir(target):
            sys.exit("%s already exists and is not empty; remove it first" % target)
        get_dialfork(target, args.repo_id, revision, args.from_local)
    if not args.no_star:
        get_star(os.path.join(args.output, "STAR"))


if __name__ == "__main__":
    main()
