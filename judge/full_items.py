# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Judge inputs for the whole of DialFork, cued as in the human study.

Every conversation gets the cue the study's sampler gives it (annotool/sampling.py:
make_fork_item / make_conformant_item, deterministic per conversation): the true
instruction for ~80% of forks, a foil for the other ~20% and for every conformant
dialogue. The 428 study conversations get exactly the cue the annotators saw. Forks
whose injection never fired (209) are left out, as in the study.

Any dataset in DialFork's folder layout works, e.g. one regenerated with another LLM
(docs/dataset.md#regenerating-dialfork); its study conversations are not marked.

    python -m judge.full_items        # results/judge/full_items.jsonl
    python -m judge.full_items -d datasets/my_dialfork -o results/judge/my_items.jsonl
"""
import argparse
import json
import os
from collections import Counter

from judge import common
from judge import items as study
from annotool.dialogs import (HALLUCINATION_TYPES, SEGMENTS, list_parent_ids, list_tasks,
                              load_domain_tasks, load_variant_pools)
from annotool.sampling import make_conformant_item, make_fork_item


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-d", "--dataset", default=common.DATASET)
    ap.add_argument("-o", "--out", default=common.FULL_ITEMS)
    args = ap.parse_args(argv)
    pools = load_variant_pools(common.GENERATE_PY)
    domains = load_domain_tasks(common.GENERATE_PY)
    judged = study.load_human_answers()     # registered but never judged do not count
    in_study = {m["ref"] for i, m in study.load_study_items().items() if i in judged}
    root = args.dataset
    if os.path.realpath(root) != os.path.realpath(common.DATASET):
        in_study = set()
    n, kinds = 0, Counter()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for task in list_tasks(root):
            for did in list_parent_ids(root, task):
                metas = [make_conformant_item(root, task, did, pools, domains)]
                metas += [make_fork_item(root, task, h, k, did, pools, domains)
                          for h in HALLUCINATION_TYPES for k in SEGMENTS]
                for m in metas:
                    if m is None:
                        kinds["skipped (missing or never injected)"] += 1
                        continue
                    m["item_id"] = m["ref"]
                    rec = study.build_record(m, [], pools, root)
                    rec["split"] = common.split_of(task)
                    rec["in_study"] = m["ref"] in in_study
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    kinds[(m["kind"], "foil" if m["is_foil"] else "true cue")] += 1
    for k, v in sorted(kinds.items(), key=str):
        print("  %-40s %d" % (k, v))
    print("wrote %d conversations to %s" % (n, args.out))


if __name__ == "__main__":
    main()
