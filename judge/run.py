# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Run one judge (model x prompt x seed) over the human-study items, stage by stage.

The three questions are asked in one chat, in order, each after the previous answer,
so the cue is withheld until the rating is given (as the server enforced for people).
Stage 3 is only asked when the conversation has a part before the fork.

Calls Ollama's /api/chat directly (OLLAMA_HOST): plain-text prompts, no constrained
decoding, the JSON is parsed out of the reply, as in methods/task3/llm.py.

Output: results/judge/outputs/<model>/<prompt>/seed<S>/judgments.jsonl (<prompt>_t<T> for a sampled
run; judgments_shard<NN>.jsonl when split with --nshards), one line per (item_id, stage),
appended as it goes. A rerun skips the stages already there, and rebuilds the chat
history from them, so each item resumes with its own answers.

    python -m judge.items                  # once: the inputs
    python -m judge.run -m qwen3.6:35b -p P0 [--limit 10] [--ids a,b] [--seed 0]
"""
import argparse
import glob
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from judge import common
from judge import prompts

APPEARS = ("yes", "partly", "no")
PREFIX = ("yes", "mostly", "no")


def chat(host, model, messages, seed, think, num_ctx, temperature=0.0, timeout=1800):
    body = {"model": model, "messages": messages, "stream": False, "think": think,
            "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}}
    req = urllib.request.Request("http://%s/api/chat" % host, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    msg = out.get("message") or {}
    return msg.get("content") or "", msg.get("thinking") or "", out


def extract_json(text):
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.I)
    if not m:
        m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(1))


def as_int(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return int(round(v))
    m = re.search(r"-?\d+", str(v))
    return int(m.group()) if m else None


def parse(stage, text, item):
    """The answer in the human CSV's terms, or ValueError (then the call is retried)."""
    obj = extract_json(text)
    if stage == 1:
        v = as_int(obj.get("slider"))
        if v is None or not 0 <= v <= 100:
            raise ValueError("slider %r" % obj.get("slider"))
        return {"slider": v}
    if stage == 2:
        a = str(obj.get("appears", "")).strip().lower()
        if a not in APPEARS:
            raise ValueError("appears %r" % obj.get("appears"))
        if a == "no":
            return {"appears": a, "onset_sys_turn": None}
        t = as_int(obj.get("onset_sys_turn"))
        if t is None or not 1 <= t <= item["n_system_turns"]:
            raise ValueError("onset %r of %d" % (obj.get("onset_sys_turn"), item["n_system_turns"]))
        return {"appears": a, "onset_sys_turn": t}
    p = str(obj.get("prefix_ok", "")).strip().lower()
    if p not in PREFIX:
        raise ValueError("prefix_ok %r" % obj.get("prefix_ok"))
    return {"prefix_ok": p}


def out_dir(model, prompt, seed, temperature=0.0):
    """Greedy runs (the default) under <prompt>/, sampled ones under <prompt>_t<T>/: at
    temperature 0 the seed changes nothing, so seeds only vary a sampled run."""
    name = common.model_dir(model)
    if temperature:
        prompt = "%s_t%g" % (prompt, temperature)
    return os.path.join(common.OUTPUTS, name, prompt, "seed%d" % seed)


def load_done(path):
    """(item_id, stage) -> record; the last line wins if a stage appears twice."""
    done = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r["raw"] or r["answer"] is not None:   # no reply = server failure, redo
                        done[(r["item_id"], r["stage"])] = r
    return done


class Judge:
    def __init__(self, args, path):
        self.args, self.path = args, path
        self.think = args.prompt == "P2"
        self.lock = threading.Lock()
        # what any shard of this run already answered counts as done (so a run can be
        # split again without redoing anything); this process writes only its own file
        self.done = {}
        for other in sorted(glob.glob(os.path.join(os.path.dirname(path), "judgments*.jsonl"))):
            if other != path:
                self.done.update(load_done(other))
        self.done.update(load_done(path))

    def write(self, rec):
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.done[(rec["item_id"], rec["stage"])] = rec

    def ask(self, item, stage, messages):
        """(answer, raw, thinking, attempts, error, seconds, meta). A reply that does not
        parse is asked again, up to --max-attempts; a server failure (no reply) is retried
        after a pause and, if it persists, raised: it is never cached as an answer."""
        a = self.args
        err, text, thinking, meta, server_fails = None, "", "", {}, 0
        attempt = 0
        while attempt < a.max_attempts:
            t0 = time.time()
            try:
                # the seed only matters when sampling (--temperature > 0): at temperature 0
                # decoding is greedy, so a retry usually repeats the same reply
                text, thinking, meta = chat(a.host, a.model, messages, a.seed + attempt,
                                            self.think, a.num_ctx, a.temperature)
            except Exception as e:      # noqa: BLE001 -- connection, HTTP 500, timeout
                server_fails += 1
                if server_fails >= a.max_server_fails:
                    raise RuntimeError("server keeps failing: %s: %s" % (type(e).__name__, e))
                time.sleep(min(60, 5 * server_fails))
                continue
            attempt += 1
            try:
                return parse(stage, text, item), text, thinking, attempt, None, time.time() - t0, meta
            except Exception as e:      # noqa: BLE001 -- any parse failure is a failed attempt
                err = "%s: %s" % (type(e).__name__, e)
        return None, text, thinking, attempt, err, time.time() - t0, meta

    def item(self, item):
        p = self.args.prompt
        messages = [{"role": "system", "content": prompts.SYSTEM}]
        said = None
        stages = (1, 2, 3) if item["has_prefix"] else (1, 2)
        for stage in stages:
            if stage == 1:
                q = prompts.stage1(item, p)
            elif stage == 2:
                q = prompts.stage2(item, p)
            else:
                q = prompts.stage3(item, p, said)
            messages.append({"role": "user", "content": q})
            rec = self.done.get((item["item_id"], stage))
            if rec is None:
                ans, text, thinking, n, err, secs, meta = self.ask(item, stage, messages)
                rec = {"item_id": item["item_id"], "stage": stage, "model": self.args.model,
                       "prompt": p, "seed": self.args.seed,
                       "temperature": self.args.temperature, "answer": ans, "raw": text,
                       "thinking": thinking, "attempts": n, "error": err,
                       "seconds": round(secs, 2),
                       "prompt_tokens": meta.get("prompt_eval_count"),
                       "output_tokens": meta.get("eval_count")}
                self.write(rec)
                if err:
                    print("[warn] %s stage %d: %s" % (item["item_id"], stage, err), flush=True)
            # the history holds the reply as given, parsed or not
            messages.append({"role": "assistant", "content": rec["raw"]})
            if stage == 2:
                said = rec["answer"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-p", "--prompt", default="P0", choices=prompts.PROMPTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--items", default=common.ITEMS)
    ap.add_argument("--ids", default="", help="comma-separated item ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="", help="instead of results/judge/outputs/<model>/<prompt>/seed<S>")
    ap.add_argument("--nshards", type=int, default=1,
                    help="split the items; this run takes every nshards-th one")
    ap.add_argument("--shard", type=int, default=None,
                    help="which one (default: $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--workers", type=int, default=1, help="parallel items (set OLLAMA_NUM_PARALLEL)")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--max-attempts", type=int, default=3, help="unparsable replies")
    ap.add_argument("--max-server-fails", type=int, default=6)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"))
    args = ap.parse_args(argv)

    with open(args.items, encoding="utf-8") as fh:
        items = [json.loads(l) for l in fh]
    if args.ids:
        keep = set(args.ids.split(","))
        items = [i for i in items if i["item_id"] in keep]
    name = "judgments.jsonl"
    if args.nshards > 1:
        shard = args.shard if args.shard is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        items = items[shard::args.nshards]
        name = "judgments_shard%02d.jsonl" % shard     # one file per process
    if args.limit:
        items = items[:args.limit]

    d = args.out_dir or out_dir(args.model, args.prompt, args.seed, args.temperature)
    os.makedirs(d, exist_ok=True)
    judge = Judge(args, os.path.join(d, name))
    todo = [i for i in items if any((i["item_id"], s) not in judge.done
                                    for s in ((1, 2, 3) if i["has_prefix"] else (1, 2)))]
    print("%s %s seed%d: %d items, %d to do -> %s" % (args.model, args.prompt, args.seed,
          len(items), len(todo), judge.path), flush=True)
    t0 = time.time()
    def one(item):
        try:
            judge.item(item)
        except RuntimeError as e:
            print("[error] %s: %s (left for a rerun)" % (item["item_id"], e), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, _ in enumerate(pool.map(one, todo), 1):
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                print("  %d/%d items, %.0fs, ~%.0fs left" % (n, len(todo), el,
                      el / n * (len(todo) - n)), flush=True)
    fails = sum(1 for r in judge.done.values() if r["answer"] is None)
    left = sum(1 for i in items for s in ((1, 2, 3) if i["has_prefix"] else (1, 2))
               if (i["item_id"], s) not in judge.done)
    print("done; %d stage answers, %d unparsed, %d stages left for a rerun"
          % (len(judge.done), fails, left))
    return 0


if __name__ == "__main__":
    sys.exit(main())
