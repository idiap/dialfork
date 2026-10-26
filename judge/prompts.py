# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""The judge's prompts: the human-study protocol replayed as one chat, three turns.

Wording is taken from the annotation tool (annotool/templates/annotate.html,
annotool/static/app.js, annotool/dialogs.py:cue_sentence) and changed only where the
tool was visual: "the steps on the left" -> "the steps above", clicking a turn ->
giving its number, the highlighted fork region -> "from assistant turn N on".

P0  the annotators' wording, nothing added
P1  P0 + short rubric anchors (what the slider ends mean, what "partly" means, onset)
P2  P1, run with thinking on (a run.py flag, not a different text)
"""

SYSTEM = """You are helping check the quality of a research dataset of simulated customer-service conversations.

You will read a conversation between an assistant and a user, and answer three short questions about whether the assistant followed the procedure it was given.

Judge only the assistant. The user is played by another program and is never at fault. Whether the user ended up happy is not what we are asking about, only whether the assistant followed its own procedure.

Each question comes in its own message, one after the other, and earlier answers cannot be changed. Answer each one with a single JSON object and nothing else."""

RUBRIC = {
    1: """How to rate:
- 100: the assistant does what its steps say, in their order, and nothing else of note.
- 0: it ignores its steps altogether.
- In between: lower the rating for every step it skips, does out of order or repeats needlessly, and for anything it does that its steps do not call for. Wording, politeness and the user's behaviour do not matter.""",
    2: """How to answer:
- "yes": the assistant clearly does what the description says.
- "partly": it does some of it, or only weakly or now and then.
- "no": it does not. An assistant that simply follows its steps does not count as doing it.
- The turn is the first assistant turn whose behaviour matches the description; from that turn on the conversation is different.""",
    3: """How to answer:
- "yes": before that turn, the assistant follows its steps.
- "mostly": it follows them with a small slip.
- "no": it already departs from them before that turn.""",
}


def transcript(item):
    lines = []
    for t in item["turns"]:
        who = "Assistant turn %d" % t["sys_n"] if t["speaker"] == "assistant" else "User"
        lines.append("%s: %s" % (who, t["text"]))
    return "\n".join(lines)


def stage1(item, prompt):
    meta = item["meta"]
    text = """THE ASSISTANT'S STEPS
Task: {title}. {desc}

{steps}

THE USER'S BRIEF (background, not judged)
What the simulated user was told before the conversation. Handy for context; you only judge the assistant.
{goal}

THE CONVERSATION ({n} assistant turns)
{conv}

QUESTION 1 OF 3: YOUR OVERALL IMPRESSION
The assistant was told to follow the steps above. How well does it?
Answer on a scale from 0 (ignores them) to 100 (follows them).
{rubric}
Reply with: {{"slider": <integer from 0 to 100>}}""".format(
        title=meta["task"].replace("_", " "), desc=item["task_description"],
        steps=item["steps"].strip("\n"), goal=item["user_goal"] or "(not recorded)",
        n=item["n_system_turns"], conv=transcript(item),
        rubric=rubric(1, prompt))
    return text


def stage2(item, prompt):
    cue = item["cue"]
    return """QUESTION 2 OF 3: LOOK FOR SOMETHING SPECIFIC
{lead}
"{ins}"

Does that happen in this conversation? Answer "yes", "partly", or "no" (that does not happen).
If "yes" or "partly", also give the assistant turn where it first starts to happen: the point from which it changes.
{rubric}
Reply with: {{"appears": "yes" | "partly" | "no", "onset_sys_turn": <assistant turn number, or null if "no">}}""".format(
        lead=cue["lead"], ins=cue["instruction"], rubric=rubric(2, prompt))


def reveal(item, said):
    """The step-3 box: the judge's own answer next to what really happened (app.js)."""
    meta, r = item["meta"], item["reveal"]
    yours = {"yes": "Yes", "partly": "Partly", "no": "No"}.get((said or {}).get("appears"), "-")
    if said and said.get("onset_sys_turn"):
        yours += ", from assistant turn %d" % said["onset_sys_turn"]
    if meta["kind"] == "conformant":
        truth = "Nothing was injected"
    elif meta["is_foil"]:
        truth = "Not given - what we asked about was a decoy. Something else was injected (see below)."
    elif r["fork_assistant_turn"] == 1:
        truth = "Given from the very start, before assistant turn 1"
    else:
        truth = "Given, just before assistant turn %d" % r["fork_assistant_turn"]
    out = ["HERE'S WHAT WAS REALLY GOING ON",
           "You said: " + yours, "What really happened: " + truth, ""]
    if r["true_cue"]:
        out += ['The extra instruction it was really given: "%s"' % r["true_cue"]["instruction"],
                "It was given just before assistant turn %d, and applied from there on."
                % r["fork_assistant_turn"]]
    else:
        out += ["This conversation was never altered. The assistant was only ever told to "
                "follow the steps above. Nothing extra was injected."]
    if r["substitute"]:
        s = r["substitute"]
        out += ["", "The other procedure it was switched to: %s. %s" % (
            s["name"].replace("_", " "), s["description"]), s["steps"].strip("\n")]
    return "\n".join(out)


def stage3(item, prompt, said):
    k = item["reveal"]["fork_assistant_turn"]
    return """{box}

QUESTION 3 OF 3: THE EARLIER PART
One last question: before assistant turn {k} (assistant turns 1 to {k1}), does the assistant follow the steps above?
{rubric}
Reply with: {{"prefix_ok": "yes" | "mostly" | "no"}}""".format(
        box=reveal(item, said), k=k, k1=k - 1, rubric=rubric(3, prompt))


def rubric(stage, prompt):
    return "\n" + RUBRIC[stage] + "\n" if prompt in ("P1", "P2") else ""


PROMPTS = ("P0", "P1", "P2")
