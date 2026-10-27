<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
SPDX-License-Identifier: MIT
-->

## What does this pull request do?

<!-- One or two sentences. For a leaderboard submission, fill in the section below; otherwise delete it. -->

## 🏆 Leaderboard submission

- **Method:** <!-- name, as it should appear in the table -->
- **Paper or code:** <!-- link -->
- **Task(s):** <!-- 1, 2 and/or 3 -->
- **Task 3 regime:** <!-- streaming / non-streaming / non-streaming + workflow (Task 3 only) -->
- **What it learned from or was calibrated on:** <!-- e.g. "the conformant dialogues of each task", or "nothing from DialFork" -->

**Scores**, as printed by `python evaluate.py --task N your_file.csv.gz`:

```
paste the output here
```

**Checklist**
- [ ] The prediction file is in `predictions/taskN/` and covers the full test set.
- [ ] A row is added to `leaderboard/v1.json` (`"source": "community"`, with a link), and `python scripts/make_leaderboard.py` was run.
- [ ] The method never trained on or tuned with the labels (fork position, fork turn, hallucination type).
- [ ] `pre-commit run --all-files` passes.
