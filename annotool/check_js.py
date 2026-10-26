# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT

"""Crude JS sanity check -- no JS engine is available on this host.

Balances {} () [] while skipping strings, template literals, comments and regex
literals, and checks that every element id the client looks up exists either in
its HTML template or in markup the client itself generates. It is not a parser;
it catches the common typo classes only. A browser check is still required.

  python3 -m annotool.check_js
"""
import re
import sys

PAIRS = {"}": "{", ")": "(", "]": "["}
# a '/' after one of these starts a regex literal, not a division
REGEX_OK = set("(,=:[!&|?{};+-*%~^<>") | {"\n"}


def balance(src):
    i, n, line = 0, len(src), 1
    stack = []
    prev = "\n"
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            prev = "\n"
            continue
        if c.isspace():
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                line += src[i] == "\n"
                i += 1
            i += 2
            continue
        if c == "/" and prev in REGEX_OK:
            i += 1
            in_class = False
            while i < n and (in_class or src[i] != "/"):
                if src[i] == "\\":
                    i += 1
                elif src[i] == "[":
                    in_class = True
                elif src[i] == "]":
                    in_class = False
                elif src[i] == "\n":
                    return "line %d: newline inside regex literal" % line
                i += 1
            i += 1
            prev = "/"
            continue
        if c in "\"'`":
            q, i = c, i + 1
            while i < n and src[i] != q:
                if src[i] == "\\":
                    i += 1
                elif src[i] == "\n":
                    line += 1
                i += 1
            i += 1
            prev = q
            continue
        if c in "{([":
            stack.append((c, line))
        elif c in "})]":
            if not stack:
                return "line %d: stray %r" % (line, c)
            op, ol = stack.pop()
            if op != PAIRS[c]:
                return "line %d: %r closes %r from line %d" % (line, c, op, ol)
        prev = c
        i += 1
    if stack:
        return "unclosed %r from line %d" % (stack[-1][0], stack[-1][1])
    return None


def ids_in(text):
    return set(re.findall(r'id=["\']?([A-Za-z0-9_]+)', text))


def main(argv=None):
    fails = 0
    for js, html in [("annotool/static/app.js", "annotool/templates/annotate.html"),
                     ("annotool/static/admin.js", "annotool/templates/admin.html"),
                     ("annotool/static/fun.js", "annotool/templates/annotate.html"),
                     ("annotool/static/login.js", "annotool/templates/login.html"),
                     ("annotool/static/render.js", None),
                     ("annotool/static/charts.js", None)]:
        src = open(js, encoding="utf-8").read()
        r = balance(src)
        if r:
            print("  FAIL %s: %s" % (js, r))
            fails += 1
        else:
            print("  ok   %s: delimiters balanced" % js)
        if html is None:
            continue
        want = set(re.findall(r'\$\("([A-Za-z0-9_]+)"\)', src))
        want |= set(re.findall(r'getElementById\("([A-Za-z0-9_]+)"\)', src))
        # available = ids in the template plus ids the client injects itself
        have = ids_in(open(html, encoding="utf-8").read()) | ids_in(src)
        miss = sorted(want - have)
        if miss:
            print("  FAIL %s: looks up ids that nothing defines: %s" % (js, miss))
            fails += 1
        else:
            print("  ok   %s: all %d looked-up ids are defined" % (js, len(want)))
    print("\n%s (%d failures)  -- still needs a real browser check"
          % ("PASSED" if not fails else "FAILURES", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
