// SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
// SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
// SPDX-License-Identifier: MIT

/* Rendering shared by the annotator page and the admin batch viewer, so the
   admin sees a conversation exactly as an annotator does. Pure: HTML in, HTML
   out; nothing here knows or infers ground truth. */
(function (global) {
  "use strict";

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  /* Tag the words "assistant" and "user" in already-escaped HTML so that, in a
     question, they look like the turns they refer to. Tags are left alone. */
  function roles(html) {
    return String(html).replace(/(<[^>]*>)|\b(assistant|user)\b/gi, function (m, tag, w) {
      if (tag) { return tag; }
      return '<span class="role-' + (w.toLowerCase() === "assistant" ? "a" : "u") + '">' +
             w + "</span>";
    });
  }

  /* The workflow as steps (see flows.py): what the assistant does, with the
     branches under the step they come from. Branches are folded by default so
     the whole tree never lands on the reader at once; example sentences sit
     behind an "e.g." toggle -- one way to phrase a step, not required wording.
     opts.examples: open the examples; opts.stagger: numbered steps fade in */
  function flow(fl, opts) {
    opts = opts || {};
    function example(st) {
      if (!st.examples.length) { return ""; }
      return '<details class="eg"' + (opts.examples ? " open" : "") +
        '><summary title="an example of how it might say this">e.g.</summary>' +
        st.examples.map(function (e) {
          return '<span class="ex">\u201c' + esc(e) + "\u201d</span>";
        }).join("") + "</details>";
    }
    function count(o) {
      var n = o.steps.length;
      return n ? n + (n === 1 ? " step" : " steps") : "\u21a9 step " + o.goto_n;
    }
    function steps(list, numbered) {
      return list.map(function (st, i) {
        var outs = st.outcomes.map(function (o) {
          return '<details class="out"><summary><span class="if">' + esc(o.if) +
            '</span><span class="cnt">' + count(o) + "</span></summary>" +
            (o.steps.length ? '<ul class="sub">' + steps(o.steps, false) + "</ul>" : "") +
            (o.goto_n ? '<div class="goto">\u21a9 back to step ' + o.goto_n + "</div>" : "") +
            "</details>";
        }).join("");
        return '<li class="st' + (st.internal ? " sys" : "") + '"' +
          (numbered ? ' style="--i:' + i + '"' : "") +
          (numbered && st.n ? ' value="' + st.n + '"' : "") + ">" +
          '<span class="do">' + esc(st.do) + "</span>" +
          (st.internal ? '<span class="systag" title="happens behind the scenes, not ' +
                         'said to the user">system</span>' : "") +
          example(st) + (outs ? '<div class="outs">' + outs + "</div>" : "") + "</li>";
      }).join("");
    }
    var anyOut = fl.steps.some(function (st) { return st.outcomes.length; });
    return '<div class="flowbar"><span class="hint">What the ' + roles("assistant") +
      " should do. The exact words do not matter.</span><span class=tools>" +
      (anyOut ? '<button type="button" class="linkbtn outall">open all branches</button>' : "") +
      '<button type="button" class="linkbtn egall">' +
      (opts.examples ? "hide all examples" : "show all examples") + "</button></span></div>" +
      '<ol class="flow' + (opts.stagger ? " stagger" : "") + '">' + steps(fl.steps, true) +
      "</ol>" +
      (fl.optional.length
        ? '<div class="optional"><b>At any point it may also</b> &mdash; doing these is ' +
          '<b>not</b> a mistake:<ul class="sub">' + steps(fl.optional, false) + "</ul></div>"
        : "");
  }

  /* the "open all" switches inside a rendered flow */
  function wireFlow(host) {
    function toggle(btn, sel, on, off) {
      btn.addEventListener("click", function () {
        var box = btn.closest(".flowbar").parentNode;
        var all = box.querySelectorAll(sel);
        var open = !Array.prototype.every.call(all, function (d) { return d.open; });
        all.forEach(function (d) { d.open = open; });
        btn.textContent = open ? off : on;
      });
    }
    host.querySelectorAll(".egall").forEach(function (b) {
      toggle(b, "details.eg", "show all examples", "hide all examples");
    });
    host.querySelectorAll(".outall").forEach(function (b) {
      toggle(b, "details.out", "open all branches", "fold all branches");
    });
  }

  /* turns: [{speaker, sys_n, text}]
     opts.bandFrom  first System turn number to band (step 3 / admin only)
     opts.pick      make System turns clickable
     opts.chosen    System turn number to mark as chosen
     opts.marks     {sys_n: "label"} small markers (admin: annotators' onsets) */
  function transcript(turns, opts) {
    opts = opts || {};
    var band = opts.bandFrom || null, banded = false, html = [];
    turns.forEach(function (t) {
      var isSys = t.speaker === "System";
      if (band && isSys && t.sys_n >= band && !banded) {
        banded = true;
        html.push('<div class="band-note">' + roles("from here on, the assistant had " +
                  "the extra instruction below") + "</div>");
      }
      var cls = ["turn", isSys ? "system" : "user"];
      if (banded) { cls.push("banded"); }
      if (isSys && opts.pick) { cls.push("pick"); }
      if (isSys && opts.chosen === t.sys_n) { cls.push("chosen"); }
      var mark = isSys && opts.marks && opts.marks[t.sys_n]
        ? '<span class="mark">' + esc(opts.marks[t.sys_n]) + "</span>" : "";
      if (isSys && opts.pick && opts.chosen === t.sys_n) {
        mark += '<span class="startmark">\u25B6 starts here</span>';
      }
      html.push('<div class="' + cls.join(" ") + '"' +
                (isSys && opts.pick ? ' data-n="' + t.sys_n + '"' : "") + ">" +
                '<div class="who">' + (isSys
                  ? '<span class="role">Assistant</span><span class="n">' + t.sys_n + "</span>"
                  : '<span class="role">User</span>') + "</div>" +
                '<div class="text">' + esc(t.text) + mark + "</div></div>");
    });
    return html.join("");
  }

  global.Render = { esc: esc, roles: roles, flow: flow, wireFlow: wireFlow,
                    transcript: transcript };
})(window);
