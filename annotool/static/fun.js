// SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
// SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
// SPDX-License-Identifier: MIT

/* Encouragement for annotators: task stages, milestone toasts, a celebration
   when a task is done. Presentation only.

   The rule every addition here must follow: reward EFFORT and PROGRESS, never
   answers or speed. Nothing here may know whether an answer matched the ground
   truth (that would teach annotators what to answer), and nothing may reward
   going faster (that buys rushed judgements). */
(function (global) {
  "use strict";
  var esc = global.Render.esc;
  var calm = global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* header: the annotator's tasks as stages -- done / current / upcoming */
  function stages(host, tasks, currentTask) {
    var passed = true;
    host.innerHTML = tasks.map(function (t) {
      var done = t.total && t.done >= t.total;
      var cur = !done && t.task === currentTask;
      var cls = done ? "done" : (cur ? "cur" : (passed ? "cur" : "next"));
      if (!done) { passed = false; }
      var pct = t.total ? Math.round(100 * t.done / t.total) : 0;
      return '<span class="stage ' + cls + '" title="' + esc(t.title) + ": " + t.done +
        " of " + t.total + ' conversations">' +
        '<span class="ic">' + (done ? "&#10003;" : (cls === "cur" ? "&#9654;" : "&#9675;")) +
        "</span>" + (t.emoji ? '<span class="em">' + t.emoji + "</span>" : "") +
        "<span class=nm>" + esc(t.title) + "</span>" +
        '<span class="ct">' + t.done + "/" + t.total + "</span>" +
        (cls === "cur" ? '<span class="mini"><i style="width:' + pct + '%"></i></span>' : "") +
        "</span>";
    }).join('<span class="sep" aria-hidden="true"></span>');
  }

  function toast(html, ms) {
    var box = document.getElementById("toasts");
    var el = document.createElement("div");
    el.className = "toast";
    el.innerHTML = html;
    box.appendChild(el);
    setTimeout(function () { el.classList.add("out"); }, ms || 3200);
    setTimeout(function () { el.remove(); }, (ms || 3200) + 500);
  }

  function confetti() {
    if (calm) { return; }
    var box = document.getElementById("confetti");
    var colors = ["#2563eb", "#f0b429", "#16a34a", "#e11d48", "#7c3aed", "#0ea5e9"];
    for (var i = 0; i < 110; i++) {
      var p = document.createElement("i");
      p.style.left = (Math.random() * 100) + "vw";
      p.style.background = colors[i % colors.length];
      p.style.animationDelay = (Math.random() * 0.6) + "s";
      p.style.animationDuration = (2.2 + Math.random() * 1.6) + "s";
      p.style.transform = "rotate(" + (Math.random() * 360) + "deg)";
      if (i % 3 === 0) { p.style.borderRadius = "50%"; }
      box.appendChild(p);
    }
    setTimeout(function () { box.innerHTML = ""; }, 4600);
  }

  /* a small puff of paper from an element, and a label floating up: shown
     after EVERY finished conversation, identical whatever was answered */
  function burst(el, label) {
    var r = el.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
    var box = document.getElementById("confetti");
    var tag = document.createElement("b");
    tag.className = "burst-label";
    tag.textContent = label || "\u2713";
    tag.style.left = x + "px"; tag.style.top = (y - 10) + "px";
    box.appendChild(tag);
    setTimeout(function () { tag.remove(); }, 1300);
    if (calm) { return; }
    var colors = ["#2f62e6", "#7c5cff", "#f5a524", "#17a673", "#ff7a59"];
    for (var i = 0; i < 26; i++) {
      var p = document.createElement("i");
      p.className = "bit";
      var a = Math.random() * Math.PI - Math.PI;        // upwards half-circle
      var d = 60 + Math.random() * 90;
      p.style.left = x + "px"; p.style.top = y + "px";
      p.style.background = colors[i % colors.length];
      p.style.setProperty("--dx", Math.cos(a) * d + "px");
      p.style.setProperty("--dy", Math.sin(a) * d + "px");
      p.style.setProperty("--rot", (Math.random() * 540 - 270) + "deg");
      box.appendChild(p);
      (function (n) { setTimeout(function () { n.remove(); }, 1100); })(p);
    }
  }

  /* chocolates: a real reward, a fixed number per batch paid out per finished
     conversation -- so, like burst(), it follows progress and nothing else */
  function chocoAmount(x) {             // 10 -> "10", 0.5 -> "0.5", 0.04 -> "0.04"
    var r = Math.round(x * 10) / 10;
    return String(Math.abs(r - Math.round(r)) < 1e-9 ? Math.round(r) : (r || x.toFixed(2)));
  }
  /* a chocolate flies from one element into another; then() once it lands */
  function chocolate(from, to, then) {
    if (calm || !from || !to) { then(); return; }
    var a = from.getBoundingClientRect(), b = to.getBoundingClientRect();
    var c = document.createElement("b");
    c.className = "choco-fly";
    c.textContent = "\uD83C\uDF6B";
    c.style.left = (a.left + a.width / 2) + "px"; c.style.top = (a.top + a.height / 2) + "px";
    c.style.setProperty("--dx", (b.left + b.width / 2) - (a.left + a.width / 2) + "px");
    c.style.setProperty("--dy", (b.top + b.height / 2) - (a.top + a.height / 2) + "px");
    document.getElementById("confetti").appendChild(c);
    setTimeout(function () {
      c.remove();
      then();
      to.classList.remove("bump"); void to.offsetWidth; to.classList.add("bump");
    }, 900);
  }
  /* the end: chocolates rain down while the total counts up to n */
  function chocoFinale(numEl, iconEl, n) {
    numEl.textContent = calm ? n : 0;
    if (calm) { return; }
    var box = document.getElementById("confetti");
    var k = Math.min(45, Math.max(14, n * 2));
    for (var i = 0; i < k; i++) {
      var c = document.createElement("b");
      c.className = "choco-rain";
      c.textContent = "\uD83C\uDF6B";
      c.style.left = (Math.random() * 96) + "vw";
      c.style.fontSize = (20 + Math.random() * 18) + "px";
      c.style.animationDelay = (Math.random() * 1.4) + "s";
      c.style.animationDuration = (2.2 + Math.random() * 1.8) + "s";
      c.style.setProperty("--rot", (Math.random() * 720 - 360) + "deg");
      box.appendChild(c);
      (function (x) { setTimeout(function () { x.remove(); }, 5600); })(c);
    }
    var t0 = null, dur = Math.min(2200, 600 + n * 60);
    function tick(t) {
      if (t0 === null) { t0 = t; }
      var p = Math.min(1, (t - t0) / dur);
      numEl.textContent = Math.round(n * (1 - Math.pow(1 - p, 3)));   // ease-out
      if (p < 1) { return requestAnimationFrame(tick); }
      iconEl.classList.add("bump");
    }
    setTimeout(function () { requestAnimationFrame(tick); }, 400);
  }

  /* compare two progress snapshots (state.tasks) and say something nice about
     what changed. Returns the task that was just finished, if any. */
  function milestones(before, after) {
    var finished = null;
    (after || []).forEach(function (t) {
      var b = (before || []).filter(function (x) { return x.task === t.task; })[0];
      if (!b || b.done === t.done || !t.total) { return; }
      var half = Math.ceil(t.total / 2);
      if (t.done >= t.total) { finished = t; }
      else if (b.done < half && t.done >= half) {
        toast("Halfway through <b>" + esc(t.title) + "</b> &mdash; nice going!");
      } else if (t.total - t.done === 1) {
        toast("One more in <b>" + esc(t.title) + "</b>!");
      } else if (t.done % 5 === 0) {
        toast(t.done + " done in <b>" + esc(t.title) + "</b> &#10003;", 2200);
      }
    });
    return finished;
  }

  global.Fun = { stages: stages, toast: toast, confetti: confetti, burst: burst,
                 milestones: milestones, chocolate: chocolate, chocoAmount: chocoAmount,
                 chocoFinale: chocoFinale, calm: calm };
})(window);
