// SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
// SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
// SPDX-License-Identifier: MIT

/* DialFork annotation client.
   The server decides which step to serve and what it may contain; this file
   only renders what it is given. Never infer ground truth here. */
(function () {
  "use strict";

  // path prefix when served behind a reverse proxy ("" when accessed directly)
  var BASE = window.BASE || "";
  var TOKEN = location.pathname.slice(BASE.length).replace(/^\/a\//, "");
  var state = null, item = null, answer = {}, tStart = 0;

  var $ = function (id) { return document.getElementById(id); };
  function post(path, body) {
    body = body || {};
    body.token = TOKEN;
    return fetch(BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (r) {
      return r.json().then(function (j) {
        if (!r.ok) { throw new Error(j.error || ("HTTP " + r.status)); }
        return j;
      });
    });
  }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  /* ---------- workflow steps (markup shared with admin: render.js) ---------- */
  function renderFlow(fl) {
    $("flow").innerHTML = Render.flow(fl);
    Render.wireFlow($("flow"));
  }

  /* ---------- transcript ---------- */
  function renderTranscript(opts) {
    opts = opts || {};
    var html = Render.transcript(item.turns, {
      bandFrom: opts.bandFrom, pick: opts.pick, chosen: answer.onset_sys_turn });
    var el = $("transcript");
    el.innerHTML = html;
    var pb = $("pickBanner");
    pb.hidden = !opts.pick;
    if (opts.pick) {
      pb.className = "pickbanner" + (answer.onset_sys_turn ? " ok" : "");
      pb.innerHTML = Render.roles(answer.onset_sys_turn
        ? "\u2713 It starts at <b>assistant turn " + answer.onset_sys_turn +
          "</b> &mdash; click another assistant turn to change"
        : "\uD83D\uDC47 <b>Click the first assistant turn where it starts to happen</b>");
    }
    if (opts.pick) {
      el.querySelectorAll(".turn.pick").forEach(function (n) {
        n.addEventListener("click", function () {
          answer.onset_sys_turn = parseInt(n.dataset.n, 10);
          answer.appears = answer.appears || "yes";
          renderStep2();
          renderTranscript({ pick: true });
        });
      });
    }
  }

  /* ---------- steps ---------- */
  function choices(name, opts, current) {
    return '<div class="choices">' + opts.map(function (o) {
      return "<label><input type=radio name=" + name + " value=" + o[0] +
             (current === o[0] ? " checked" : "") + "><span>" + esc(o[1]) + "</span></label>";
    }).join("") + "</div>";
  }

  function renderStep1() {
    setStep(1);
    $("cueBox").innerHTML = "";
    ask("The assistant was told to follow the steps on the left. How well does it?");
    var v = answer.slider == null ? 50 : answer.slider;
    $("controls").innerHTML =
      '<div class="row"><div><input type=range id="sl" min=0 max=100 value=' + v + ">" +
      '<div class="anchors"><span>0 &mdash; ignores them</span>' +
      "<span>follows them &mdash; 100</span></div></div>" +
      '<div class="sliderval" id="slv">' + v + "</div></div>";
    var sl = $("sl");
    // Continue waits until the slider has actually been used, so an untouched
    // default of 50 is never recorded by a stray click (landing on 50 is fine)
    function touched() {
      answer.slider = parseInt(sl.value, 10);
      answer.sliderTouched = true;
      $("slv").textContent = sl.value;
      $("slv").classList.remove("untouched");
      $("hint").textContent = "";
      setNext(true);
    }
    ["input", "change", "pointerdown", "keydown"].forEach(function (ev) {
      sl.addEventListener(ev, touched);
    });
    if (!answer.sliderTouched) {
      $("slv").classList.add("untouched");
      $("hint").textContent = "Move the slider to your answer first.";
    }
    setNext(!!answer.sliderTouched);
  }

  function renderStep2() {
    setStep(2);
    $("cueBox").innerHTML =
      '<div class="cue">' + Render.roles(esc(item.cue.lead)) +
      '<span class="ins">' + esc(item.cue.instruction) + "</span></div>";
    ask("Does that happen in this conversation?");
    var c = choices("appears",
      [["yes", "Yes"], ["partly", "Partly"], ["no", "No, that does not happen"]],
      answer.appears);
    // After yes/partly the answer is not complete until a turn is picked in the
    // transcript -- say so loudly, here and in the conversation panel
    var picking = answer.appears === "yes" || answer.appears === "partly";
    var pickMsg = !picking ? "" : (answer.onset_sys_turn
      ? '<div class="pickcall ok">\u2713 It starts at <b>assistant turn ' +
        answer.onset_sys_turn + "</b> &mdash; click another turn to change</div>"
      : '<div class="pickcall need">\uD83D\uDC46 <b>Now click the assistant turn where it ' +
        "first starts to happen</b> &mdash; the point from which it changes</div>");
    $("controls").innerHTML = c + Render.roles(pickMsg);
    $("controls").querySelectorAll("input[name=appears]").forEach(function (r) {
      r.addEventListener("change", function () {
        answer.appears = r.value;
        if (r.value === "no") { answer.onset_sys_turn = null; }
        renderStep2();
        renderTranscript({ pick: answer.appears !== "no" });
      });
    });
    $("hint").textContent = "";
    setNext(answer.appears === "no" ||
            (!!answer.appears && !!answer.onset_sys_turn));
  }

  /* Your answer next to what really happened: neutral, no score, no praise,
     no colour for right/wrong. Information, never a verdict (see fun.js). */
  function compareHtml(r, y) {
    var said = { yes: "Yes", partly: "Partly", no: "No" }[y.appears] || "\u2014";
    var yours = said + (y.onset ? ", from assistant turn " + y.onset : "");
    var truth;
    if (r.kind === "conformant") {
      truth = "Nothing was injected";
    } else if (r.is_foil) {
      truth = "Not given \u2014 what we asked about was a decoy. Something else was " +
              "injected (see below).";
    } else {
      truth = r.fork_assistant_turn === 1 ? "Given from the very start, before assistant turn 1"
                                : "Given, just before assistant turn " + r.fork_assistant_turn;
    }
    return '<div class="compare"><div><span class="lab">You said</span>' + esc(yours) +
      '</div><div><span class="lab">What really happened</span>' + esc(truth) +
      "</div></div>";
  }

  function renderStep3() {
    var r = item.reveal;
    setStep(3);
    // role styling for OUR words only; the instruction is quoted as it was given
    var box = '<div class="reveal-head">Here\'s what was really going on</div>\u0001',
        ins = "";
    if (r.true_cue) {
      ins = "<span class=ins>" + esc(r.true_cue.instruction) + "</span>";
      box += "<b>The extra instruction it was really given:</b>\u0000" +
            "<div style='margin-top:6px;font-size:13px'>It was given just before " +
            "<b>assistant turn " + r.fork_assistant_turn + "</b>, and applied from there on " +
            "&mdash; highlighted in the conversation.</div>";
    } else {
      box += "<b>This conversation was never altered.</b>" +
            "<div style='margin-top:6px;font-size:13px'>The assistant was only ever " +
            "told to follow the steps on the left. Nothing extra was injected.</div>";
    }
    var sub = "";
    if (r.substitute) {
      sub = "<details style='margin-top:8px'><summary>The other procedure it was switched to" +
            "</summary><div style='font-size:13px'><b>" + esc(r.substitute.name) +
            "</b> &mdash; " + esc(r.substitute.description) +
            Render.flow(r.substitute.flow) + "</div></details>";
    }
    $("cueBox").innerHTML = '<div class="cue reveal">' +
      Render.roles(box).replace("\u0001", compareHtml(r, item.yours || {}))
                       .replace("\u0000", ins) + sub + "</div>";
    Render.wireFlow($("cueBox"));

    // Say plainly what is left to do. The reveal is information, never a
    // verdict on the answers just given (see fun.js: reward effort, not answers).
    // nothing before the fork to judge: conformant, or injected before turn 1
    var noPrefix = r.kind === "conformant" || r.fork_assistant_turn === 1;
    if (noPrefix) {
      ask("That's it for this one \u2014 nothing else to answer.");
      $("controls").innerHTML =
        "<p class='hint' style='margin:0 0 8px'>" + (r.kind === "conformant" ? "" :
          "The extra instruction was there from the very first assistant turn, so " +
          "there is nothing before it to check. ") +
        "If anything stood out, leave a note. Otherwise, on to the next conversation!</p>" +
        "<textarea id='note' placeholder='Optional note'>" + esc(answer.note || "") +
        "</textarea>";
      answer.prefix_ok = "n/a";
    } else {
      ask("One last question: before the highlighted part, does the assistant follow " +
          "the steps on the left?");
      $("controls").innerHTML =
        choices("prefix", [["yes", "Yes"], ["mostly", "Mostly"], ["no", "No"]],
                answer.prefix_ok) +
        "<textarea id='note' style='margin-top:10px' placeholder='" +
        "Optional note: anything that went wrong, or that the description above did " +
        "not match'>" + esc(answer.note || "") + "</textarea>";
      $("controls").querySelectorAll("input[name=prefix]").forEach(function (el) {
        el.addEventListener("change", function () {
          answer.prefix_ok = el.value;
          setNext(true);
        });
      });
    }
    $("note").addEventListener("input", function () { answer.note = $("note").value; });
    $("hint").textContent = "";
    setNext(noPrefix || !!answer.prefix_ok);
  }

  function setNext(ok) {
    $("next").disabled = !ok;
    var label = "Continue";
    if (item && item.step === 2 && !ok &&
        (answer.appears === "yes" || answer.appears === "partly")) {
      label = "Click a turn first \u2191";
    }
    if (item && item.step === 3 && state && state.current) {
      var t = (state.tasks || []).filter(function (x) { return x.task === item.task; })[0];
      label = t && t.total - t.done <= 1 ? "Finish this task \u2192" : "Next conversation \u2192";
    }
    $("next").textContent = label;
  }
  function ask(text) { $("question").innerHTML = Render.roles(esc(text)); }

  var STEP_TITLE = ["", "Your overall impression", "Look for something specific",
                    "The earlier part"];
  function setStep(n) {
    var dock = $("dock");
    if (dock.dataset.step !== String(n)) {        // a small hop when the question changes
      dock.dataset.step = n;
      dock.classList.remove("swap"); void dock.offsetWidth; dock.classList.add("swap");
    }
    $("stepno").innerHTML = "<b>Step " + n + " of 3</b> \u00b7 " + STEP_TITLE[n];
    $("stepdots").innerHTML = [1, 2, 3].map(function (i) {
      return "<i class='" + (i === n ? "on" : (i < n ? "past" : "")) + "'></i>";
    }).join("");
  }

  /* ---------- flow ---------- */
  function loadItem() {
    return post("/api/item", { item_id: state.current.item_id }).then(function (d) {
      if (d.done) { return refresh(); }
      item = d;
      answer = {};
      tStart = Date.now();
      $("taskEmoji").textContent = d.emoji || "";
      $("taskName").textContent = d.topic ? d.topic.charAt(0).toUpperCase() + d.topic.slice(1)
                                          : d.task_title;
      $("taskDesc").textContent = d.task_description;
      $("convTask").textContent = (d.emoji ? d.emoji + " " : "") + d.task_title;
      var ct = (state.tasks || []).filter(function (x) { return x.task === d.task; })[0];
      $("convTitle").textContent = ct
        ? "Conversation " + Math.min(ct.done + 1, ct.total) + " of " + ct.total
        : "Conversation";
      $("userGoal").textContent = d.user_goal || "(not recorded)";
      renderFlow(d.flow);
      $("flowWrap").hidden = !d.has_flowchart;
      if (d.has_flowchart) { $("flowImg").src = BASE + "/flow/" + d.task + ".jpg"; }
      $("err").textContent = "";
      $("dock").hidden = false;
      $("work").classList.remove("enter");
      void $("work").offsetWidth;          // restart the entrance animation
      $("work").classList.add("enter");
      if (d.step === 1) { renderTranscript({}); renderStep1(); }
      else if (d.step === 2) { renderTranscript({ pick: false }); renderStep2(); }
      else { renderTranscript({ bandFrom: d.reveal.fork_assistant_turn }); renderStep3(); }
      window.scrollTo(0, 0);
    });
  }

  function submit() {
    var payload;
    if (item.step === 1) { payload = { slider: answer.slider }; }
    else if (item.step === 2) {
      payload = { appears: answer.appears,
                  onset_sys_turn: answer.appears === "no" ? null : answer.onset_sys_turn };
    } else { payload = { prefix_ok: answer.prefix_ok || "n/a", note: answer.note || "" }; }
    setNext(false);
    return post("/api/answer", {
      item_id: item.item_id, step: item.step, payload: payload,
      ms: Date.now() - tStart
    }).then(function (s) {
      var before = state.tasks, beforeC = state.chocolates;
      state = s;
      var gain = chocoGain(beforeC, s.chocolates);
      if (gain > 0) { heldChoco = beforeC; }   // the counter moves when the chocolate lands
      paintProgress();
      if (item.step === 3) {
        // the same little burst after every conversation, whatever was answered
        Fun.burst($("next"), "\u2713 done" +
                  (gain > 0 ? " \u00b7 +" + Fun.chocoAmount(gain) + " \uD83C\uDF6B" : ""));
        if (gain > 0) {
          Fun.chocolate($("next"), $("choco"), function () { heldChoco = null; paintMe(); });
        }
        var finished = Fun.milestones(before, s.tasks);
        if (finished) { return celebrate(finished); }
      }
      return route();
    }).catch(function (e) {
      $("err").textContent = e.message;
      setNext(true);
    });
  }

  /* progress is shown per task: the stages in the header, and where we are
     inside the current one */
  // the name exactly as the admin entered it ("Maria", "Annotator 3", a
  // nickname...): the admin decides how each person is greeted
  function firstName() {
    return ((state && state.annotator && state.annotator.name) || "").trim();
  }
  /* chocolates earned, whole and the share of the next one, as one number */
  function chocoGain(a, b) {
    if (!a || !b || !b.total) { return 0; }
    return (b.count + b.next) - (a.count + a.next);
  }
  var heldChoco = null;   // what the counter shows while a chocolate is in flight
  function chocoHtml() {
    var c = heldChoco || state.chocolates;
    if (!c || !c.total) { return ""; }
    var pct = Math.round(100 * c.next);
    return '<span class="choco" id="choco" title="' + c.count + " chocolate" +
      (c.count === 1 ? "" : "s") + " earned of " + c.total + "; the bar is " +
      'the next one">\uD83C\uDF6B<b>' + c.count + "</b>" +
      '<i class="fill"><i style="width:' + pct + '%"></i></i></span>';
  }
  /* telling people up front, once: on the welcome (consent) screen */
  function chocoPitch(el) {
    var c = state && state.chocolates;
    el.hidden = !(c && c.total);
    if (el.hidden) { return; }
    el.innerHTML = "<span>\uD83C\uDF6B</span><span><b>You collect real chocolates!</b> " +
      "Your conversations are worth <b>" + c.total + "</b> in all, and every " +
      "conversation you finish earns your share of them, whatever you answer. Watch them " +
      "pile up next to your name; when you are done, show the final count to claim them." +
      (c.count ? " You have <b>" + c.count + "</b> so far." : "") + "</span>";
  }
  /* the person, in the header: initials, name, one medal per finished task,
     chocolates earned */
  function paintMe() {
    var name = (state.annotator && state.annotator.name) || "";
    var done = (state.tasks || []).filter(function (t) {
      return t.total && t.done >= t.total; }).length;
    var ini = name.trim().split(/\s+/).map(function (w) { return w.charAt(0); })
                  .join("").slice(0, 2).toUpperCase();
    $("me").innerHTML = '<span class="av">' + Render.esc(ini || "?") + "</span>" +
      '<span class="nm">' + Render.esc(name) + "</span>" +
      (done ? '<span class="medals" title="' + done + " task" + (done > 1 ? "s" : "") +
              ' finished">\uD83C\uDFC5\u00d7' + done + "</span>" : "") + chocoHtml();
    $("me").hidden = !name;
  }
  function paintProgress() {
    paintMe();
    var cur = state.current && state.current.task;
    Fun.stages($("stages"), state.tasks || [], cur);
    var t = (state.tasks || []).filter(function (x) { return x.task === cur; })[0];
    $("meta").textContent = "";     // the count lives on the stage chip and the panel title
  }

  /* where to go next: the intro of a task not seen yet, else the next item */
  function route() {
    if (!state.current) { return refresh(); }
    if (state.current.need_intro) { return showIntro(state.current.task, false); }
    return loadItem();
  }

  /* ---------- learning a task: three short slides ---------- */
  var introAt = 0, introTask = null, introReview = false;
  var ORD = ["first", "second", "third", "fourth", "fifth", "sixth"];
  var NUM = ["one", "two", "three", "four", "five", "six"];
  function hello(i, n) {
    if (n <= 1) { return "Here's your task!"; }
    var of = " (of " + (NUM[n - 1] || n) + ")";
    if (i === 0) { return "Let's begin with your first task" + of + "!"; }
    if (i === n - 1) { return "On to your last task" + of + "!"; }
    return "On to your " + (ORD[i] || (i + 1) + "th") + " task" + of + "!";
  }
  function goSlide(k) {
    document.querySelectorAll("#intro .slide").forEach(function (el) {
      el.hidden = +el.dataset.slide !== k;
      if (!el.hidden) {                       // replay the entrance for this slide
        el.classList.remove("in"); void el.offsetWidth; el.classList.add("in");
      }
    });
    $("introDots").innerHTML = [0, 1, 2].map(function (i) {
      return '<i class="' + (i === k ? "on" : (i < k ? "past" : "")) + '"></i>';
    }).join("");
    window.scrollTo(0, 0);
  }
  document.querySelectorAll("#intro [data-go]").forEach(function (b) {
    b.addEventListener("click", function () { goSlide(+b.dataset.go); });
  });
  function showIntro(task, reviewing) {
    return post("/api/intro", { task: task }).then(function (d) {
      introTask = task; introReview = reviewing; introAt = Date.now();
      var ts = state.tasks || [], i = 0;
      ts.forEach(function (x, k) { if (x.task === task) { i = k; } });
      $("introEmoji").textContent = d.emoji || "\u2728";
      $("introKicker").textContent = ts.length > 1
        ? "Task " + (i + 1) + " of " + ts.length + " \u00b7 " + d.task_title : d.task_title;
      var fn = firstName();
      var h = hello(i, ts.length);
      $("introHello").textContent = fn && !reviewing
        ? "Hi " + fn + "! " + h.charAt(0).toUpperCase() + h.slice(1) : h;
      var job = d.task_description ? d.task_description.charAt(0).toLowerCase() +
                                     d.task_description.slice(1) : "";
      $("introAbout").innerHTML = "It's all about <b>" + Render.esc(d.topic) + "</b>" +
        (job ? ": the " + Render.roles("assistant") + "'s job is to " + Render.esc(job) : ".");
      $("introStepsKicker").textContent = (d.emoji ? d.emoji + "  " : "") +
        (reviewing ? "A quick refresher" : "The steps") + " \u00b7 " + d.task_title;
      $("introFlow").innerHTML = Render.flow(d.flow, { stagger: true });
      Render.wireFlow($("introFlow"));
      $("introFlowWrap").hidden = !d.has_flowchart;
      if (d.has_flowchart) {
        $("introFlowImg").src = $("introFlowLink").href = BASE + "/flow/" + task + ".jpg";
      }
      $("introGo").textContent = reviewing ? "Back to the conversation"
                                           : "Got it \u2014 let's start!";
      $("introSkip").hidden = $("introBack").hidden = $("introDots").hidden = reviewing;
      $("work").hidden = true; $("dock").hidden = true; $("intro").hidden = false;
      goSlide(reviewing ? 2 : 0);
    });
  }
  function leaveIntro(action) {
    $("introGo").disabled = $("introSkip").disabled = true;
    return post("/api/intro_done", { task: introTask, action: action,
                                     ms: Date.now() - introAt })
      .then(function (s) {
        state = s;
        $("intro").hidden = true; $("work").hidden = false;
        paintProgress();
        if (!introReview) {
          Fun.toast("Let's go &mdash; <b>" + Render.esc(introTask.replace(/_/g, " ")) +
                    "</b>, " + ((state.tasks || []).filter(function (x) {
                      return x.task === introTask; })[0] || {}).total + " conversations");
        }
        if (introReview && item) {       // back to the conversation, as it was
          $("dock").hidden = false;
          return null;
        }
        return loadItem();
      })
      .then(function () { $("introGo").disabled = $("introSkip").disabled = false; });
  }
  $("introGo").addEventListener("click", function () {
    leaveIntro(introReview ? "reviewed" : "read");
  });
  $("introSkip").addEventListener("click", function () { leaveIntro("skipped"); });
  $("reviewIntro").addEventListener("click", function () {
    if (state.current) { showIntro(state.current.task, true); }
  });

  /* ---------- a task finished ---------- */
  function celebrate(t) {
    Fun.confetti();
    var next = state.current;
    $("celTitle").textContent = (t.emoji ? t.emoji + " " : "") +
      t.title.charAt(0).toUpperCase() + t.title.slice(1) + " \u2014 done!";
    var c = state.chocolates;
    $("celBody").innerHTML = "You judged all <b>" + t.total + "</b> conversations of this " +
      "task. Thank you!" + (c && c.total ? "<br>Chocolates so far: <b>" + c.count +
      " \uD83C\uDF6B</b>" : "") + (next ? "<br>Next up: <b>" +
      Render.esc((next.task || "").replace(/_/g, " ")) + "</b>." : "");
    $("celNext").textContent = next ? "On to the next task \u2192" : "Finish";
    $("work").hidden = true; $("dock").hidden = true; $("celebrate").hidden = false;
    return Promise.resolve();
  }
  $("celNext").addEventListener("click", function () {
    $("celebrate").hidden = true;
    route();
  });

  function refresh() {
    return post("/api/state").then(function (s) {
      state = s;
      paintProgress();
      if (!s.annotator.consented) {
        if (firstName()) { $("consentHello").textContent = "Welcome, " + firstName() + "!"; }
        chocoPitch($("consentChoco"));
        $("consent").hidden = false; $("work").hidden = true;
        $("dock").hidden = true; $("finished").hidden = true;
        return;
      }
      $("consent").hidden = true;
      if (!s.current) {
        $("work").hidden = true; $("dock").hidden = true; $("intro").hidden = true;
        $("finishedBody").innerHTML = (s.tasks || []).length
          ? '<div class="badges">' + s.tasks.map(function (t) {
              return '<span class="badge"><span class="ic">&#10003;</span>' +
                Render.esc(t.title) + "<small>" + t.total + " conversations</small></span>";
            }).join("") + "</div>" : "";
        var c = s.chocolates;
        if (c && c.total && c.count) {
          // first thing under the title: it is what they take away from this screen
          $("finishedBody").innerHTML = '<div class="choco-total"><span id="chocoEndIc">' +
            "\uD83C\uDF6B</span><div>Go and claim your <b>" + '<span id="chocoEndN">' + c.count +
            "</span> chocolate" + (c.count === 1 ? "" : "s") +
            "</b>!<small>Show this screen when you collect them. Your link keeps showing " +
            "this count, so it is safe to close the tab.</small></div></div>" +
            $("finishedBody").innerHTML;
        }
        $("finished").hidden = false;
        if ($("chocoEndN")) { Fun.chocoFinale($("chocoEndN"), $("chocoEndIc"), c.count); }
        return;
      }
      $("finished").hidden = true; $("work").hidden = false;
      return route();
    });
  }

  $("consentBtn").addEventListener("click", function () {
    post("/api/consent").then(function (s) { state = s; return refresh(); });
  });
  $("next").addEventListener("click", submit);
  /* the question card floats over the page and changes height from step to
     step; keep its height in --dockh so the side panel and the transcript
     always leave room for it and nothing hides behind it */
  if (window.ResizeObserver) {
    new ResizeObserver(function () {
      var h = $("dock").hidden ? 0 : $("dock").offsetHeight + 28;
      document.documentElement.style.setProperty("--dockh", h + "px");
    }).observe($("dock"));
  }
  document.addEventListener("keydown", function (e) {
    // Ctrl/Cmd+Enter only: a bare Enter was the easiest way to submit by accident
    if (e.key !== "Enter" || !(e.ctrlKey || e.metaKey)) { return; }
    if (!$("dock").hidden && !$("next").disabled) { e.preventDefault(); submit(); }
  });

  var greeted = false;
  function welcomeBack() {
    if (greeted || !state || !state.current || state.current.need_intro) { return; }
    greeted = true;
    var t = (state.tasks || []).filter(function (x) { return x.task === state.current.task; })[0];
    if (t && t.done > 0) {
      Fun.toast("Welcome back" + (firstName() ? ", <b>" + Render.esc(firstName()) + "</b>" : "") +
                "! " + (t.total - t.done) + " to go in " + Render.esc(t.title) + ".", 4200);
    }
  }
  refresh().then(welcomeBack).catch(function (e) {
    document.body.innerHTML =
      '<div class="center card"><h1>Cannot load</h1><p>' + esc(e.message) +
      "</p><p>Check that you opened the exact link you were sent.</p></div>";
  });
})();
