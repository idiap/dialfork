// SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
// SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
// SPDX-License-Identifier: MIT

(function () {
  "use strict";
  var data = null;
  // path prefix when served behind a reverse proxy ("" when accessed directly)
  var BASE = window.BASE || "";
  var $ = function (id) { return document.getElementById(id); };

  /* The session is an HttpOnly cookie the browser sends by itself. A 401 means
     it expired (back to the login page); a must_change 403 means the public
     default password is still in use (nothing works until it is changed). */
  function post(path, body) {
    return fetch(BASE + path, { method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) })
      .then(function (r) { return r.json().then(function (j) {
        if (r.status === 401) { location.replace(BASE + "/admin"); }
        if (j.must_change) { openPassword(true); }
        if (!r.ok) { var e = new Error(j.error || r.status); e.status = r.status; throw e; }
        return j; }); });
  }

  /* ---------- password ---------- */
  var forced = false;
  function openPassword(mandatory) {
    forced = mandatory;
    $("pwTitle").textContent = mandatory ? "Choose your own admin password"
                                         : "Change the admin password";
    $("pwWhy").textContent = mandatory
      ? "You are using the default password. The code is public, so the default is too: " +
        "it only works for this first login. Pick your own to open the panel."
      : "Changing it logs out every other admin session.";
    $("pwCancel").hidden = mandatory;
    $("pwErr").textContent = "";
    $("pwModal").hidden = false;
    $("pwCur").focus();
  }
  $("pwBtn").addEventListener("click", function () { openPassword(false); });
  $("pwCancel").addEventListener("click", function () { $("pwModal").hidden = true; });
  $("pwForm").addEventListener("submit", function (e) {
    e.preventDefault();
    if ($("pwNew").value !== $("pwNew2").value) {
      $("pwErr").textContent = "The two new passwords do not match."; return;
    }
    $("pwSave").disabled = true;
    post("/api/admin/password", { current: $("pwCur").value, "new": $("pwNew").value })
      .then(function () {
        $("pwModal").hidden = true;
        ["pwCur", "pwNew", "pwNew2"].forEach(function (i) { $(i).value = ""; });
        flash("ok", "Password changed. Other admin sessions were logged out.");
        if (forced) { forced = false; refresh().then(startPolling); }
      })
      .catch(function (err) { $("pwErr").textContent = err.message; })
      .then(function () { $("pwSave").disabled = false; });
  });
  $("logout").addEventListener("click", function () {
    post("/api/admin/logout").then(function () { location.replace(BASE + "/admin"); });
  });
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  /* ---------- flash ---------- */
  var timer = null;
  function flash(kind, html, keep) {
    clearTimeout(timer);
    $("flash").innerHTML = '<div class="msg ' + kind + '">' + html + "</div>";
    if (!keep) { timer = setTimeout(function () { $("flash").innerHTML = ""; }, 6000); }
  }
  function clearFlash() { clearTimeout(timer); $("flash").innerHTML = ""; }

  /* Any destructive call: run it, and if the server asks for confirmation,
     show what would be lost and let the user decide. */
  function act(path, body, okMsg) {
    return post(path, body)
      .then(function (d) { data = d; paint(); flash("ok", okMsg(d)); })
      .catch(function (e) {
        if (e.status === 409) {
          flash("warn", esc(e.message) +
            '<div class="row"><button class="btn danger sm" id="yes">Yes, do it</button>' +
            '<button class="btn ghost sm" id="no">Cancel</button></div>', true);
          $("yes").addEventListener("click", function () {
            clearFlash();
            body.force = true;
            act(path, body, okMsg);
          });
          $("no").addEventListener("click", clearFlash);
        } else { flash("bad", esc(e.message)); }
      });
  }

  /* ---------- pieces ---------- */
  function tile(k, v, s, icon) {
    return "<div class=tile>" + (icon ? "<span class=ti>" + icon + "</span>" : "") +
           "<div class=k>" + esc(k) + "</div><div class=v>" +
           esc(v) + "</div><div class=s>" + esc(s) + "</div></div>";
  }
  function progress(done, total) {
    var pct = total ? Math.round(100 * done / total) : 0;
    return '<div class="prog' + (total && done >= total ? " full" : "") +
           '"><span class="track"><i style="width:' + pct +
           '%"></i></span><span class="pct">' + done + " / " + total + "</span></div>";
  }
  function compText(c) {
    return c.segments.map(function (n, i) { return "k" + (i + 1) + "·" + n; })
             .join("  ") + "   conf·" + c.conformant +
           (c.one_per_source ? "" : "   ⚠ repeats a source");
  }
  /* balanced (as every block is, by construction) -> one small chip, details on
     hover; anything unusual is spelled out */
  function compChip(c, size) {
    var even = c.segments.every(function (n) { return n === c.segments[0]; }) &&
               c.conformant * 5 === size && c.one_per_source;
    return even
      ? '<span class="okchip" title="' + esc(compText(c)) + '">\u2713 balanced</span>'
      : '<span class="hint">' + esc(compText(c)) + "</span>";
  }

  /* ---------- overview ---------- */
  function paintTiles() {
    var cov = data.coverage, n = data.n_items;
    var done = data.annotators.reduce(function (a, x) { return a + x.done; }, 0);
    var assigned = data.annotators.reduce(function (a, x) { return a + x.assigned; }, 0);
    $("tiles").innerHTML =
      tile("Conversations", n, "drawn into batches so far", "\uD83D\uDCAC") +
      tile("Judged once", cov["1"] + cov["2+"], "of " + n + " covered", "\u2705") +
      tile("Judged twice", cov["2+"], "feeds the agreement statistic", "\uD83D\uDC65") +
      tile("Answers in", done + " / " + assigned, "of what is assigned", "\uD83D\uDCDD");
    $("meta").textContent = data.responses + " answers" +
      (data.median_seconds_per_step
        ? " · median " + data.median_seconds_per_step.toFixed(0) + "s per question" : "");
  }

  /* ---------- annotators ---------- */
  /* where annotators reach the server: --public-url if the server was given one,
     else wherever this admin page is open (fine when that address is public) */
  function linkBase() {
    return (data && data.public_url) || location.origin + BASE;
  }
  function localLinks() {
    return /^https?:\/\/(localhost|127\.|\[::1\]|0\.0\.0\.0)/.test(linkBase());
  }
  /* whether each task's intro was read (and for how long) or skipped */
  /* 77000 -> "1 min 17 s" */
  function dur(ms) {
    var sec = Math.round(ms / 1000);
    if (sec < 60) { return sec + " s"; }
    var m = Math.floor(sec / 60); sec %= 60;
    if (m < 60) { return m + " min" + (sec ? " " + sec + " s" : ""); }
    return Math.floor(m / 60) + " h" + (m % 60 ? " " + (m % 60) + " min" : "");
  }
  /* where each person is: never opened their link / opened it, nothing finished
     yet / some finished / everything they hold finished */
  var STATUS = { notstarted: "not started", started: "started", ongoing: "ongoing",
                 done: "done" };
  function status(a) {
    if (!a.consented) { return "notstarted"; }
    if (!a.done) { return "started"; }
    return a.assigned && a.done >= a.assigned ? "done" : "ongoing";
  }
  /* the people table is grouped by status, in the order people move through it;
     any group can be folded (remembered in this browser only). Not started and
     started share a group: either way nothing is finished yet, so they need a nudge */
  var GROUPS = [{ key: "todo", label: "nothing finished yet", members: ["notstarted", "started"] },
                { key: "ongoing", label: STATUS.ongoing, members: ["ongoing"] },
                { key: "done", label: STATUS.done, members: ["done"] }];
  function loadFolds(key) {
    try { return JSON.parse(localStorage.getItem(key) || "{}") || {}; } catch (e) { return {}; }
  }
  function saveFolds(key, v) {
    try { localStorage.setItem(key, JSON.stringify(v)); } catch (e) {}
  }
  var folded = loadFolds("annotool.peopleFolded");
  var foldedTasks = loadFolds("annotool.batchesFolded");   // batches, by task
  function chocoCell(a) {
    return "<b>" + (a.allowance ? a.chocolates : "&ndash;") + "</b>" +
      '<span class="hint"> / <span data-choco="' + a.id + '" title="their allowance ' +
      'for finishing everything they hold; click to change" style="cursor:text;' +
      'text-decoration:underline dotted">' + a.allowance + "</span></span>";
  }
  function introText(intros) {
    var ks = Object.keys(intros || {});
    if (!ks.length) { return ""; }
    return '<div class="hint" style="margin-top:3px" title="the task introduction shown ' +
      'before a task\'s first conversation">intro: ' + ks.map(function (k) {
        var x = intros[k];
        return esc(k.replace(/_/g, " ")) + " " + (x.action === "skipped" ? "skipped"
          : "read" + (x.ms ? " " + dur(x.ms) : ""));
      }).join(" · ") + "</div>";
  }
  function paintPeople() {
    var n = data.annotators.length;
    $("peopleSub").innerHTML = (n ? n + (n === 1 ? " person" : " people") : "") +
      (n && data.chocolates_allowed ? ' · <span title="earned so far / owed if everyone ' +
        'finishes">\uD83C\uDF6B ' + data.chocolates_owed + " / " +
        data.chocolates_allowed + "</span>" : "") +
      (n && localLinks() ? ' <span class="warnchip" title="Personal links are built from ' +
        esc(linkBase()) + ', which only works on this machine. Start the server with ' +
        '--public-url https://your-address to give annotators links they can open.">' +
        "\u26A0 links point to " + esc(linkBase().replace(/^https?:\/\//, "")) +
        "</span>" : "");
    if (!n) {
      $("people").innerHTML = "<tr><td colspan=6 class=empty>" +
        "No annotators yet. Add one to get a personal link you can send them.</td></tr>";
      return;
    }
    var base = linkBase();
    var row = function (a) {
      var url = base + "/a/" + a.token;
      var chips = a.batches.length
        ? a.batches.map(function (b) {
            return '<span class="chip">' + esc(b) +
              '<button title="take back from ' + esc(a.name) + '" data-un="' +
              a.id + '|' + esc(b) + '">×</button></span>';
          }).join("")
        : '<span class="hint">nothing assigned</span>';
      var st = status(a);
      return '<tr class="srow s-' + st + '"><td><b class=nm data-rename=\'' + a.id +
          "' title='click to rename' style='cursor:text'>" + esc(a.name) + "</b><br>" +
          '<span class="stpill s-' + st + '" title="' + ({
            notstarted: "has not opened their link yet",
            started: "opened their link, nothing finished yet",
            ongoing: "finished some of their conversations",
            done: "finished everything they hold" })[st] + '">' + STATUS[st] + "</span></td>" +
        // no URL text: it is the same for everyone up to an unreadable token.
        // Copy it to send, open it to see what they see; hover shows the address.
        '<td><div class="linkcell">' +
          '<button class="btn ghost sm" data-copy="' + esc(url) + '" title="' + esc(url) +
          '">copy link</button>' +
          '<a class="btn ghost sm" href="' + esc(url) + '" target="_blank" rel="noopener" ' +
          'title="open as this annotator">open</a></div></td>' +
        "<td>" + progress(a.done, a.assigned) + "</td>" +
        "<td>" + chocoCell(a) + "</td>" +
        "<td>" + chips + introText(a.intros) + "</td>" +
        '<td class="right"><button class="btn danger sm" data-delp="' + a.id +
          '">Delete</button></td></tr>';
    };
    $("people").innerHTML = GROUPS.map(function (g) {
      var group = [];
      g.members.forEach(function (k) {   // within a group, in status order
        group = group.concat(data.annotators.filter(function (a) { return status(a) === k; }));
      });
      if (!group.length) { return ""; }
      var shut = !!folded[g.key];
      var counts = g.members.length > 1 ? '<span class="grpnames grplab">' +
        g.members.map(function (k) {
          return '<span class="sdot s-' + k + '"></span>' + STATUS[k] + " " +
            group.filter(function (a) { return status(a) === k; }).length;
        }).join(" · ") + "</span>" : "";
      return '<tr class="grp"><td colspan=6><button class="grpbtn" data-fold="' + g.key +
        '" aria-expanded="' + !shut + '" title="click to ' + (shut ? "unfold" : "fold") +
        '"><span class="caret">' + (shut ? "\u25B8" : "\u25BE") + "</span>" +
        '<span class="grplab">' + (g.members.length > 1 ? "" : '<span class="sdot s-' +
        g.key + '"></span>') + g.label + " <b>" + group.length + "</b></span>" + counts +
        (shut ? '<span class="grpnames">' + group.map(function (a) {
          return esc(a.name); }).join(", ") + "</span>" : "") + "</button></td></tr>" +
        (shut ? "" : group.map(row).join(""));
    }).join("");
  }

  /* ---------- batches ---------- */
  function paintBatches() {
    var un = data.batches.filter(function (b) { return !b.taken_by.length; }).length;
    var nt = {};
    data.batches.forEach(function (b) { nt[b.task] = 1; });
    $("batchSub").textContent = data.batches.length + " batches · " +
      Object.keys(nt).length + " tasks · " + (un ? un + " unassigned" : "all assigned");
    if (!data.batches.length) {
      $("batches").innerHTML = "<tr><td colspan=6 class=empty>No batches yet. " +
        "Pick a task above and make one.</td></tr>";
      return;
    }
    var row = function (b) {
      var holders = b.taken_by.length
        ? b.taken_by.map(function (p) {
            var id = (data.annotators.filter(function (a) { return a.name === p; })[0]
                      || {}).id;
            return '<span class="chip' + (b.taken_by.length > 1 ? " dbl" : "") + '">' +
              esc(p) + '<button title="take back" data-un="' + id + "|" + esc(b.name) +
              '">×</button></span>';
          }).join("")
        : '<span class="hint">unassigned</span>';
      var opts = b.eligible.map(function (p) {
        var id = (data.annotators.filter(function (a) { return a.name === p; })[0]
                  || {}).id;
        return "<option value='" + id + "'>" + esc(p) + "</option>";
      }).join("");
      var give = opts
        ? "<span class=giverow><select data-give='" + esc(b.name) + "'>" + opts +
          "</select><button class='btn ghost sm' data-giveb='" + esc(b.name) +
          "'>Assign</button></span>"
        : '<span class="hint">' + (data.annotators.length ? "everyone has these" :
                                   "add an annotator") + "</span>";
      // who is not offered, and why: folded, since it can be long
      var others = b.blocked.filter(function (x) { return b.taken_by.indexOf(x.name) < 0; });
      if (others.length)
        give += '<details class="nooffer" data-keep="no:' + esc(b.name) + '"><summary>\uD83D\uDEAB ' + others.length +
          " not offered</summary><div class=why>They already hold some of these source " +
          "dialogues, and nobody may see two versions of one dialogue. The number is how " +
          "many they hold.</div>" + others.map(function (x) {
            return '<span class="nochip">' + esc(x.name) + " <b>" + x.overlap + "</b></span>";
          }).join("") + "</details>";
      var num = (b.name.match(/-(\d+)$/) || [])[1] || "";
      return "<tr><td><button class='linkbtn bname' data-view='" + esc(b.name) +
          "' title='see what is inside " + esc(b.name) + "'>" +
          (b.emoji ? "<span class=bem>" + b.emoji + "</span>" : "") +
          "<b>" + esc(b.task.replace(/_/g, " ")) + "</b>" +
          (num ? " <span class=bno>#" + num + "</span>" : "") + "</button>" +
          "<br><span class=hint>" + b.size + " conversations</span></td>" +
        "<td>" + compChip(b.composition, b.size) + "</td>" +
        "<td>" + progress(b.judged, b.size) + "</td>" +
        "<td>" + holders + "</td>" +
        "<td>" + give + "</td>" +
        '<td class="right"><button class="btn danger sm" data-delb="' + esc(b.name) +
          '">Delete</button></td></tr>';
    };
    // grouped by task (batches arrive sorted by task, then name); each task folds
    var tasks = [];
    data.batches.forEach(function (b) {
      if (tasks.indexOf(b.task) < 0) { tasks.push(b.task); }
    });
    $("batches").innerHTML = tasks.map(function (t) {
      var group = data.batches.filter(function (b) { return b.task === t; });
      var size = 0, judged = 0, people = {};
      group.forEach(function (b) {
        size += b.size; judged += b.judged;
        b.taken_by.forEach(function (p) { people[p] = 1; });
      });
      var np = Object.keys(people).length, shut = !!foldedTasks[t];
      return '<tr class="grp"><td colspan=2><button class="grpbtn" data-foldt="' + esc(t) +
        '" aria-expanded="' + !shut + '" title="click to ' + (shut ? "unfold" : "fold") +
        '"><span class="caret">' + (shut ? "\u25B8" : "\u25BE") + "</span>" +
        '<span class="grplab">' + (group[0].emoji ? group[0].emoji + " " : "") +
        esc(t.replace(/_/g, " ")) + "</span>" + '<span class="grpnames grplab">' + group.length +
        " batch" + (group.length === 1 ? "" : "es") + " · " + size +
        " conversations</span></button></td>" +
        "<td>" + progress(judged, size) + "</td>" +
        '<td colspan=3><span class="hint">' + np + (np === 1 ? " person" : " people") +
        "</span></td></tr>" + (shut ? "" : group.map(row).join(""));
    }).join("");
  }

  /* The clipboard API only exists on HTTPS or localhost; over plain HTTP fall
     back to the old copy command, and failing that show the link to copy by hand. */
  function copyText(text, el) {
    var label = el.textContent;
    function done() {
      el.textContent = "copied \u2713";
      setTimeout(function () { el.textContent = label; }, 1400);
    }
    function fallback() {
      var t = document.createElement("textarea");
      t.value = text; t.style.position = "fixed"; t.style.opacity = "0";
      document.body.appendChild(t); t.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      t.remove();
      if (ok) { done(); } else { window.prompt("Copy this link:", text); }
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, fallback);
    } else { fallback(); }
  }

  /* ---------- wiring ---------- */
  function wire() {
    document.querySelectorAll("[data-copy]").forEach(function (el) {
      el.addEventListener("click", function () { copyText(el.dataset.copy, el); });
    });
    document.querySelectorAll("[data-un]").forEach(function (el) {
      el.addEventListener("click", function () {
        var p = el.dataset.un.split("|");
        act("/api/admin/unassign", { annotator_id: +p[0], batch: p[1] },
            function (d) { return "Took back <b>" + esc(d.removed.what) + "</b>" +
              (d.removed.answers_deleted ? ", deleting " + d.removed.answers_deleted +
               " answered conversation(s)." : "."); });
      });
    });
    document.querySelectorAll("[data-delp]").forEach(function (el) {
      el.addEventListener("click", function () {
        act("/api/admin/annotator_delete", { annotator_id: +el.dataset.delp },
            function (d) { return "Deleted <b>" + esc(d.removed.what) + "</b>" +
              (d.removed.answers_deleted ? " and " + d.removed.answers_deleted +
               " answered conversation(s)." : "."); });
      });
    });
    document.querySelectorAll("[data-delb]").forEach(function (el) {
      el.addEventListener("click", function () {
        act("/api/admin/batch_delete", { batch: el.dataset.delb },
            function (d) { return "Deleted batch <b>" + esc(d.removed.what) + "</b>" +
              (d.removed.answers_deleted ? " and " + d.removed.answers_deleted +
               " answered conversation(s)." : "."); });
      });
    });
    document.querySelectorAll("[data-view]").forEach(function (el) {
      el.addEventListener("click", function () { Viewer.batch(el.dataset.view); });
    });
    document.querySelectorAll("[data-giveb]").forEach(function (el) {
      el.addEventListener("click", function () {
        var name = el.dataset.giveb;
        var sel = document.querySelector("select[data-give='" + name + "']");
        act("/api/admin/assign", { annotator_id: +sel.value, batch: name },
            function () { return "Assigned <b>" + esc(name) + "</b>."; });
      });
    });
    document.querySelectorAll("[data-fold]").forEach(function (el) {
      el.addEventListener("click", function () {
        folded[el.dataset.fold] = !folded[el.dataset.fold];
        saveFolds("annotool.peopleFolded", folded);
        paint();
      });
    });
    document.querySelectorAll("[data-foldt]").forEach(function (el) {
      el.addEventListener("click", function () {
        foldedTasks[el.dataset.foldt] = !foldedTasks[el.dataset.foldt];
        saveFolds("annotool.batchesFolded", foldedTasks);
        paint();
      });
    });
    document.querySelectorAll("[data-choco]").forEach(function (el) {
      el.addEventListener("click", function () {
        var id = +el.dataset.choco, old = el.textContent;
        var inp = document.createElement("input");
        inp.type = "text"; inp.value = old; inp.style.width = "44px";
        el.replaceWith(inp); inp.focus(); inp.select();
        var finish = function (save) {
          var v = inp.value.trim();
          if (!save || v === old || !/^\d+$/.test(v)) { paint(); return; }
          post("/api/admin/annotator_chocolates", { annotator_id: id, chocolates: +v })
            .then(function (d) { data = d; paint();
              flash("ok", "Allowance set to <b>" + esc(v) + " \uD83C\uDF6B</b>" +
                    (+v ? "." : ": chocolates are hidden for them.")); })
            .catch(function (e) { flash("bad", esc(e.message)); paint(); });
        };
        inp.addEventListener("keydown", function (e) {
          if (e.key === "Enter") { finish(true); }
          if (e.key === "Escape") { finish(false); }
        });
        inp.addEventListener("blur", function () { finish(true); });
      });
    });
    document.querySelectorAll("[data-rename]").forEach(function (el) {
      el.addEventListener("click", function () {
        var id = +el.dataset.rename, old = el.textContent;
        var inp = document.createElement("input");
        inp.type = "text"; inp.value = old; inp.style.width = "150px";
        el.replaceWith(inp); inp.focus(); inp.select();
        var finish = function (save) {
          var v = inp.value.trim();
          if (!save || !v || v === old) { paint(); return; }
          post("/api/admin/annotator_rename", { annotator_id: id, name: v })
            .then(function (d) { data = d; paint();
              flash("ok", "Renamed to <b>" + esc(v) + "</b>."); })
            .catch(function (e) { flash("bad", esc(e.message)); paint(); });
        };
        inp.addEventListener("keydown", function (e) {
          if (e.key === "Enter") { finish(true); }
          if (e.key === "Escape") { finish(false); }
        });
        inp.addEventListener("blur", function () { finish(true); });
      });
    });
  }

  $("addPerson").addEventListener("click", function () {
    var n = $("newName").value.trim();
    if (!n) { $("newName").focus(); return; }
    var c = $("newChoco").value.trim();
    post("/api/admin/annotator_add", { name: n, chocolates: /^\d+$/.test(c) ? +c : 10 })
      .then(function (d) { data = d; $("newName").value = ""; paint();
        flash("ok", "Added <b>" + esc(n) + "</b>. Copy their link and send it."); })
      .catch(function (e) { flash("bad", esc(e.message)); });
  });
  $("newName").addEventListener("keydown", function (e) {
    if (e.key === "Enter") { $("addPerson").click(); }
  });

  /* the New-batch controls: rebuilt on every paint, keeping the selection */
  function paintNewBatch() {
    var t = $("newTask").value, f = $("newFor").value;
    $("newTask").innerHTML = '<option value="">task…</option>' +
      data.tasks.map(function (x) {
        var small = x.sources !== null && x.sources < data.mini;
        var bits = [];
        if (x.sources !== null) { bits.push(x.sources + " dialogues"); }
        if (x.mean_system_turns) { bits.push("~" + x.mean_system_turns.toFixed(1) + " turns"); }
        if (x.batches) { bits.push(x.batches + " batch" + (x.batches > 1 ? "es" : "")); }
        return "<option value='" + esc(x.task) + "'" + (small ? " disabled" : "") + ">" +
          esc(x.task) + (bits.length ? "  (" + bits.join(", ") + ")" : "") +
          (small ? " — too small" : "") + "</option>";
      }).join("");
    $("newFor").innerHTML = '<option value="">for anyone</option>' +
      data.annotators.map(function (a) {
        return "<option value='" + a.id + "'>for " + esc(a.name) + "</option>";
      }).join("");
    $("newTask").value = t; $("newFor").value = f;
    if ($("newFor").value !== f) { $("newFor").value = ""; }
  }

  $("addBatch").addEventListener("click", function () {
    var task = $("newTask").value;
    if (!task) { $("newTask").focus(); flash("warn", "Pick a task first."); return; }
    var body = { task: task, size: +$("newSize").value };
    if ($("newFor").value) { body.annotator_id = +$("newFor").value; }
    $("addBatch").disabled = true;
    post("/api/admin/batch_add", body)
      .then(function (d) {
        data = d;
        if (foldedTasks[task]) {
          foldedTasks[task] = false;
          saveFolds("annotool.batchesFolded", foldedTasks);
        }
        paint();
        var c = d.created_batch;
        flash("ok", "Created <b>" + esc(c.name) + "</b> with " + c.size +
              " conversations" + (c["for"] ? " and gave it to <b>" + esc(c["for"]) +
              "</b>." : ". Give it to someone below.")); })
      .catch(function (e) { flash("bad", esc(e.message)); })
      .then(function () { $("addBatch").disabled = false; });
  });

  /* ---------- report ---------- */
  var SEGLAB = { 0: "conformant", 1: "k=1", 2: "k=2", 3: "k=3", 4: "k=4" };

  function paintExclude() {
    if (!data.annotators.length) { $("excludeBox").innerHTML = ""; return; }
    var ticked = excluded();
    $("excludeBox").innerHTML =
      '<details class="help" data-keep="exclude" style="margin-bottom:14px"><summary>' +
      "Exclude an annotator" +
      "</summary><div class='msg warn' style='margin-top:9px'><div class='exgrid'>" +
      data.annotators.map(function (a) {
        return '<label title="' + STATUS[status(a)] + ", " + a.done + " / " + a.assigned +
               ' finished"><input type="checkbox" class="exq" value="' + esc(a.name) + '"' +
               (ticked.indexOf(a.name) >= 0 ? " checked" : "") + '> <span class="sdot s-' +
               status(a) + '"></span>' + esc(a.name) + "</label>";
      }).join("") + "</div>" +
      "<div style='margin-top:9px'>Decide the <b>rule</b> before looking at the " +
      "results, not after. Section 7b of the report gives the objective basis: each " +
      "annotator's foil false-confirm rate, which needs no comparison with anyone " +
      "else. Whoever you drop, say so and say why wherever the numbers are reported." +
      "</div></div></details>";
  }
  function excluded() {
    return Array.prototype.slice.call(document.querySelectorAll(".exq:checked"))
      .map(function (c) { return c.value; });
  }
  function chartBox(title, sub, legend) {
    var d = document.createElement("div");
    d.className = "chartbox";
    d.innerHTML = "<h3>" + esc(title) + "</h3><p class=sub>" + esc(sub) + "</p>" +
      (legend || "") + "<div class=host></div>";
    return d;
  }
  function legend(pairs) {
    return "<div class=legend>" + pairs.map(function (p) {
      return "<span><i style='background:var(--series-" + p[1] + ")'></i>" +
             esc(p[0]) + "</span>"; }).join("") + "</div>";
  }

  function paintReport(r) {
    var host = $("report"), s = r.summary;
    var pre = (r.excluded && r.excluded.length)
      ? "<div class='msg warn'>Excluding <b>" + esc(r.excluded.join(", ")) +
        "</b> — say so, and say why, wherever these numbers are reported.</div>" : "";
    if (!s.n_done) { host.innerHTML = pre + "<p class=empty>Nothing submitted yet.</p>"; return; }
    var o = s.ordering, q = s.rq1.breaking.quartet, sup = s.rq1.breaking.later_higher;
    var ci = function (x, d) { return x.lo == null ? "" :
      " · CI [" + x.lo.toFixed(d) + ", " + x.hi.toFixed(d) + "]"; };
    var sg = function (v) { return (v >= 0 ? "+" : "") + v.toFixed(2); };
    host.innerHTML = pre + "<div class=tiles>" +
      tile("Completed", s.n_done + " / " + s.n_items, "conversations") +
      tile("Double-judged", s.n_double, "feeds agreement") +
      (q.n ? tile("Fork order τ-b", sg(q.mean),
                  "quartets, WIC/SSPC/GR · n=" + q.n + ci(q, 2)) : "") +
      (o.n ? tile("Within annotator τ-b", sg(o.mean),
                  "WIC/SSPC/GR · n=" + o.n + " person×task" + ci(o, 2)) : "") +
      (sup ? tile("Later fork rated higher", sup.pct.toFixed(0) + "%",
                  "same annotator, WIC/SSPC/GR · chance 50%" + ci(sup, 0)) : "") + "</div>";

    var c1 = chartBox("RQ1 · Conformance by fork position",
      "Mean of the step-1 slider with 95% CI. Bars: forks of the types that break the " +
      "workflow (WIC, SSPC, GR); diamonds: all four types. CSC keeps every component " +
      "of the workflow and adds a competing objective, so it changes little with k. " +
      "Conformant conversations are the anchor: they should sit highest, and the forks " +
      "should rise with k.",
      legend([["WIC, SSPC, GR forks", 1], ["conformant (control)", 2]]).replace(
        "</div>", "<span><i class=dia></i>all four types</span></div>"));
    host.appendChild(c1);
    Charts.columns(c1.querySelector(".host"), s.slider_by_segment.map(function (x) {
      return { label: SEGLAB[x.segment], value: x.mean, lo: x.lo, hi: x.hi,
               n: x.n, series: x.segment === 0 ? 2 : 1,
               mark: x.segment === 0 ? null : x.mean_all };
    }), { max: 100, ticks: [0, 50, 100], markLabel: "all four types",
          fmtVal: function (v) { return v.toFixed(0); } });

    var c2 = chartBox("Was the instructed behaviour confirmed?",
      "A foil shows an instruction from a different hallucination type, so the " +
      "correct answer for a foil is 'no'. A small gap between these bars would mean " +
      "the annotators are rubber-stamping and would invalidate the validity rates.");
    host.appendChild(c2);
    Charts.columns(c2.querySelector(".host"), s.confirmation.map(function (x) {
      return { label: x.group.replace(" (", " ("), value: x.rate,
               lo: x.lo, hi: x.hi, n: x.n, series: 1 };
    }), { max: 100, ticks: [0, 50, 100],
          fmtVal: function (v) { return v.toFixed(0) + "%"; },
          fmtTick: function (t) { return t + "%"; } });

    if (s.confirmation_by_type.length) {
      var c3 = chartBox("Confirmation rate per hallucination type",
        "True instruction only. A low bar means that corruption type is not " +
        "reliably realised by the generator.");
      host.appendChild(c3);
      Charts.columns(c3.querySelector(".host"), s.confirmation_by_type.map(function (x) {
        return { label: x.group, value: x.rate, n: x.n, series: 1 };
      }), { max: 100, ticks: [0, 50, 100],
            fmtVal: function (v) { return v.toFixed(0) + "%"; },
            fmtTick: function (t) { return t + "%"; } });
    }

    var c4 = chartBox("How far off was the onset?",
      "|annotator's first turn − the injected fork turn|, in assistant turns. " +
      "This is what validates fork_assistant_turn as a label.");
    host.appendChild(c4);
    Charts.columns(c4.querySelector(".host"), s.onset_error.map(function (x) {
      return { label: String(x.err), value: x.n, series: 1 };
    }), { fmtVal: function (v) { return v; } });

    if (s.metric_correlation) {
      var c5 = chartBox("Do the automatic metrics track the annotators?",
        "Spearman ρ between human deviation (100 − mean slider) and each metric's " +
        "distance from the conformant original, with 95% CI, over the annotated " +
        "forks that have Task 2 results. Higher = agrees more with people.",
        legend([["metric", 1], ["length-only baseline", 2]]));
      host.appendChild(c5);
      if (s.metric_correlation.length) {
        Charts.columns(c5.querySelector(".host"), s.metric_correlation.map(function (x) {
          return { label: x.method, value: Math.max(0, x.rho), lo: x.lo, hi: x.hi,
                   n: x.n, series: /count-diff/.test(x.method) ? 2 : 1,
                   text: (x.rho >= 0 ? "+" : "") + x.rho.toFixed(2) };
        }), { max: 1, ticks: [0, 0.5, 1],
              fmtVal: function (v, r) { return r && r.text ? r.text : v.toFixed(2); } });
      } else {
        c5.querySelector(".host").innerHTML =
          "<p class=empty>No Task 2 results for the tasks annotated so far.</p>";
      }
    }

    var md = document.createElement("details");
    md.className = "help";
    md.innerHTML = "<summary>Full report</summary><div class=md id=mdbody></div>";
    host.appendChild(md);
    md.querySelector("#mdbody").innerHTML = mdToHtml(r.markdown);
    if (s.notes && s.notes.length) {
      var nb = document.createElement("details");
      nb.className = "help";
      nb.dataset.keep = "notes";
      nb.innerHTML = "<summary>Annotators' notes (" + s.notes.length + ")</summary>" +
        "<p class=hint>Click a note to open its conversation, with the ground truth and " +
        "everyone's answers; ← → step through the notes.</p>" +
        '<table class="md notes"><thead><tr><th>conversation</th><th>who</th>' +
        "<th>note</th></tr></thead><tbody>" + s.notes.map(function (n, i) {
          return '<tr data-note="' + i + '" tabindex="0"><td><b>' + esc(n.task) + "</b><br>" +
            esc(n.type) + (n.segment ? " · k=" + n.segment : "") +
            (n.is_foil ? ' <span class="foil">foil</span>' : "") + "</td><td>" +
            esc(n.annotator) + "</td><td>" + esc(n.note) + "</td></tr>";
        }).join("") + "</tbody></table>";
      host.appendChild(nb);
      var ids = s.notes.map(function (n) { return n.item_id; });
      nb.querySelectorAll("[data-note]").forEach(function (tr) {
        var go = function () {
          Viewer.list(ids, +tr.dataset.note, "Annotators' notes");
        };
        tr.addEventListener("click", go);
        tr.addEventListener("keydown", function (e) { if (e.key === "Enter") { go(); } });
      });
    }
    var p = document.createElement("p");
    p.innerHTML = '<button class="btn ghost sm" id="copyMd">copy markdown</button>';
    host.appendChild(p);
    $("copyMd").addEventListener("click", function () {
      navigator.clipboard.writeText(r.markdown).then(function () {
        $("copyMd").textContent = "copied";
        setTimeout(function () { $("copyMd").textContent = "copy markdown"; }, 1400);
      });
    });
  }

  function mdToHtml(src) {
    var out = [], rows = null;
    src.split("\n").forEach(function (line) {
      var t = line.trim();
      if (/^\|/.test(t)) {
        var cells = t.replace(/^\||\|$/g, "").split("|").map(function (c) { return c.trim(); });
        if (/^[-: ]+$/.test(cells.join(""))) { return; }
        rows = rows || []; rows.push(cells); return;
      }
      if (rows) { out.push(table(rows)); rows = null; }
      if (/^### /.test(t)) { out.push("<h3>" + inline(t.slice(4)) + "</h3>"); }
      else if (/^## /.test(t)) { out.push("<h2>" + inline(t.slice(3)) + "</h2>"); }
      else if (/^# /.test(t)) { out.push("<h1>" + inline(t.slice(2)) + "</h1>"); }
      else if (/^- /.test(t)) { out.push("<div>&bull; " + inline(t.slice(2)) + "</div>"); }
      else if (t) { out.push("<p>" + inline(t) + "</p>"); }
    });
    if (rows) { out.push(table(rows)); }
    return out.join("");
  }
  function table(rows) {
    var head = rows.shift();
    return "<table><thead><tr>" + head.map(function (c) {
      return "<th>" + inline(c) + "</th>"; }).join("") + "</tr></thead><tbody>" +
      rows.map(function (r) { return "<tr>" + r.map(function (c) {
        return "<td>" + inline(c) + "</td>"; }).join("") + "</tr>"; }).join("") +
      "</tbody></table>";
  }
  function inline(s) {
    return esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
                 .replace(/`(.+?)`/g, "<code>$1</code>")
                 .replace(/\*(.+?)\*/g, "<i>$1</i>");
  }

  $("reload").addEventListener("click", function () {
    $("reload").disabled = true;
    refresh().then(function () { $("reload").disabled = false; })
             .catch(function () { $("reload").disabled = false; });
  });

  $("run").addEventListener("click", function () {
    $("run").disabled = true;
    $("report").innerHTML = "<p class=empty>Computing…</p>";
    post("/api/admin/report", { metrics: $("metrics").checked, exclude: excluded() })
      .then(paintReport)
      .catch(function (e) {
        $("report").innerHTML = "<div class='msg bad'>" + esc(e.message) + "</div>"; })
      .then(function () { $("run").disabled = false; });
  });

  /* The page repaints itself every few seconds; without this, any folded box
     someone had just opened would snap shut on the next refresh. */
  function paint() {
    var open = {};
    document.querySelectorAll("details[data-keep]").forEach(function (d) {
      if (d.open) { open[d.dataset.keep] = true; }
    });
    paintTiles(); paintPeople(); paintNewBatch(); paintBatches(); paintExclude(); wire();
    document.querySelectorAll("details[data-keep]").forEach(function (d) {
      if (open[d.dataset.keep]) { d.open = true; }
    });
    stamp();
  }

  /* The page used to fetch once on load and never again, so anything that
     happened afterwards -- another admin adding a batch, or annotators simply
     working -- was invisible until a manual reload. Poll instead. */
  var lastAt = 0, polling = null;
  function stamp() {
    lastAt = Date.now();
    var el = $("updated");
    if (el) { el.textContent = "updated just now"; }
  }
  function tickStamp() {
    var el = $("updated");
    if (!el || !lastAt) { return; }
    var s = Math.round((Date.now() - lastAt) / 1000);
    el.textContent = s < 5 ? "updated just now"
      : s < 90 ? "updated " + s + "s ago" : "updated " + Math.round(s / 60) + " min ago";
  }
  /* ---------- batch viewer ----------
     Grid of the batch's blocks (type x fork position, plus conformant); click a
     conversation to read it exactly as annotators see it, with the ground truth
     and everyone's answers alongside. */
  var Viewer = (function () {
    var cur = null, order = [], pos = -1, listTitle = "";
    var TYPES = ["WIC", "SSPC", "GR", "CSC"];
    var TYPE_NAME = { WIC: "workflow identity", SSPC: "step selection",
                      GR: "goal replacement", CSC: "conflicting constraints" };

    function show(on) {
      $("viewer").hidden = !on;
      document.body.style.overflow = on ? "hidden" : "";
    }
    function chip(it) {
      return '<button class="vchip' + (it.judged ? " judged" : "") + '" data-item="' +
        esc(it.item_id) + '" title="' + esc(it.ref) + (it.is_foil ? " · foil" : "") +
        " · judged by " + it.judged + '">#' + esc(it.dialog_id) +
        (it.is_foil ? ' <span class="foil">foil</span>' : "") +
        (it.judged ? ' <span class="nj">' + it.judged + "✓</span>" : "") + "</button>";
    }
    function batch(name) {
      return post("/api/admin/batch_view", { batch: name }).then(function (b) {
        cur = b; pos = -1;
        order = [];
        var cell = {};
        b.items.forEach(function (it) {
          var k = it.kind === "conformant" ? "conf" : it.type + "|" + it.segment;
          (cell[k] = cell[k] || []).push(it);
        });
        TYPES.forEach(function (t) {
          [1, 2, 3, 4].forEach(function (k) {
            (cell[t + "|" + k] || []).forEach(function (it) { order.push(it); });
          });
        });
        (cell.conf || []).forEach(function (it) { order.push(it); });
        $("vTitle").textContent = b.name;
        $("vSub").textContent = b.items.length + " conversations · " +
          (b.held_by.length ? "held by " + b.held_by.join(", ") : "not assigned");
        $("vBack").hidden = $("vPrev").hidden = $("vNext").hidden = true;
        $("vBody").innerHTML = '<div class="wrap">' +
          '<p class="hint" style="margin-top:0">Every conversation comes from a different ' +
          "source dialogue (#id). Rows are the hallucination type, columns where the fork " +
          "happens (k=1 earliest, most of the conversation affected). <b>foil</b> = the " +
          "cue shown in step 2 names an instruction that was never injected.</p>" +
          '<table class="vgrid"><thead><tr><th></th>' +
          [1, 2, 3, 4].map(function (k) { return "<th>k=" + k + "</th>"; }).join("") +
          "</tr></thead><tbody>" +
          TYPES.map(function (t) {
            return "<tr><th>" + t + '<span class="hint">' + TYPE_NAME[t] + "</span></th>" +
              [1, 2, 3, 4].map(function (k) {
                return "<td>" + (cell[t + "|" + k] || []).map(chip).join("") + "</td>";
              }).join("") + "</tr>";
          }).join("") +
          '<tr><th>conformant<span class="hint">never altered</span></th>' +
          '<td colspan="4">' + (cell.conf || []).map(chip).join("") + "</td></tr>" +
          "</tbody></table></div>";
        $("vBody").querySelectorAll("[data-item]").forEach(function (el) {
          el.addEventListener("click", function () { item(el.dataset.item); });
        });
        show(true);
      }).catch(function (e) { flash("bad", esc(e.message)); });
    }
    function item(id) {
      pos = order.map(function (x) { return x.item_id; }).indexOf(id);
      return post("/api/admin/item_view", { item_id: id }).then(function (d) {
        var m = d.meta, r = d.reveal;
        var what = m.kind === "conformant" ? "conformant (never altered)"
          : m.type + " · k=" + m.segment + " · fork at assistant turn " + m.fork_assistant_turn + " · " + m.variant_id;
        $("vTitle").textContent = (cur ? cur.name : listTitle + " (" + (pos + 1) + "/" +
                                   order.length + ")") + "  ·  #" + m.dialog_id;
        $("vSub").textContent = (cur ? "" : m.ref + " · ") + what;
        $("vBack").hidden = !cur;
        $("vPrev").hidden = $("vNext").hidden = false;
        $("vPrev").disabled = pos <= 0;
        $("vNext").disabled = pos < 0 || pos >= order.length - 1;

        var cue = '<div class="cue"><b>Step 2 cue shown to annotators' +
          (m.is_foil ? ' <span class="foil">foil — ' + esc(m.cue_type) + " " +
                       esc(m.cue_variant) + ", never injected</span>" : " (the true one)") +
          ":</b> " + Render.roles(esc(d.cue.lead)) + '<span class="ins">' + esc(d.cue.instruction) +
          "</span></div>";
        var truth = r.true_cue
          ? '<div class="cue"><b>What was actually injected (from assistant turn ' + r.fork_assistant_turn +
            "):</b><span class=ins>" + esc(r.true_cue.instruction) + "</span>" +
            (r.injected_text ? "<details><summary>full injected prompt</summary><pre class=inj>" +
             esc(r.injected_text) + "</pre></details>" : "") + "</div>"
          : "";
        var ans = d.answers.length
          ? '<table class="md vans"><tr><th>annotator</th><th>slider</th><th>appears</th>' +
            "<th>onset</th><th>before fork</th><th>note</th></tr>" +
            d.answers.map(function (a) {
              return "<tr><td>" + esc(a.annotator) + (a.complete ? "" : " <i>(partial)</i>") +
                "</td><td>" + (a.slider == null ? "" : a.slider) + "</td><td>" +
                esc(a.appears || "") + "</td><td>" + (a.onset ? "assistant turn " + a.onset : "") +
                "</td><td>" + esc(a.prefix_ok || "") + "</td><td>" + esc(a.note || "") +
                "</td></tr>";
            }).join("") + "</table>"
          : '<p class="hint">Nobody has answered this one yet.</p>';
        var marks = {};
        d.answers.forEach(function (a) {
          if (a.onset) { marks[a.onset] = (marks[a.onset] ? marks[a.onset] + ", " : "") +
                                         a.annotator + "'s onset"; }
        });

        $("vBody").innerHTML = '<main style="padding-bottom:40px">' +
          '<div class="card" style="margin-bottom:16px">' + cue + truth + ans + "</div>" +
          '<div class="cols"><aside class="card side">' +
          "<h2>The task</h2><p class='task-name'>" + esc(d.task_title) + "</p>" +
          "<p style='margin:0'>" + esc(d.task_description) + "</p>" +
          "<div class='goal'><b>The user's brief (what the simulated user was told)</b>" + esc(d.user_goal || "(not recorded)") +
          "</div><h2>Steps the <span class=role-a>assistant</span> was given</h2>" +
          Render.flow(d.flow) + "</aside>" +
          "<section class='card'><h2>The conversation</h2>" +
          Render.transcript(d.turns, { bandFrom: m.kind === "fork" ? m.fork_assistant_turn : null,
                                       marks: marks }) +
          "</section></div></main>";
        Render.wireFlow($("vBody"));
        $("viewer").scrollTop = 0;
      }).catch(function (e) { flash("bad", esc(e.message)); });
    }
    $("vClose").addEventListener("click", function () { show(false); });
    $("vBack").addEventListener("click", function () { batch(cur.name); });
    $("vPrev").addEventListener("click", function () { if (pos > 0) { item(order[pos - 1].item_id); } });
    $("vNext").addEventListener("click", function () {
      if (pos >= 0 && pos < order.length - 1) { item(order[pos + 1].item_id); } });
    document.addEventListener("keydown", function (e) {
      if ($("viewer").hidden) { return; }
      if (e.key === "Escape") { show(false); }
      else if (e.key === "ArrowRight" && !$("vNext").hidden) { $("vNext").click(); }
      else if (e.key === "ArrowLeft" && !$("vPrev").hidden) { $("vPrev").click(); }
    });
    /* open conversations outside any batch (e.g. the ones annotators left a note
       on), stepping through `ids` with the arrows */
    function list(ids, start, title) {
      cur = null; listTitle = title || "";
      order = ids.map(function (i) { return { item_id: i }; });
      return item(ids[start || 0]).then(function () { show(true); });
    }
    return { batch: batch, list: list };
  })();

  function busy() {
    // never yank the page out from under an edit in progress
    return !$("viewer").hidden || !$("pwModal").hidden ||
           document.querySelector("#people input[type=text]") !== null ||
           document.activeElement === $("newName") ||
           ["newTask", "newSize", "newFor", "newChoco"].indexOf(
             (document.activeElement || {}).id) >= 0;
  }
  function refresh(quiet) {
    return post("/api/admin/stats").then(function (d) {
      var keep = excluded();
      data = d;
      paint();
      document.querySelectorAll(".exq").forEach(function (c) {
        c.checked = keep.indexOf(c.value) >= 0;
      });
    }).catch(function (e) { if (!quiet) { throw e; } });
  }
  /* Refresh every 5 minutes, and whenever the tab comes back into view (if the
     data is more than 30 s old); the Refresh button is there for anything sooner. */
  function startPolling() {
    if (polling) { return; }
    setInterval(tickStamp, 15000);
    polling = setInterval(function () {
      if (!document.hidden && !busy()) { refresh(true); }
    }, 300000);
    var back = function () {
      if (!document.hidden && !busy() && Date.now() - lastAt > 30000) { refresh(true); }
    };
    document.addEventListener("visibilitychange", back);
    window.addEventListener("focus", back);
  }

  /* downloads: the session cookie goes along, so a plain link is enough */
  function exportHref(what) {
    return BASE + "/api/admin/export?what=" + what + ($("metrics").checked ? "&metrics=1" : "");
  }
  // the links need an href from the start (a link without one gets no hand
  // cursor and is not focusable); keep it in step with the metrics box
  function setExportHrefs() {
    $("expCsv").href = exportHref("csv");
    $("expZip").href = exportHref("zip");
  }
  setExportHrefs();
  $("metrics").addEventListener("change", setExportHrefs);
  ["Csv", "Zip"].forEach(function (k) {
    $("exp" + k).addEventListener("click", function () {
      flash("ok", "Preparing the download" + ($("metrics").checked
        ? " with the metric correlation (takes a few seconds)" : "") + "\u2026");
    });
  });

  refresh().then(startPolling).catch(function (e) {
    if (e.status === 401 || e.status === 403) { return; }   // login / password form
    flash("bad", esc(e.message), true);
  });
})();
