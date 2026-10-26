// SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
// SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
// SPDX-License-Identifier: MIT

/* Admin login: posts the password; the server answers with a session cookie
   (HttpOnly -- this script never sees it) and the panel is served on reload. */
(function () {
  "use strict";
  // path prefix when served behind a reverse proxy ("" when accessed directly)
  var BASE = window.BASE || "";
  var $ = function (id) { return document.getElementById(id); };
  $("loginForm").addEventListener("submit", function (e) {
    e.preventDefault();
    $("go").disabled = true;
    $("err").textContent = "";
    fetch(BASE + "/api/admin/login", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: $("pw").value })
    }).then(function (r) {
      return r.json().then(function (j) {
        if (!r.ok) { throw new Error(j.error || ("HTTP " + r.status)); }
        location.replace(BASE + "/admin");
      });
    }).catch(function (err) {
      $("err").textContent = err.message;
      $("pw").select();
      var card = $("loginForm");
      card.classList.remove("shake"); void card.offsetWidth; card.classList.add("shake");
    }).then(function () { $("go").disabled = false; });
  });
})();
