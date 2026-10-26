// SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute
// SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
// SPDX-License-Identifier: MIT

/* Minimal inline-SVG charts. No library.
   Specs follow the house data-viz rules: bars <=24px with a 4px rounded
   data-end square at the baseline, a 2px surface gap between neighbours,
   hairline recessive gridlines, values as text tokens (never the series
   colour), a legend whenever there are two series, and a hover tooltip. */
(function (global) {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  function el(n, a) {
    var e = document.createElementNS(NS, n);
    for (var k in a) { if (a[k] != null) { e.setAttribute(k, a[k]); } }
    return e;
  }
  /* bar with a rounded top and square feet, drawn as a path */
  function barPath(x, y, w, h, r) {
    r = Math.min(r, w / 2, h);
    if (h <= 0.5) { return "M" + x + "," + (y + h) + "h" + w; }
    return "M" + x + "," + (y + h) +
           "V" + (y + r) + "a" + r + "," + r + " 0 0 1 " + r + ",-" + r +
           "h" + (w - 2 * r) + "a" + r + "," + r + " 0 0 1 " + r + "," + r +
           "V" + (y + h) + "Z";
  }

  /* rows: [{label, value, lo, hi, series}]  series 1 or 2 */
  function columns(host, rows, opts) {
    opts = opts || {};
    host.innerHTML = "";
    var W = opts.width || 520, H = opts.height || 190;
    var padL = 34, padR = 10, padT = 18, padB = 34;
    var max = opts.max != null ? opts.max
      : Math.max.apply(null, rows.map(function (r) {
          return Math.max(r.value || 0, r.hi || 0); }).concat([1]));
    var min = opts.min != null ? opts.min : 0;
    var plotW = W - padL - padR, plotH = H - padT - padB;
    var svg = el("svg", { class: "viz", viewBox: "0 0 " + W + " " + H,
                          preserveAspectRatio: "xMidYMid meet",
                          role: "img", "aria-label": opts.title || "chart" });
    var y = function (v) { return padT + plotH - (v - min) / (max - min) * plotH; };

    (opts.ticks || [min, (min + max) / 2, max]).forEach(function (t) {
      svg.appendChild(el("line", { class: "grid", x1: padL, x2: W - padR,
                                   y1: y(t), y2: y(t) }));
      var lab = el("text", { class: "axis", x: padL - 6, y: y(t) + 4,
                             "text-anchor": "end" });
      lab.textContent = opts.fmtTick ? opts.fmtTick(t) : Math.round(t);
      svg.appendChild(lab);
    });

    var band = plotW / rows.length;
    var bw = Math.min(24, band - 2);           /* cap thickness; 2px surface gap */
    rows.forEach(function (r, i) {
      var cx = padL + band * i + band / 2;
      var x = cx - bw / 2;
      var v = r.value == null ? null : r.value;
      if (v != null) {
        var top = y(Math.max(v, min)), base = y(min);
        var p = el("path", {
          class: "bar", d: barPath(x, Math.min(top, base), bw, Math.abs(base - top), 4),
          fill: "var(--series-" + (r.series || 1) + ")" });
        var t = el("title");
        t.textContent = r.label + ": " + (opts.fmtVal ? opts.fmtVal(v, r) : v) +
                        (r.n != null ? "  (n=" + r.n + ")" : "");
        p.appendChild(t);
        svg.appendChild(p);
        if (r.lo != null && r.hi != null && r.hi > r.lo) {
          svg.appendChild(el("line", { class: "ci", x1: cx, x2: cx,
                                       y1: y(r.lo), y2: y(r.hi) }));
          svg.appendChild(el("line", { class: "ci", x1: cx - 4, x2: cx + 4,
                                       y1: y(r.hi), y2: y(r.hi) }));
          svg.appendChild(el("line", { class: "ci", x1: cx - 4, x2: cx + 4,
                                       y1: y(r.lo), y2: y(r.lo) }));
        }
        /* the value sits above the CI and above the marker, so neither hides it */
        var lift = Math.min(top, base, r.hi != null ? y(r.hi) : top,
                            r.mark != null ? y(r.mark) - 5 : top);
        var vt = el("text", { class: "val", x: cx,
                              y: lift - (lift < Math.min(top, base) ? 7 : 5),
                              "text-anchor": "middle" });
        vt.textContent = opts.fmtVal ? opts.fmtVal(v, r) : v;
        svg.appendChild(vt);
      }
      /* optional second measure on the same scale: a diamond with a surface ring */
      if (r.mark != null) {
        var my = y(r.mark), s = 5;
        var dm = el("path", { class: "mark",
          d: "M" + cx + "," + (my - s) + "l" + s + "," + s + "l-" + s + "," + s +
             "l-" + s + ",-" + s + "Z" });
        var mt = el("title");
        mt.textContent = r.label + ", " + (opts.markLabel || "marker") + ": " +
          (opts.fmtVal ? opts.fmtVal(r.mark, r) : r.mark);
        dm.appendChild(mt);
        svg.appendChild(dm);
      }
      var lt = el("text", { class: "axis", x: cx, y: H - padB + 15,
                            "text-anchor": "middle" });
      lt.textContent = r.label;
      svg.appendChild(lt);
      if (r.n != null) {
        var nt = el("text", { class: "axis", x: cx, y: H - padB + 27,
                              "text-anchor": "middle" });
        nt.textContent = "n=" + r.n;
        svg.appendChild(nt);
      }
    });
    svg.appendChild(el("line", { class: "grid", x1: padL, x2: W - padR,
                                 y1: y(min), y2: y(min) }));
    host.appendChild(svg);
  }

  global.Charts = { columns: columns };
})(window);
