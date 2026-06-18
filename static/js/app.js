(function () {
  "use strict";

  function setProgress(progress, status) {
    var bar = document.getElementById("job-progress");
    if (!bar) return;
    bar.style.width = progress + "%";
    bar.classList.toggle("progress-bar-animated", status === "running");
    bar.classList.toggle("bg-danger", status === "error");
    bar.classList.toggle("bg-success", status === "done");
  }

  function setStatus(status, message) {
    var label = document.getElementById("job-status-label");
    if (label) {
      label.textContent = status;
      label.className = "badge " + (
        status === "done"  ? "bg-success" :
        status === "error" ? "bg-danger"  : "bg-warning"
      );
    }
    var msg = document.getElementById("job-message");
    if (msg && message) msg.textContent = message;
  }

  window.startJobPolling = function (opts) {
    if (!opts || !opts.jobId) return;
    var url = "/api/job/" + opts.jobId;
    var interval = 1000;
    var stopped = false;

    function tick() {
      if (stopped) return;
      fetch(url)
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (!data) return;
          var progress = (data.status === "running")
            ? Math.max(10, data.progress || 50)
            : 100;
          setProgress(progress, data.status);
          setStatus(data.status, data.message || "");

          if (data.status === "done") {
            stopped = true;
            if (opts.summaryUrl && opts.renderSummary) {
              fetch(opts.summaryUrl)
                .then(function (r) { return r.json(); })
                .then(function (s) {
                  var sb = document.getElementById("summary-block");
                  if (sb) {
                    sb.style.display = "";
                    opts.renderSummary(s, sb);
                  }
                });
            }
          } else if (data.status === "error") {
            stopped = true;
          } else {
            setTimeout(tick, interval);
          }
        })
        .catch(function () { setTimeout(tick, interval * 2); });
    }
    tick();
  };

  window.initDropzone = function () {
    var dz = document.getElementById("dropzone");
    var input = document.getElementById("image-input");
    if (!dz || !input) return;

    var prompt = document.getElementById("dropzone-prompt");
    var fname  = document.getElementById("dropzone-filename");
    var fnameText = document.getElementById("dropzone-filename-text");

    function showFile(file) {
      if (!file) return;
      if (prompt) prompt.style.display = "none";
      if (fname) fname.style.display = "block";
      if (fnameText) fnameText.textContent = file.name +
          " (" + (file.size / 1024).toFixed(1) + " KB)";
    }

    dz.addEventListener("click", function () { input.click(); });
    input.addEventListener("change", function () {
      if (input.files && input.files[0]) showFile(input.files[0]);
    });

    ["dragenter", "dragover"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) {
        e.preventDefault(); e.stopPropagation();
        dz.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) {
        e.preventDefault(); e.stopPropagation();
        dz.classList.remove("dragover");
      });
    });
    dz.addEventListener("drop", function (e) {
      var files = e.dataTransfer && e.dataTransfer.files;
      if (files && files[0]) {
        input.files = files;
        showFile(files[0]);
      }
    });

    var resetBtn = document.getElementById("reset-btn");
    if (resetBtn) {
      resetBtn.addEventListener("click", function () {
        if (prompt) prompt.style.display = "block";
        if (fname) fname.style.display = "none";
      });
    }
  };

  window.initPlotModal = function () {
    var modal = document.getElementById("plotModal");
    if (!modal) return;
    modal.addEventListener("show.bs.modal", function (event) {
      var trigger = event.relatedTarget;
      if (!trigger) return;
      var src = trigger.getAttribute("data-plot-src");
      var title = trigger.getAttribute("data-plot-title") || "";
      var desc = trigger.getAttribute("data-plot-desc") || "";
      var img = document.getElementById("plotModalImg");
      var t = document.getElementById("plotModalTitle");
      var d = document.getElementById("plotModalDesc");
      if (img) img.src = src || "";
      if (t) t.textContent = title;
      if (d) d.textContent = desc;
    });
  };
})();
