(() => {
  "use strict";

  const csrf = document.querySelector('meta[name="csrf-token"]').content;
  const elements = {
    list: document.getElementById("review-list"),
    progress: document.getElementById("progress"),
    progressLabel: document.getElementById("progress-label"),
    position: document.getElementById("item-position"),
    time: document.getElementById("item-time"),
    status: document.getElementById("item-status"),
    loading: document.getElementById("loading-state"),
    editor: document.getElementById("review-editor"),
    player: document.getElementById("clip-player"),
    text: document.getElementById("transcript-text"),
    language: document.getElementById("transcript-language"),
    notes: document.getElementById("review-notes"),
    approve: document.getElementById("approve-button"),
    correct: document.getElementById("correct-button"),
    unclear: document.getElementById("unclear-button"),
    noSpeech: document.getElementById("no-speech-button"),
    previous: document.getElementById("previous-button"),
    next: document.getElementById("next-button"),
    completion: document.getElementById("completion"),
    completionCopy: document.getElementById("completion-copy"),
    export: document.getElementById("export-button"),
    exportStatus: document.getElementById("export-status"),
    toast: document.getElementById("toast"),
  };

  let state = null;
  let currentIndex = 0;

  const seconds = (milliseconds) => (milliseconds / 1000).toFixed(1);

  function showToast(message) {
    elements.toast.textContent = message;
    elements.toast.hidden = false;
    window.setTimeout(() => { elements.toast.hidden = true; }, 2600);
  }

  function currentItem() { return state.items[currentIndex]; }
  function currentDecision() { return state.decisions[currentItem().id]; }

  function renderProgress() {
    const { progress } = state;
    elements.progress.value = progress.reviewed;
    elements.progress.max = progress.total;
    elements.progressLabel.textContent = `${progress.reviewed} of ${progress.total} reviewed`;
    const complete = progress.pending === 0;
    elements.completion.hidden = !complete;
    if (complete) {
      elements.completionCopy.textContent = progress.unclear
        ? `All spans are reviewed. ${progress.unclear} span${progress.unclear === 1 ? " remains" : "s remain"} explicitly uncertain; local export is allowed, but approval and GIGA admission remain blocked.`
        : "Every reviewed span is resolved or removed as no speech. The transcript can now be exported locally.";
      elements.export.disabled = !progress.export_ready;
      if (state.export) {
        elements.export.textContent = "Export again";
        elements.exportStatus.textContent = `Last export succeeded · quality ${state.export.quality_status} · no GIGA event emitted.`;
      }
    }
  }

  function renderList() {
    const buttons = elements.list.querySelectorAll("button[data-review-index]");
    buttons.forEach((button, index) => {
      const item = state.items[index];
      const decision = state.decisions[item.id];
      button.setAttribute("aria-current", index === currentIndex ? "true" : "false");
      button.dataset.status = decision.status;
      button.lastElementChild.textContent = decision.status;
    });
  }

  function renderCurrent({ focus = false } = {}) {
    const item = currentItem();
    const decision = currentDecision();
    elements.position.textContent = `Span ${currentIndex + 1} of ${state.items.length}`;
    elements.time.textContent = `${seconds(item.start_ms)}–${seconds(item.end_ms)}s`;
    elements.status.textContent = decision.status;
    elements.status.dataset.status = decision.status;
    elements.player.src = `/clips/${encodeURIComponent(item.id)}.wav`;
    elements.text.value = decision.text;
    elements.language.value = decision.language;
    elements.notes.value = decision.notes;
    elements.previous.disabled = currentIndex === 0;
    elements.next.disabled = currentIndex === state.items.length - 1;
    renderList();
    if (focus) elements.player.focus();
  }

  function select(index, options = {}) {
    if (!state || index < 0 || index >= state.items.length) return;
    elements.player.pause();
    currentIndex = index;
    renderCurrent(options);
    elements.list.querySelector(`[data-review-index="${index}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  function nextPending() {
    for (let offset = 1; offset <= state.items.length; offset += 1) {
      const index = (currentIndex + offset) % state.items.length;
      if (state.decisions[state.items[index].id].status === "pending") return index;
    }
    return Math.min(currentIndex + 1, state.items.length - 1);
  }

  async function saveDecision(status) {
    const item = currentItem();
    let response;
    try {
      response = await fetch(`/api/decision/${encodeURIComponent(item.id)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
        body: JSON.stringify({
          status,
          text: elements.text.value,
          language: elements.language.value,
          notes: elements.notes.value,
        }),
      });
    } catch {
      showToast("Could not reach the local reviewer. Reload the page and try again.");
      return;
    }
    if (!response.ok) {
      showToast("Decision was not saved. Check the transcript and language.");
      return;
    }
    const payload = await response.json();
    state.decisions[item.id] = {
      status,
      text: status === "approved" ? item.proposed_text : status === "no_speech" ? "" : elements.text.value.trim(),
      language: status === "approved" ? item.proposed_language : status === "no_speech" ? "unknown" : elements.language.value,
      notes: elements.notes.value.trim(),
    };
    state.progress = payload.progress;
    renderProgress();
    const destination = nextPending();
    renderList();
    showToast(status === "approved" ? "Approved unchanged" : status === "corrected" ? "Correction saved" : status === "no_speech" ? "Marked as no speech" : "Marked unclear");
    if (destination !== currentIndex) select(destination, { focus: true });
    else renderCurrent();
  }

  async function exportReview() {
    elements.export.disabled = true;
    elements.export.textContent = "Exporting…";
    elements.exportStatus.textContent = "Running the quality check and writing the reviewed transcript…";
    let response;
    try {
      response = await fetch("/api/export", {
        method: "POST",
        headers: { "X-CSRF-Token": csrf },
      });
    } catch {
      elements.export.textContent = "Try export again";
      elements.export.disabled = false;
      elements.exportStatus.textContent = "Export was interrupted because the local reviewer disconnected. Reload the page, then try again.";
      showToast("Export was interrupted. Reload the page and try again.");
      return;
    }
    const payload = await response.json();
    elements.export.textContent = "Export reviewed transcript";
    elements.export.disabled = !state.progress.export_ready;
    if (!response.ok) {
      elements.exportStatus.textContent = "Export did not run. Finish every pending span, then try again.";
      showToast("Export remains blocked until every span is resolved.");
      return;
    }
    state.export = payload;
    elements.export.textContent = "Export again";
    showToast(`Export complete · quality ${payload.quality_status}`);
    elements.exportStatus.textContent = `Export succeeded · quality ${payload.quality_status} · no GIGA event emitted.`;
    elements.completionCopy.textContent = `Reviewed transcript exported locally with quality status ${payload.quality_status}. No GIGA event was emitted.`;
  }

  elements.list.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-review-index]");
    if (button) select(Number(button.dataset.reviewIndex), { focus: true });
  });
  elements.approve.addEventListener("click", () => saveDecision("approved"));
  elements.correct.addEventListener("click", () => saveDecision("corrected"));
  elements.unclear.addEventListener("click", () => saveDecision("unclear"));
  elements.noSpeech.addEventListener("click", () => saveDecision("no_speech"));
  elements.previous.addEventListener("click", () => select(currentIndex - 1, { focus: true }));
  elements.next.addEventListener("click", () => select(currentIndex + 1, { focus: true }));
  elements.export.addEventListener("click", exportReview);

  document.addEventListener("keydown", (event) => {
    const editing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
    if (editing) {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        saveDecision("corrected");
      }
      return;
    }
    if (event.key === " ") {
      event.preventDefault();
      elements.player.paused ? elements.player.play() : elements.player.pause();
    } else if (event.key.toLowerCase() === "a") {
      saveDecision("approved");
    } else if (event.key.toLowerCase() === "u") {
      saveDecision("unclear");
    } else if (event.key.toLowerCase() === "n") {
      saveDecision("no_speech");
    } else if (event.key === "ArrowLeft") {
      select(currentIndex - 1, { focus: true });
    } else if (event.key === "ArrowRight") {
      select(currentIndex + 1, { focus: true });
    } else if (event.key.toLowerCase() === "e") {
      elements.text.focus();
    }
  });

  fetch("/api/state")
    .then((response) => {
      if (!response.ok) throw new Error("state unavailable");
      return response.json();
    })
    .then((payload) => {
      state = payload;
      const firstPending = state.items.findIndex((item) => state.decisions[item.id].status === "pending");
      currentIndex = firstPending >= 0 ? firstPending : 0;
      elements.loading.hidden = true;
      elements.editor.hidden = false;
      renderProgress();
      renderCurrent();
    })
    .catch(() => {
      elements.loading.textContent = "The local review package could not be loaded.";
    });
})();
