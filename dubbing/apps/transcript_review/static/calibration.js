(() => {
  "use strict";

  const csrf = document.querySelector('meta[name="csrf-token"]').content;
  const elements = {
    list: document.getElementById("clip-list"),
    progress: document.getElementById("progress"),
    progressLabel: document.getElementById("progress-label"),
    position: document.getElementById("clip-position"),
    time: document.getElementById("clip-time"),
    status: document.getElementById("clip-status"),
    loading: document.getElementById("loading-state"),
    editor: document.getElementById("editor"),
    player: document.getElementById("clip-player"),
    turns: document.getElementById("turn-list"),
    template: document.getElementById("turn-template"),
    notes: document.getElementById("clip-notes"),
    addTurn: document.getElementById("add-turn-button"),
    save: document.getElementById("save-button"),
    noSpeech: document.getElementById("no-speech-button"),
    saveStatus: document.getElementById("save-status"),
    previous: document.getElementById("previous-button"),
    next: document.getElementById("next-button"),
    completion: document.getElementById("completion"),
    export: document.getElementById("export-button"),
    exportStatus: document.getElementById("export-status"),
  };

  let state = null;
  let currentIndex = 0;
  let dirty = false;

  const seconds = (milliseconds) => (milliseconds / 1000).toFixed(2);
  const minutes = (milliseconds) => (milliseconds / 60000).toFixed(1);

  function currentItem() { return state.items[currentIndex]; }
  function currentDecision() { return state.decisions[currentItem().id]; }

  function setDirty() {
    dirty = true;
    elements.saveStatus.textContent = "Unsaved changes";
  }

  function turnElement(turn) {
    const row = elements.template.content.firstElementChild.cloneNode(true);
    row.dataset.turnId = turn.id || "";
    row.querySelector('[data-field="start"]').value = seconds(turn.start_ms);
    row.querySelector('[data-field="end"]').value = seconds(turn.end_ms);
    row.querySelector('[data-field="speaker"]').value = turn.speaker;
    row.querySelector('[data-field="language"]').value = turn.language;
    row.querySelector('[data-field="text"]').value = turn.text;
    return row;
  }

  function collectTurns() {
    return [...elements.turns.querySelectorAll(".turn-row")].map((row, index) => ({
      id: row.dataset.turnId || `${currentItem().id}-human-${String(index + 1).padStart(3, "0")}`,
      start_ms: Math.round(Number(row.querySelector('[data-field="start"]').value) * 1000),
      end_ms: Math.round(Number(row.querySelector('[data-field="end"]').value) * 1000),
      speaker: row.querySelector('[data-field="speaker"]').value,
      language: row.querySelector('[data-field="language"]').value,
      text: row.querySelector('[data-field="text"]').value.trim(),
    }));
  }

  function captureDraft() {
    if (!state || !dirty) return;
    const decision = currentDecision();
    decision.turns = collectTurns();
    decision.notes = elements.notes.value.trim();
  }

  function renderProgress() {
    const { progress } = state;
    elements.progress.value = progress.reviewed;
    elements.progress.max = progress.total;
    elements.progressLabel.textContent = `${progress.reviewed} of ${progress.total} clips complete`;
    elements.completion.hidden = !progress.export_ready;
    if (state.export) {
      elements.export.textContent = "Export again";
      elements.exportStatus.textContent = `Last export succeeded · ${state.export.turn_count} timestamped turns · no GIGA event emitted.`;
    }
  }

  function renderList() {
    elements.list.querySelectorAll("button[data-clip-index]").forEach((button, index) => {
      const decision = state.decisions[state.items[index].id];
      button.setAttribute("aria-current", index === currentIndex ? "true" : "false");
      button.dataset.status = decision.status;
      button.lastElementChild.textContent = decision.status.replace("_", " ");
    });
  }

  function renderCurrent({ focus = false } = {}) {
    const item = currentItem();
    const decision = currentDecision();
    elements.position.textContent = `Clip ${currentIndex + 1} of ${state.items.length} · ${item.target}`;
    elements.time.textContent = `${minutes(item.start_ms)}–${minutes(item.end_ms)}m`;
    elements.status.textContent = decision.status.replace("_", " ");
    elements.status.dataset.status = decision.status;
    elements.player.src = `/clips/${encodeURIComponent(item.id)}.wav`;
    elements.turns.replaceChildren(...decision.turns.map(turnElement));
    elements.notes.value = decision.notes;
    elements.previous.disabled = currentIndex === 0;
    elements.next.disabled = currentIndex === state.items.length - 1;
    elements.saveStatus.textContent = decision.status === "pending" ? "Review every prefilled turn before saving." : "Saved locally";
    dirty = false;
    renderList();
    if (focus) elements.player.focus();
  }

  function select(index, options = {}) {
    if (!state || index < 0 || index >= state.items.length) return;
    captureDraft();
    elements.player.pause();
    currentIndex = index;
    renderCurrent(options);
    elements.list.querySelector(`[data-clip-index="${index}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest" });
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
    const turns = status === "no_speech" ? [] : collectTurns();
    elements.save.disabled = true;
    elements.noSpeech.disabled = true;
    elements.saveStatus.textContent = "Saving clip…";
    let response;
    try {
      response = await fetch(`/api/decision/${encodeURIComponent(item.id)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
        body: JSON.stringify({ status, turns, notes: elements.notes.value }),
      });
    } catch {
      elements.saveStatus.textContent = "Could not reach the local reviewer. Reload and try again.";
      elements.save.disabled = false;
      elements.noSpeech.disabled = false;
      return;
    }
    elements.save.disabled = false;
    elements.noSpeech.disabled = false;
    if (!response.ok) {
      elements.saveStatus.textContent = "Clip was not saved. Check timestamps, text, language and speaker labels.";
      return;
    }
    const payload = await response.json();
    state.decisions[item.id] = {
      status,
      turns,
      notes: elements.notes.value.trim(),
    };
    state.progress = payload.progress;
    dirty = false;
    renderProgress();
    renderList();
    elements.saveStatus.textContent = status === "no_speech" ? "Saved as no speech" : "Clip saved locally";
    const destination = nextPending();
    if (destination !== currentIndex) select(destination, { focus: true });
    else renderCurrent();
  }

  async function exportGroundTruth() {
    elements.export.disabled = true;
    elements.export.textContent = "Exporting…";
    elements.exportStatus.textContent = "Binding timestamps, speaker labels and correction lineage…";
    let response;
    try {
      response = await fetch("/api/export", {
        method: "POST",
        headers: { "X-CSRF-Token": csrf },
      });
    } catch {
      elements.export.disabled = false;
      elements.export.textContent = "Try export again";
      elements.exportStatus.textContent = "Export was interrupted. Reload the page and try again.";
      return;
    }
    const payload = await response.json();
    elements.export.disabled = false;
    if (!response.ok) {
      elements.export.textContent = "Export ground truth";
      elements.exportStatus.textContent = "Export is blocked until every clip is saved.";
      return;
    }
    state.export = payload;
    elements.export.textContent = "Export again";
    elements.exportStatus.textContent = `Export succeeded · ${payload.turn_count} timestamped turns · no GIGA event emitted.`;
  }

  elements.list.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-clip-index]");
    if (button) select(Number(button.dataset.clipIndex), { focus: true });
  });
  elements.turns.addEventListener("input", setDirty);
  elements.notes.addEventListener("input", setDirty);
  elements.turns.addEventListener("click", (event) => {
    const action = event.target.closest("button[data-action]");
    if (!action) return;
    const row = action.closest(".turn-row");
    if (action.dataset.action === "remove") row.remove();
    if (action.dataset.action === "start-now") row.querySelector('[data-field="start"]').value = elements.player.currentTime.toFixed(2);
    if (action.dataset.action === "end-now") row.querySelector('[data-field="end"]').value = elements.player.currentTime.toFixed(2);
    setDirty();
  });
  elements.addTurn.addEventListener("click", () => {
    const start = Math.min(elements.player.currentTime, currentItem().duration_ms / 1000 - 0.1);
    elements.turns.append(turnElement({
      id: "",
      start_ms: Math.round(start * 1000),
      end_ms: Math.round(Math.min(start + 3, currentItem().duration_ms / 1000) * 1000),
      speaker: "UNKNOWN",
      language: "unknown",
      text: "[unclear]",
    }));
    setDirty();
  });
  elements.save.addEventListener("click", () => saveDecision("complete"));
  elements.noSpeech.addEventListener("click", () => saveDecision("no_speech"));
  elements.previous.addEventListener("click", () => select(currentIndex - 1, { focus: true }));
  elements.next.addEventListener("click", () => select(currentIndex + 1, { focus: true }));
  elements.export.addEventListener("click", exportGroundTruth);
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      saveDecision("complete");
    }
  });
  window.addEventListener("beforeunload", (event) => {
    if (dirty) event.preventDefault();
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
      elements.loading.textContent = "The local calibration package could not be loaded.";
    });
})();
