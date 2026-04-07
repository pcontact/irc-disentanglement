(function () {
  "use strict";

  const palette = [
    "#1b80bf",
    "#1d9a6c",
    "#bf6a1b",
    "#944fbc",
    "#b43e5a",
    "#5d7c2f",
    "#0b8b8c",
    "#cc8b19",
    "#5d57d7",
    "#b15423",
  ];

  const state = {
    fileName: "",
    rows: [],
    edges: [],
    incoming: new Map(),
    adjacency: new Map(),
    componentMembers: new Map(),
    hoveredId: null,
    pinnedComponentId: null,
    diagnostics: {
      rows: 0,
      roots: 0,
      invalidTargets: [],
      duplicateIds: [],
      parseErrors: [],
    },
  };

  const dom = {
    dropzone: document.getElementById("dropzone"),
    fileInput: document.getElementById("file-input"),
    resetButton: document.getElementById("reset-button"),
    loadedFileName: document.getElementById("loaded-file-name"),
    viewerTitle: document.getElementById("viewer-title"),
    statsGrid: document.getElementById("stats-grid"),
    validationList: document.getElementById("validation-list"),
    viewerViewport: document.getElementById("viewer-viewport"),
    viewerScene: document.getElementById("viewer-scene"),
    transcriptList: document.getElementById("transcript-list"),
    edgesLayer: document.getElementById("edges-layer"),
  };

  let edgeDrawScheduled = false;
  let resizeObserver = null;

  init();

  function init() {
    bindControls();
    setupResizeHandling();
    renderEmptyState({
      resetFileName: true,
      title: "No transcript loaded",
      message: "Use the drop zone above to load a transcript JSONL file.",
    });
  }

  function bindControls() {
    dom.dropzone.addEventListener("click", () => dom.fileInput.click());
    dom.dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        dom.fileInput.click();
      }
    });
    dom.fileInput.addEventListener("change", async (event) => {
      const [file] = event.target.files || [];
      if (file) {
        await loadFile(file);
      }
      event.target.value = "";
    });

    ["dragenter", "dragover"].forEach((name) => {
      dom.dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dom.dropzone.classList.add("is-dragging");
      });
    });
    ["dragleave", "dragend", "drop"].forEach((name) => {
      dom.dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dom.dropzone.classList.remove("is-dragging");
      });
    });
    dom.dropzone.addEventListener("drop", async (event) => {
      const [file] = event.dataTransfer.files || [];
      if (file) {
        await loadFile(file);
      }
    });

    dom.resetButton.addEventListener("click", () => {
      state.hoveredId = null;
      state.pinnedComponentId = null;
      applyHighlightState();
    });

    dom.viewerViewport.addEventListener("scroll", scheduleEdgeDraw, { passive: true });
    window.addEventListener("resize", scheduleEdgeDraw);
  }

  function setupResizeHandling() {
    if (!("ResizeObserver" in window)) {
      return;
    }
    resizeObserver = new ResizeObserver(() => {
      scheduleEdgeDraw();
    });
    resizeObserver.observe(dom.viewerViewport);
    resizeObserver.observe(dom.transcriptList);
  }

  async function loadFile(file) {
    const text = await file.text();
    const parsed = parseJsonl(text);

    state.fileName = file.name;
    state.rows = parsed.rows;
    state.edges = parsed.edges;
    state.incoming = parsed.incoming;
    state.adjacency = parsed.adjacency;
    state.componentMembers = parsed.componentMembers;
    state.hoveredId = null;
    state.pinnedComponentId = null;
    state.diagnostics = parsed.diagnostics;

    renderTranscript();
    renderDiagnostics();
    applyHighlightState();
    scheduleEdgeDraw();
  }

  function parseJsonl(text) {
    const diagnostics = {
      rows: 0,
      roots: 0,
      invalidTargets: [],
      duplicateIds: [],
      parseErrors: [],
    };

    const rawRows = [];
    const seenIds = new Map();

    text.split(/\r?\n/).forEach((line, index) => {
      const lineNumber = index + 1;
      const trimmed = line.trim();
      if (!trimmed) {
        return;
      }

      let parsedLine;
      try {
        parsedLine = JSON.parse(line);
      } catch (error) {
        diagnostics.parseErrors.push(`Line ${lineNumber}: ${error.message}`);
        return;
      }

      if (!Number.isInteger(parsedLine.id)) {
        diagnostics.parseErrors.push(`Line ${lineNumber}: missing integer "id"`);
        return;
      }

      if (typeof parsedLine.message !== "string") {
        diagnostics.parseErrors.push(`Line ${lineNumber}: missing string "message"`);
        return;
      }

      let targetId = null;
      if (Number.isInteger(parsedLine.outgoing_edge_target_id)) {
        targetId = parsedLine.outgoing_edge_target_id;
      } else {
        diagnostics.invalidTargets.push(
          `id ${parsedLine.id}: outgoing_edge_target_id is not an integer`
        );
      }

      if (seenIds.has(parsedLine.id)) {
        diagnostics.duplicateIds.push(
          `id ${parsedLine.id} appears on lines ${seenIds.get(parsedLine.id)} and ${lineNumber}`
        );
        return;
      }

      seenIds.set(parsedLine.id, lineNumber);
      rawRows.push({
        id: parsedLine.id,
        message: parsedLine.message,
        targetId,
        lineNumber,
      });
    });

    rawRows.sort((left, right) => left.id - right.id);

    const idToRow = new Map(rawRows.map((row, index) => [row.id, { ...row, sortedIndex: index }]));
    const adjacency = new Map();
    const incoming = new Map();

    rawRows.forEach((row) => {
      adjacency.set(row.id, new Set());
      incoming.set(row.id, []);
    });

    const edges = [];
    rawRows.forEach((row) => {
      if (row.targetId === null) {
        return;
      }
      if (row.targetId === -1) {
        diagnostics.roots += 1;
        return;
      }
      if (!idToRow.has(row.targetId)) {
        diagnostics.invalidTargets.push(`id ${row.id}: target ${row.targetId} is missing`);
        return;
      }

      edges.push({
        sourceId: row.id,
        targetId: row.targetId,
      });
      incoming.get(row.targetId).push(row.id);
      adjacency.get(row.id).add(row.targetId);
      adjacency.get(row.targetId).add(row.id);
    });

    const componentMembers = new Map();
    const visited = new Set();
    let nextComponentId = 0;

    rawRows.forEach((row) => {
      if (visited.has(row.id)) {
        return;
      }
      const componentId = `component-${nextComponentId++}`;
      const stack = [row.id];
      const members = [];
      visited.add(row.id);

      while (stack.length) {
        const currentId = stack.pop();
        members.push(currentId);
        for (const neighborId of adjacency.get(currentId)) {
          if (visited.has(neighborId)) {
            continue;
          }
          visited.add(neighborId);
          stack.push(neighborId);
        }
      }

      members.sort((left, right) => left - right);
      componentMembers.set(componentId, members);
      members.forEach((memberId) => {
        const rowState = idToRow.get(memberId);
        rowState.componentId = componentId;
      });
    });

    const rows = rawRows.map((row) => {
      const enriched = idToRow.get(row.id);
      const degree = adjacency.get(row.id).size;
      const hasInvalidTarget = row.targetId !== null && row.targetId !== -1 && !idToRow.has(row.targetId);
      return {
        ...enriched,
        degree,
        incomingCount: incoming.get(row.id).length,
        hasInvalidTarget,
        isNeutral: degree === 0 || row.targetId === null || hasInvalidTarget,
        color: colorForComponent(enriched.componentId),
      };
    });

    edges.forEach((edge) => {
      const componentId = idToRow.get(edge.sourceId).componentId;
      edge.componentId = componentId;
      edge.color = colorForComponent(componentId);
    });

    diagnostics.rows = rows.length;
    return {
      rows,
      edges,
      incoming,
      adjacency,
      componentMembers,
      diagnostics,
    };
  }

  function colorForComponent(componentId) {
    const numericPart = Number(componentId.split("-").pop() || 0);
    return palette[numericPart % palette.length];
  }

  function renderTranscript() {
    dom.loadedFileName.textContent = state.fileName || "None yet";
    dom.viewerTitle.textContent = state.rows.length
      ? `${state.rows.length} messages in ${state.fileName}`
      : "Waiting for a file";
    dom.resetButton.disabled = !state.rows.length;

    if (!state.rows.length) {
      renderEmptyState({
        title: state.fileName ? "No renderable rows found" : "No transcript loaded",
        message: state.fileName
          ? "This file did not contain any valid transcript rows. Check the validation panel for details."
          : "Use the drop zone above to load a transcript JSONL file.",
      });
      return;
    }

    const fragment = document.createDocumentFragment();

    state.rows.forEach((row) => {
      const element = document.createElement("article");
      element.className = "message-row";
      if (row.isNeutral) {
        element.classList.add("is-neutral");
      }
      element.dataset.id = String(row.id);
      element.dataset.componentId = row.componentId;
      element.style.setProperty("--thread-color", row.color);
      element.title = buildRowTitle(row);

      const idCell = document.createElement("span");
      idCell.className = "row-id";
      idCell.textContent = row.id;

      const messageCell = document.createElement("p");
      messageCell.className = "row-message";
      messageCell.textContent = row.message;

      element.appendChild(idCell);
      element.appendChild(messageCell);
      bindRowInteractions(element, row);
      fragment.appendChild(element);
    });

    dom.transcriptList.replaceChildren(fragment);
  }

  function buildRowTitle(row) {
    if (row.targetId === -1) {
      return `id ${row.id}: root message`;
    }
    if (row.targetId === null) {
      return `id ${row.id}: invalid outgoing_edge_target_id`;
    }
    return `id ${row.id}: replies to ${row.targetId}`;
  }

  function bindRowInteractions(element, row) {
    element.addEventListener("mouseenter", () => {
      state.hoveredId = row.id;
      applyHighlightState();
    });
    element.addEventListener("mouseleave", () => {
      state.hoveredId = null;
      applyHighlightState();
    });
    element.addEventListener("click", () => {
      state.pinnedComponentId =
        state.pinnedComponentId === row.componentId ? null : row.componentId;
      applyHighlightState();
    });
  }

  function renderDiagnostics() {
    const counts = [
      state.diagnostics.rows,
      state.diagnostics.roots,
      state.diagnostics.invalidTargets.length,
      state.diagnostics.duplicateIds.length,
      state.diagnostics.parseErrors.length,
    ];

    Array.from(dom.statsGrid.querySelectorAll("dd")).forEach((node, index) => {
      node.textContent = String(counts[index] || 0);
    });

    const details = [];

    if (!state.fileName) {
      details.push(createDiagnosticItem("No file loaded yet."));
    } else {
      details.push(
        createDiagnosticItem(
          `${state.diagnostics.rows} valid rows loaded from ${state.fileName}.`
        )
      );
      if (!state.rows.length) {
        details.push(createDiagnosticItem("No renderable rows were produced from this file.", true));
      }
      if (!state.diagnostics.invalidTargets.length) {
        details.push(createDiagnosticItem("No invalid targets detected."));
      }
      state.diagnostics.invalidTargets
        .slice(0, 8)
        .forEach((message) => details.push(createDiagnosticItem(message, true)));
      state.diagnostics.duplicateIds
        .slice(0, 8)
        .forEach((message) => details.push(createDiagnosticItem(message, true)));
      state.diagnostics.parseErrors
        .slice(0, 8)
        .forEach((message) => details.push(createDiagnosticItem(message, true)));
    }

    dom.validationList.replaceChildren(...details);
  }

  function createDiagnosticItem(text, isWarning) {
    const item = document.createElement("li");
    item.textContent = text;
    if (isWarning) {
      item.classList.add("warning");
    }
    return item;
  }

  function renderEmptyState(options) {
    const settings = options || {};
    if (settings.resetFileName) {
      dom.loadedFileName.textContent = "None yet";
      dom.viewerTitle.textContent = "Waiting for a file";
    } else {
      dom.loadedFileName.textContent = state.fileName || "None yet";
      dom.viewerTitle.textContent = state.fileName
        ? `${state.rows.length} valid messages in ${state.fileName}`
        : "Waiting for a file";
    }
    dom.transcriptList.innerHTML = [
      '<div class="empty-state">',
      `<h3>${settings.title || "No transcript loaded"}</h3>`,
      `<p>${settings.message || "Use the drop zone above to load a transcript JSONL file."}</p>`,
      "</div>",
    ].join("");
    dom.edgesLayer.replaceChildren();
  }

  function applyHighlightState() {
    const localIds = getLocalHighlightIds();
    const pinnedMembers = state.pinnedComponentId
      ? new Set(state.componentMembers.get(state.pinnedComponentId) || [])
      : null;

    dom.transcriptList.querySelectorAll(".message-row").forEach((element) => {
      const rowId = Number(element.dataset.id);
      const componentId = element.dataset.componentId;
      const isFocused = rowId === state.hoveredId;
      const isNeighbor = state.hoveredId !== null && localIds.has(rowId) && !isFocused;
      const isPinned =
        state.pinnedComponentId !== null && componentId === state.pinnedComponentId;

      let shouldDim = false;
      if (state.pinnedComponentId !== null && componentId !== state.pinnedComponentId) {
        shouldDim = true;
      }
      if (state.hoveredId !== null) {
        const inLocalGroup = localIds.has(rowId);
        if (state.pinnedComponentId === null && !inLocalGroup) {
          shouldDim = true;
        }
        if (state.pinnedComponentId !== null && pinnedMembers && pinnedMembers.has(rowId) && !inLocalGroup) {
          shouldDim = true;
        }
      }

      element.classList.toggle("is-focused", isFocused);
      element.classList.toggle("is-neighbor", isNeighbor);
      element.classList.toggle("is-pinned", isPinned);
      element.classList.toggle("is-dim", shouldDim);
    });

    dom.edgesLayer.querySelectorAll(".thread-edge").forEach((path) => {
      const sourceId = Number(path.dataset.sourceId);
      const targetId = Number(path.dataset.targetId);
      const componentId = path.dataset.componentId;
      const isLocalEdge = state.hoveredId !== null && (sourceId === state.hoveredId || targetId === state.hoveredId);
      const isPinned = state.pinnedComponentId !== null && componentId === state.pinnedComponentId;

      let shouldDim = false;
      if (state.pinnedComponentId !== null && componentId !== state.pinnedComponentId) {
        shouldDim = true;
      }
      if (state.hoveredId !== null) {
        if (state.pinnedComponentId === null && !isLocalEdge) {
          shouldDim = true;
        }
        if (state.pinnedComponentId !== null && isPinned && !isLocalEdge) {
          shouldDim = true;
        }
      }

      path.classList.toggle("is-focused", isLocalEdge);
      path.classList.toggle("is-dim", shouldDim);
    });
  }

  function getLocalHighlightIds() {
    if (state.hoveredId === null) {
      return new Set();
    }
    const result = new Set([state.hoveredId]);
    const hoveredRow = state.rows.find((row) => row.id === state.hoveredId);
    if (!hoveredRow) {
      return result;
    }
    if (hoveredRow.targetId !== null && hoveredRow.targetId !== -1) {
      result.add(hoveredRow.targetId);
    }
    (state.incoming.get(state.hoveredId) || []).forEach((id) => result.add(id));
    return result;
  }

  function scheduleEdgeDraw() {
    if (edgeDrawScheduled) {
      return;
    }
    edgeDrawScheduled = true;
    window.requestAnimationFrame(() => {
      edgeDrawScheduled = false;
      drawEdges();
    });
  }

  function drawEdges() {
    if (!state.rows.length) {
      dom.edgesLayer.replaceChildren();
      return;
    }

    const rowElements = new Map(
      Array.from(dom.transcriptList.querySelectorAll(".message-row")).map((element) => [
        Number(element.dataset.id),
        element,
      ])
    );

    if (!rowElements.size) {
      dom.edgesLayer.replaceChildren();
      return;
    }

    const sceneWidth = dom.viewerScene.clientWidth;
    const sceneHeight = Math.max(dom.transcriptList.offsetHeight + 48, dom.viewerViewport.clientHeight);
    dom.viewerScene.style.minHeight = `${sceneHeight}px`;
    dom.edgesLayer.setAttribute("viewBox", `0 0 ${sceneWidth} ${sceneHeight}`);
    dom.edgesLayer.setAttribute("width", String(sceneWidth));
    dom.edgesLayer.setAttribute("height", String(sceneHeight));

    const fragment = document.createDocumentFragment();

    state.edges.forEach((edge) => {
      const sourceRow = rowElements.get(edge.sourceId);
      const targetRow = rowElements.get(edge.targetId);
      if (!sourceRow || !targetRow) {
        return;
      }

      const sourceY = sourceRow.offsetTop + sourceRow.offsetHeight / 2;
      const targetY = targetRow.offsetTop + targetRow.offsetHeight / 2;
      const anchorX = sourceRow.offsetLeft - 14;
      const distance = Math.abs(getSortedIndex(edge.sourceId) - getSortedIndex(edge.targetId));
      const depth = Math.min(136, 28 + distance * 7);
      const controlX = Math.max(18, anchorX - depth);
      const pathData = [
        `M ${anchorX} ${sourceY}`,
        `C ${controlX} ${sourceY}, ${controlX} ${targetY}, ${anchorX} ${targetY}`,
      ].join(" ");

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", pathData);
      path.setAttribute("class", "thread-edge");
      path.dataset.sourceId = String(edge.sourceId);
      path.dataset.targetId = String(edge.targetId);
      path.dataset.componentId = edge.componentId;
      path.style.setProperty("--edge-color", edge.color);
      fragment.appendChild(path);
    });

    dom.edgesLayer.replaceChildren(fragment);
    applyHighlightState();
  }

  function getSortedIndex(rowId) {
    const row = state.rows.find((entry) => entry.id === rowId);
    return row ? row.sortedIndex : 0;
  }
})();
