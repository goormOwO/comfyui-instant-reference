import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const EXTENSION_NAME = "comfyui-reference.instant-reference-lora";
const TRAINING_NODE_NAMES = new Set([
  "InstantReferenceLoRA",
  "Instant Reference LoRA",
  "InstantReferenceLoRATrain",
  "Instant Reference LoRA Train",
]);
let profileSlotMapPromise = null;
let cacheRefreshTimer = null;
const CACHE_REFRESH_INTERVAL_MS = 5000;
const LIBRARY_STYLE_ID = "instant-reference-lora-library-style";

async function fetchJson(path, options = {}) {
  const response = await api.fetchApi(path, options);
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.error) {
        message = payload.error;
      }
    } catch {
      // Ignore JSON parsing errors and keep the generic message.
    }
    throw new Error(message);
  }
  return response.json();
}

async function getProfileSlotMap() {
  if (!profileSlotMapPromise) {
    profileSlotMapPromise = fetchJson("/instant-reference-lora/profiles")
      .then((payload) => payload?.profiles || {})
      .catch((error) => {
        profileSlotMapPromise = null;
        throw error;
      });
  }
  return profileSlotMapPromise;
}

function showToast(severity, summary, detail) {
  app.extensionManager?.toast?.add?.({
    severity,
    summary,
    detail,
    life: 4000,
  });
}

function extractStringOutput(message) {
  const candidates = [
    message?.string,
    message?.text,
    message?.lora_path,
    message?.loraPath,
  ];

  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) {
      return candidate;
    }
    if (Array.isArray(candidate)) {
      const firstString = candidate.find((value) => typeof value === "string" && value.trim());
      if (firstString) {
        return firstString;
      }
      for (const item of candidate) {
        if (Array.isArray(item)) {
          const nested = item.find((value) => typeof value === "string" && value.trim());
          if (nested) {
            return nested;
          }
        }
      }
    }
  }

  return "";
}

function nodeMatches(nodeData, names) {
  return names.has(nodeData?.name) || names.has(nodeData?.display_name);
}

function isTrainingNode(nodeData) {
  return nodeMatches(nodeData, TRAINING_NODE_NAMES);
}

function isTrainingNodeInstance(node) {
  return TRAINING_NODE_NAMES.has(node?.type) || TRAINING_NODE_NAMES.has(node?.comfyClass);
}

function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name) ?? null;
}

function findInputIndex(node, name) {
  return node.inputs?.findIndex((input) => input.name === name) ?? -1;
}

function ensureOptionalInput(node, name, type) {
  if (findInputIndex(node, name) !== -1) {
    return;
  }
  node.addInput(name, type);
}

function removeOptionalInput(node, name) {
  const index = findInputIndex(node, name);
  if (index === -1) {
    return;
  }
  if (typeof node.removeInput === "function") {
    node.removeInput(index);
  } else {
    node.inputs.splice(index, 1);
  }
}

function getManagedSlots(profiles) {
  const managedSlots = new Map();
  for (const profile of Object.values(profiles || {})) {
    for (const slot of profile?.slots || []) {
      if (!slot?.name || !slot?.type) {
        continue;
      }
      if (slot.type === "MODEL" || slot.type === "CLIP") {
        continue;
      }
      if (!managedSlots.has(slot.name)) {
        managedSlots.set(slot.name, slot.type);
      }
    }
  }
  return managedSlots;
}

function scheduleProfileInputSync(node) {
  if (!node) {
    return;
  }
  if (node.__instantReferenceLoraProfileSyncTimer) {
    window.clearTimeout(node.__instantReferenceLoraProfileSyncTimer);
  }
  node.__instantReferenceLoraProfileSyncTimer = window.setTimeout(() => {
    node.__instantReferenceLoraProfileSyncTimer = null;
    syncProfileInputs(node);
  }, 0);
}

async function syncProfileInputs(node) {
  const profileWidget = findWidget(node, "profile");
  if (!profileWidget) {
    return;
  }

  let profiles;
  try {
    profiles = await getProfileSlotMap();
  } catch (error) {
    showToast("error", "Instant Reference LoRA", error.message);
    return;
  }

  const selectedKey = profileWidget.value;
  const selectedProfile = profiles?.[selectedKey];
  const requiredSlots = new Map((selectedProfile?.slots || []).map((slot) => [slot.name, slot.type]));
  const managedSlots = getManagedSlots(profiles);

  for (const [slotName, slotType] of managedSlots.entries()) {
    if (requiredSlots.has(slotName)) {
      ensureOptionalInput(node, slotName, requiredSlots.get(slotName) || slotType);
    } else {
      removeOptionalInput(node, slotName);
    }
  }

  node.setDirtyCanvas(true, true);
}

async function refreshCacheInfo(node) {
  const clearCacheWidget = node.__instantReferenceLoraClearCacheWidget;
  if (!clearCacheWidget) {
    return;
  }
  const baseLabel = "Clear Cache";
  if (node.__instantReferenceLoraRefreshingCache) {
    return;
  }
  node.__instantReferenceLoraRefreshingCache = true;
  clearCacheWidget.name = `${baseLabel} (...)`;
  node.setDirtyCanvas(true, true);
  try {
    const payload = await fetchJson("/instant-reference-lora/cache-info");
    clearCacheWidget.name = `${baseLabel} (${payload.total_human})`;
  } catch (error) {
    clearCacheWidget.name = `${baseLabel} (?)`;
    showToast("error", "Instant Reference LoRA", error.message);
  }
  node.__instantReferenceLoraRefreshingCache = false;
  node.setDirtyCanvas(true, true);
}

function startAutoCacheRefresh() {
  if (cacheRefreshTimer) {
    return;
  }
  cacheRefreshTimer = window.setInterval(() => {
    const nodes = app.graph?._nodes || [];
    for (const node of nodes) {
      if (isTrainingNodeInstance(node)) {
        refreshCacheInfo(node);
      }
    }
  }, CACHE_REFRESH_INTERVAL_MS);
}

function ensureLibraryStyles() {
  if (document.getElementById(LIBRARY_STYLE_ID)) {
    return;
  }
  const style = document.createElement("style");
  style.id = LIBRARY_STYLE_ID;
  style.textContent = `
    .ir-lora-library-backdrop {
      position: fixed;
      inset: 0;
      z-index: 10000;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.62);
    }
    .ir-lora-library {
      width: min(1120px, calc(100vw - 36px));
      height: min(780px, calc(100vh - 36px));
      display: flex;
      flex-direction: column;
      overflow: hidden;
      border: 1px solid #4b5563;
      border-radius: 8px;
      background: #15171a;
      color: #f3f4f6;
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.45);
      font-family: Arial, sans-serif;
    }
    .ir-lora-library-header {
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      padding: 14px 16px;
      border-bottom: 1px solid #374151;
    }
    .ir-lora-library-title {
      font-size: 16px;
      font-weight: 700;
    }
    .ir-lora-library-subtitle {
      margin-top: 4px;
      color: #a7b0bd;
      font-size: 12px;
    }
    .ir-lora-library-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .ir-lora-library-button {
      min-height: 32px;
      border: 1px solid #64748b;
      border-radius: 6px;
      padding: 6px 10px;
      background: #23272f;
      color: #f8fafc;
      cursor: pointer;
      font-size: 12px;
    }
    .ir-lora-library-button:hover {
      background: #2f3641;
    }
    .ir-lora-library-button-danger {
      border-color: #b45353;
      background: #3a2020;
    }
    .ir-lora-library-body {
      min-height: 0;
      overflow: auto;
      padding: 16px;
    }
    .ir-lora-library-status {
      padding: 32px 12px;
      color: #cbd5e1;
      text-align: center;
    }
    .ir-lora-library-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 12px;
    }
    .ir-lora-library-card {
      overflow: hidden;
      border: 1px solid #3f4754;
      border-radius: 8px;
      background: #1c2026;
    }
    .ir-lora-library-thumb {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      display: block;
      background: #2a3038;
    }
    .ir-lora-library-placeholder {
      width: 100%;
      aspect-ratio: 1 / 1;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #2a3038;
      color: #a7b0bd;
      font-size: 13px;
    }
    .ir-lora-library-card-body {
      padding: 10px;
    }
    .ir-lora-library-name {
      overflow-wrap: anywhere;
      color: #f8fafc;
      font-size: 13px;
      font-weight: 700;
    }
    .ir-lora-library-meta {
      margin-top: 6px;
      color: #b8c0cc;
      font-size: 12px;
      line-height: 1.45;
    }
    .ir-lora-library-tags {
      margin-top: 8px;
      max-height: 42px;
      overflow: hidden;
      color: #d6dde7;
      font-size: 12px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }
    .ir-lora-library-card-actions {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      margin-top: 10px;
    }
  `;
  document.head.appendChild(style);
}

function formatLibraryDate(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "";
  }
  return new Date(value * 1000).toLocaleString();
}

async function saveLibraryItem(item, refresh) {
  const fallback = item.file_name || `${item.name || "instant_reference_lora"}.safetensors`;
  const filename = window.prompt("Save LoRA as", fallback);
  if (filename === null) {
    return;
  }
  const payload = await fetchJson("/instant-reference-lora/library/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: item.id, filename }),
  });
  showToast("info", "Instant Reference LoRA", `Saved to ${payload.saved_lora_path}`);
  await refresh();
}

async function deleteLibraryItem(item, refresh) {
  const confirmed = window.confirm(`Delete generated LoRA "${item.file_name}"?`);
  if (!confirmed) {
    return;
  }
  await fetchJson("/instant-reference-lora/library/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: item.id }),
  });
  showToast("warn", "Instant Reference LoRA", "Generated LoRA deleted.");
  await refresh();
}

function createLibraryCard(item, refresh) {
  const card = document.createElement("div");
  card.className = "ir-lora-library-card";

  if (item.has_thumbnail && item.thumbnail_url) {
    const image = document.createElement("img");
    image.className = "ir-lora-library-thumb";
    image.loading = "lazy";
    image.src = api.apiURL(item.thumbnail_url);
    image.alt = item.file_name || "LoRA thumbnail";
    card.appendChild(image);
  } else {
    const placeholder = document.createElement("div");
    placeholder.className = "ir-lora-library-placeholder";
    placeholder.textContent = "No thumbnail";
    card.appendChild(placeholder);
  }

  const body = document.createElement("div");
  body.className = "ir-lora-library-card-body";

  const name = document.createElement("div");
  name.className = "ir-lora-library-name";
  name.textContent = item.file_name || item.name || item.id;
  body.appendChild(name);

  const meta = document.createElement("div");
  meta.className = "ir-lora-library-meta";
  const profile = item.profile_name || item.profile || "unknown profile";
  const saved = item.saved ? "Saved" : "Generated";
  meta.textContent = `${profile} · ${item.size_human || ""} · ${saved}`;
  body.appendChild(meta);

  const date = document.createElement("div");
  date.className = "ir-lora-library-meta";
  date.textContent = formatLibraryDate(item.modified_at);
  body.appendChild(date);

  if (item.tags) {
    const tags = document.createElement("div");
    tags.className = "ir-lora-library-tags";
    tags.textContent = item.tags;
    body.appendChild(tags);
  }

  const actions = document.createElement("div");
  actions.className = "ir-lora-library-card-actions";

  const saveButton = document.createElement("button");
  saveButton.className = "ir-lora-library-button";
  saveButton.textContent = "Save";
  saveButton.addEventListener("click", async () => {
    try {
      await saveLibraryItem(item, refresh);
    } catch (error) {
      showToast("error", "Instant Reference LoRA", error.message);
    }
  });
  actions.appendChild(saveButton);

  const deleteButton = document.createElement("button");
  deleteButton.className = "ir-lora-library-button ir-lora-library-button-danger";
  deleteButton.textContent = "Delete";
  deleteButton.addEventListener("click", async () => {
    try {
      await deleteLibraryItem(item, refresh);
    } catch (error) {
      showToast("error", "Instant Reference LoRA", error.message);
    }
  });
  actions.appendChild(deleteButton);

  body.appendChild(actions);
  card.appendChild(body);
  return card;
}

function openLoraLibrary() {
  ensureLibraryStyles();

  const backdrop = document.createElement("div");
  backdrop.className = "ir-lora-library-backdrop";

  const modal = document.createElement("div");
  modal.className = "ir-lora-library";
  backdrop.appendChild(modal);

  const header = document.createElement("div");
  header.className = "ir-lora-library-header";
  modal.appendChild(header);

  const titleWrap = document.createElement("div");
  const title = document.createElement("div");
  title.className = "ir-lora-library-title";
  title.textContent = "Instant Reference LoRA Library";
  const subtitle = document.createElement("div");
  subtitle.className = "ir-lora-library-subtitle";
  subtitle.textContent = "Review generated LoRAs, save keepers, or delete misses.";
  titleWrap.appendChild(title);
  titleWrap.appendChild(subtitle);
  header.appendChild(titleWrap);

  const headerActions = document.createElement("div");
  headerActions.className = "ir-lora-library-actions";
  header.appendChild(headerActions);

  const refreshButton = document.createElement("button");
  refreshButton.className = "ir-lora-library-button";
  refreshButton.textContent = "Refresh";
  headerActions.appendChild(refreshButton);

  const closeButton = document.createElement("button");
  closeButton.className = "ir-lora-library-button";
  closeButton.textContent = "Close";
  headerActions.appendChild(closeButton);

  const body = document.createElement("div");
  body.className = "ir-lora-library-body";
  modal.appendChild(body);

  const close = () => backdrop.remove();
  closeButton.addEventListener("click", close);
  backdrop.addEventListener("click", (event) => {
    if (event.target === backdrop) {
      close();
    }
  });

  const refresh = async () => {
    body.innerHTML = "";
    const loading = document.createElement("div");
    loading.className = "ir-lora-library-status";
    loading.textContent = "Loading LoRAs...";
    body.appendChild(loading);
    try {
      const payload = await fetchJson("/instant-reference-lora/library");
      body.innerHTML = "";
      const items = Array.isArray(payload.items) ? payload.items : [];
      subtitle.textContent = `${items.length} generated LoRAs · Save copies to ${payload.lora_dir || "LoRA folder"}`;
      if (!items.length) {
        const empty = document.createElement("div");
        empty.className = "ir-lora-library-status";
        empty.textContent = "No generated LoRAs yet.";
        body.appendChild(empty);
        return;
      }
      const grid = document.createElement("div");
      grid.className = "ir-lora-library-grid";
      for (const item of items) {
        grid.appendChild(createLibraryCard(item, refresh));
      }
      body.appendChild(grid);
    } catch (error) {
      body.innerHTML = "";
      const failed = document.createElement("div");
      failed.className = "ir-lora-library-status";
      failed.textContent = error.message;
      body.appendChild(failed);
      showToast("error", "Instant Reference LoRA", error.message);
    }
  };

  refreshButton.addEventListener("click", refresh);
  document.body.appendChild(backdrop);
  refresh();
}

function ensureNodeWidgets(node) {
  if (node.__instantReferenceLoraWidgetsReady) {
    return;
  }
  node.__instantReferenceLoraWidgetsReady = true;

  node.addWidget("button", "Open Profiles Folder", null, async () => {
    try {
      await fetchJson("/instant-reference-lora/open-profiles", { method: "POST" });
      showToast("info", "Instant Reference LoRA", "Opened profiles folder.");
    } catch (error) {
      showToast("error", "Instant Reference LoRA", error.message);
    }
  }, { serialize: false });

  node.addWidget("button", "Open LoRA Library", null, () => {
    openLoraLibrary();
  }, { serialize: false });

  const clearCacheWidget = node.addWidget("button", "Clear Cache (...)", null, async () => {
    const confirmed = window.confirm("Clear the Instant Reference LoRA cache?");
    if (!confirmed) {
      return;
    }
    try {
      const payload = await fetchJson("/instant-reference-lora/clear-cache", { method: "POST" });
      showToast("warn", "Instant Reference LoRA", `Cache cleared. Remaining: ${payload.total_human}`);
      await refreshCacheInfo(node);
    } catch (error) {
      showToast("error", "Instant Reference LoRA", error.message);
    }
  }, { serialize: false });
  node.__instantReferenceLoraClearCacheWidget = clearCacheWidget;

  refreshCacheInfo(node);
  scheduleProfileInputSync(node);
  startAutoCacheRefresh();
}

app.registerExtension({
  name: EXTENSION_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isTrainingNode(nodeData)) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      ensureNodeWidgets(this);
      const profileWidget = findWidget(this, "profile");
      if (profileWidget && !profileWidget.__instantReferenceLoraWrapped) {
        const originalCallback = profileWidget.callback;
        profileWidget.callback = (...args) => {
          const callbackResult = originalCallback ? originalCallback.apply(profileWidget, args) : undefined;
          scheduleProfileInputSync(this);
          return callbackResult;
        };
        profileWidget.__instantReferenceLoraWrapped = true;
      }
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      ensureNodeWidgets(this);
      scheduleProfileInputSync(this);
      return result;
    };

    const onAdded = nodeType.prototype.onAdded;
    nodeType.prototype.onAdded = function () {
      const result = onAdded ? onAdded.apply(this, arguments) : undefined;
      ensureNodeWidgets(this);
      scheduleProfileInputSync(this);
      return result;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
      const loraPath = extractStringOutput(message);
      if (typeof loraPath === "string" && loraPath.trim()) {
        this.__instantReferenceLoraLoraPath = loraPath;
      }
      return result;
    };
  },
});
