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

async function downloadLora(node) {
  let loraPath = node.__instantReferenceLoraLoraPath;
  if (!loraPath) {
    try {
      const payload = await fetchJson("/instant-reference-lora/last-lora");
      if (payload?.exists && typeof payload.path === "string" && payload.path.trim()) {
        loraPath = payload.path;
        node.__instantReferenceLoraLoraPath = loraPath;
      }
    } catch (error) {
      showToast("error", "Instant Reference LoRA", error.message);
      return;
    }
  }
  if (!loraPath) {
    showToast("warn", "Instant Reference LoRA", "Run the node first to create a LoRA.");
    return;
  }

  try {
    const query = new URLSearchParams({ path: loraPath });
    const response = await api.fetchApi(`/instant-reference-lora/download?${query.toString()}`);
    if (!response.ok) {
      let message = `Download failed: ${response.status}`;
      try {
        const payload = await response.json();
        if (payload?.error) {
          message = payload.error;
        }
      } catch {
        // Keep the generic message.
      }
      throw new Error(message);
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = loraPath.split(/[\\/]/).pop() || "instant_reference_lora.safetensors";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    showToast("error", "Instant Reference LoRA", error.message);
  }
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

  node.addWidget("button", "Download LoRA", null, async () => {
    await downloadLora(node);
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
