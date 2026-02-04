(() => {
  const DB_NAME = "aiDartScorer";
  const STORE_NAME = "handles";
  const DB_VERSION = 1;
  const HANDLE_KEY = "dataCollectionFolder";

  const openDb = () =>
    new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);

      req.onupgradeneeded = () => {
        // Create a tiny IndexedDB store to persist File System Access handles.
        // Handles are origin-bound and must be stored in IndexedDB to be reusable across sessions.
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME);
        }
      };

      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });

  const withStore = async (mode, fn) => {
    // Small helper to run a single IDB request inside a transaction and return its result.
    const db = await openDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, mode);
      const store = tx.objectStore(STORE_NAME);
      let request;
      try {
        request = fn(store);
      } catch (err) {
        reject(err);
        return;
      }
      if (!request) {
        resolve(undefined);
        return;
      }
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  };

  const saveDirectoryHandle = (handle) =>
    // Persist the directory handle selected in Settings (used by Data Collection mode).
    withStore("readwrite", (store) => store.put(handle, HANDLE_KEY));

  const loadDirectoryHandle = () =>
    // Restore the directory handle on app start (permission may still need to be re-granted).
    withStore("readonly", (store) => store.get(HANDLE_KEY));

  const clearDirectoryHandle = () =>
    // Allows "reset" flows: force user to re-select a folder.
    withStore("readwrite", (store) => store.delete(HANDLE_KEY));

  const ensurePermission = async (handle, mode = "readwrite") => {
    // Browser permissions are per-handle and can be "prompt"/"denied"/"granted".
    // This helper upgrades "prompt" to "granted" by requesting permission when needed.
    if (!handle) return false;
    const opts = { mode };
    const current = await handle.queryPermission(opts);
    if (current === "granted") return true;
    return (await handle.requestPermission(opts)) === "granted";
  };

  // Global bridge used by settings.js + data-collection.js to avoid duplicating IDB logic.
  window.aiDartScorerFS = {
    saveDirectoryHandle,
    loadDirectoryHandle,
    clearDirectoryHandle,
    ensurePermission,
  };
})();
