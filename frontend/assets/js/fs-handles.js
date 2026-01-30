(() => {
  const DB_NAME = "aiDartScorer";
  const STORE_NAME = "handles";
  const DB_VERSION = 1;
  const HANDLE_KEY = "dataCollectionFolder";

  const openDb = () =>
    new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);

      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME);
        }
      };

      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });

  const withStore = async (mode, fn) => {
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
    withStore("readwrite", (store) => store.put(handle, HANDLE_KEY));

  const loadDirectoryHandle = () =>
    withStore("readonly", (store) => store.get(HANDLE_KEY));

  const clearDirectoryHandle = () =>
    withStore("readwrite", (store) => store.delete(HANDLE_KEY));

  const ensurePermission = async (handle, mode = "readwrite") => {
    if (!handle) return false;
    const opts = { mode };
    const current = await handle.queryPermission(opts);
    if (current === "granted") return true;
    return (await handle.requestPermission(opts)) === "granted";
  };

  window.aiDartScorerFS = {
    saveDirectoryHandle,
    loadDirectoryHandle,
    clearDirectoryHandle,
    ensurePermission,
  };
})();
