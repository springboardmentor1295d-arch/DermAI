const BASE = "/api";
export const session = {
  get token() { return localStorage.getItem("token"); },
  get role() { return localStorage.getItem("role"); },
  set(t, role, name) { localStorage.setItem("token", t); localStorage.setItem("role", role); localStorage.setItem("name", name || ""); },
  clear() { localStorage.clear(); },
};
async function req(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (session.token) headers.Authorization = `Bearer ${session.token}`;
  if (opts.json) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
  const r = await fetch(BASE + path, { ...opts, headers });
  if (r.status === 401) { session.clear(); location.reload(); }
  if (!r.ok) {
    let d = await r.text(); try { const j = JSON.parse(d); d = typeof j.detail === "string" ? j.detail : j.detail?.message || JSON.stringify(j.detail); } catch {}
    throw new Error(d || r.statusText);
  }
  return opts.blob ? r.blob() : r.json();
}
export const api = {
  login: (email, password) => req("/auth/login", { method: "POST", body: new URLSearchParams({ username: email, password }) }),
  register: (b) => req("/auth/register", { method: "POST", json: b }),
  cases: () => req("/cases"), case: (id) => req(`/cases/${id}`),
  createCase: (b) => req("/cases", { method: "POST", json: b }),
  saveNotes: (id, t) => req(`/cases/${id}/notes`, { method: "PATCH", json: { clinical_notes: t } }),
  quality: (file) => { const f = new FormData(); f.append("file", file); return req("/quality/check", { method: "POST", body: f }); },
  upload: (id, file, source, allowPoor) => { const f = new FormData(); f.append("file", file); f.append("source", source); f.append("allow_poor", allowPoor); return req(`/cases/${id}/images`, { method: "POST", body: f }); },
  analyze: (imgId, force) => req(`/images/${imgId}/analyze?force=${!!force}`, { method: "POST" }),
  researchAnalyze: (imgId, metadata) => req(`/images/${imgId}/research-analyze`, { method: "POST", json: metadata }),
  analyses: (q = "") => req(`/analyses${q}`), compare: (a, b) => req(`/analyses/${a}/compare/${b}`),
  notifications: () => req("/notifications"),
  statsUser: () => req("/stats/user"), statsClin: () => req("/stats/clinician"), statsAdmin: () => req("/stats/admin"),
  models: () => req("/admin/models"), activate: (id) => req(`/admin/models/${id}/activate`, { method: "POST" }),
  report: async (id, fmt) => {
    const blob = await req(`/reports/${id}.${fmt}`, { blob: true });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `dermai_${id}.${fmt}`; a.click();
  },
};
export const media = (u) => `${BASE}${u}`;
