import React, { useEffect, useRef, useState } from "react";
import { ArcElement, BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, Tooltip } from "chart.js";
import { Bar, Doughnut } from "react-chartjs-2";
import { api, media, session } from "./api.js";
ChartJS.register(ArcElement, BarElement, CategoryScale, LinearScale, Tooltip, Legend);

const DISCLAIMER = "AI-assisted analysis for screening/research and clinical decision support. Not a medical diagnosis.";
const riskClass = (r = "") => r.split(" ")[0];
const Disclaimer = () => <div className="banner">⚠ {DISCLAIMER}</div>;


function Auth({ onDone }) {
  const [mode, setMode] = useState("login");
  const [f, setF] = useState({
    email: "",
    password: "",
    full_name: "",
    role: "patient"
  });
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const set = k => e =>
    setF(old => ({ ...old, [k]: e.target.value }));

  const go = async e => {
    e.preventDefault();
    setErr("");
    setBusy(true);

    try {
      if (mode === "register")
        await api.register(f);

      const r = await api.login(f.email, f.password);
      session.set(r.access_token, r.role, r.name);
      onDone();
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  return <main className="auth-layout">

    <section className="auth-hero">

      <div className="brand">
        <div className="brand-logo">+</div>
        <span>DermAI</span>
      </div>

      <div className="eyebrow">
        AI-POWERED DERMATOLOGY
      </div>

      <h1>
        Advanced AI for
        <br />
        <span>Better Skin Insights.</span>
      </h1>

      <p>
        An intelligent skin image analysis platform
        combining deep learning, clinical information
        and explainable artificial intelligence.
      </p>

      <div className="feature-list">

        <div className="feature">
          <span className="feature-icon">01</span>
          <div>
            <b>Intelligent Skin Analysis</b>
            <p>AI-assisted skin lesion classification.</p>
          </div>
        </div>

        <div className="feature">
          <span className="feature-icon">02</span>
          <div>
            <b>Clinical Metadata Integration</b>
            <p>Combining image and clinical features.</p>
          </div>
        </div>

        <div className="feature">
          <span className="feature-icon">03</span>
          <div>
            <b>Explainable AI</b>
            <p>Visual explanations and prediction insights.</p>
          </div>
        </div>

        <div className="feature">
          <span className="feature-icon">04</span>
          <div>
            <b>Structured Analysis</b>
            <p>Organized case records and analysis results.</p>
          </div>
        </div>

      </div>

      <small className="mut">
        AI-assisted research platform.
        Not a medical diagnosis.
      </small>

    </section>

    <div className="card auth-card">

      <div className="login-logo">+</div>

      <h2>
        {mode === "login"
          ? "Welcome back"
          : "Create your account"}
      </h2>

      <p className="mut">
        {mode === "login"
          ? "Sign in to continue to DermAI"
          : "Get started with DermAI"}
      </p>

      <form onSubmit={go}>

        {mode === "register" && <>

          <label>Full name</label>
          <input
            placeholder="Enter your full name"
            value={f.full_name}
            onChange={set("full_name")}
            required
          />

          <label>Account type</label>
          <select value={f.role} onChange={set("role")}>
            <option value="patient">Patient / General user</option>
            <option value="clinician">Dermatologist / Clinician</option>
            <option value="researcher">Medical researcher</option>
          </select>

        </>}

        <label>Email address</label>

        <input
          type="email"
          placeholder="Enter your email"
          value={f.email}
          onChange={set("email")}
          autoComplete="email"
          required
        />

        <label>Password</label>

        <div className="password-field">

          <input
            type={showPassword ? "text" : "password"}
            placeholder="Enter your password"
            value={f.password}
            onChange={set("password")}
            autoComplete={mode === "login"
              ? "current-password"
              : "new-password"}
            required
          />

          <button
            type="button"
            className="password-toggle"
            onClick={() => setShowPassword(!showPassword)}
          >
            {showPassword ? "Hide" : "Show"}
          </button>

        </div>

        {err && <p className="err">{err}</p>}

        <button className="auth-submit" disabled={busy}>
          {busy
            ? "Please wait..."
            : mode === "login"
            ? "Sign in"
            : "Create account"}
        </button>

      </form>

      <div className="auth-switch">

        {mode === "login"
          ? "New to DermAI?"
          : "Already have an account?"}

        <button
          type="button"
          className="text-button"
          onClick={() => {
            setMode(mode === "login" ? "register" : "login");
            setErr("");
          }}
        >
          {mode === "login" ? "Create account" : "Sign in"}
        </button>

      </div>

      <div className="login-footer">
        DermAI | AI-assisted Skin Image Analysis
      </div>

    </div>

  </main>;
}

function Dashboard() {
  const role = session.role; const [d, setD] = useState(null); const [u, setU] = useState(null);
  useEffect(() => { api.statsUser().then(setU).catch(() => {});
    if (["clinician", "admin", "researcher"].includes(role)) api.statsClin().then(setD).catch(() => {});
    if (["admin", "ml_engineer"].includes(role)) api.statsAdmin().then(s => setD(p => ({ ...p, admin: s }))).catch(() => {}); }, []);
  const dist = d?.disease_distribution || {};
  return <main><h2>Dashboard</h2>
    {u && <div className="card"><h3>My recent analyses</h3><div className="kpi">{u.total}</div><table><tbody>{u.recent.map(a => <tr key={a.id}><td>#{a.id}</td><td>{a.name}</td><td>{(a.confidence * 100).toFixed(0)}%</td><td className={riskClass(a.risk)}>{a.risk}</td></tr>)}</tbody></table></div>}
    {d?.total_cases !== undefined && <div className="grid"><div className="card"><div className="kpi">{d.total_cases}</div>Total cases</div><div className="card"><div className="kpi">{d.needing_review}</div>Cases requiring review</div>
      <div className="card"><div className="kpi">{d.avg_confidence ? (d.avg_confidence * 100).toFixed(0) + "%" : "-"}</div>Avg confidence</div></div>}
    {d?.total_cases !== undefined && <div className="grid">
      <div className="card"><h3>Disease distribution</h3><Doughnut data={{ labels: Object.keys(dist), datasets: [{ data: Object.values(dist), backgroundColor: ["#0f766e", "#b91c1c", "#b45309", "#15803d", "#6d28d9", "#0369a1"] }] }} /></div>
      <div className="card"><h3>Prediction confidence</h3><Bar data={{ labels: d.confidence_hist.map((_, i) => `${i * 10}-${i * 10 + 10}%`), datasets: [{ label: "Analyses", data: d.confidence_hist, backgroundColor: "#0f766e" }] }} /></div>
      <div className="card"><h3>Model performance</h3>{d.active_model ? <><b>{d.active_model.name} v{d.active_model.version}</b>{Object.entries(d.active_model.metrics || {}).map(([k, v]) => <div key={k}>{k}: {typeof v === "number" ? v.toFixed(3) : v}</div>)}</> : "No active model"}</div></div>}
    {d?.admin && <div className="card"><h3>Administration</h3>Users: {d.admin.users} · Analyses: {d.admin.analyses} · Health: {d.admin.platform_health}
      <h4>Model usage</h4>{d.admin.model_usage.map((m, i) => <div key={i}>{m.name} v{m.version}: {m.count}</div>)}
      <h4>Recent activity</h4><table><tbody>{d.admin.recent_activity.map((l, i) => <tr key={i}><td>{String(l.at).slice(0, 19)}</td><td>{l.action}</td><td>{l.detail}</td></tr>)}</tbody></table></div>}
  </main>;
}

function Result({ a }) {
  return <div className="card"><h3>{a.predicted_name} <span className={`tag ${riskClass(a.risk)}`}>{a.risk}</span></h3>
    <div className="mut">Confidence {(a.confidence * 100).toFixed(1)}% · Quality {a.quality_label} · Model {a.model} · {new Date(a.created_at).toLocaleString()}</div>
    {a.needs_review && <p className="err">Low confidence or poor image: clinical review recommended.</p>}
    <div className="grid" style={{ margin: "12px 0" }}>{[["Original", a.original_url], ["Preprocessed (ROI)", a.preprocessed_url], [`Lesion mask (${a.seg_method})`, a.mask_url], [`Explanation (${a.explain_method})`, a.cam_url]].map(([t, u]) =>
      <figure key={t} style={{ margin: 0 }}><img src={media(u)} /><figcaption className="mut">{t}</figcaption></figure>)}</div>
    <b>Top predictions</b>{a.top3.map(([c, p]) => <div key={c}>{c} {(p * 100).toFixed(1)}%<div className="bar"><div style={{ width: p * 100 + "%" }} /></div></div>)}
    <p className="mut">Explanation maps show influential regions; they are not proof of clinical correctness.</p>
    {["pdf", "json", "csv"].map(f => <button key={f} className="ghost" onClick={() => api.report(a.id, f)} style={{ marginRight: 6 }}>Report .{f}</button>)}
  </div>;
}


const FUSION_NUMERIC = [
  ["age", "Age", 0, 120],
  ["fitspatrick", "Fitzpatrick skin type", 1, 6],
  ["diameter_1", "Lesion diameter 1 (mm)", 0.1, undefined],
  ["diameter_2", "Lesion diameter 2 (mm)", 0.1, undefined]
];

const FUSION_BOOLEAN = [
  ["smoke", "Smoking history"],
  ["drink", "Alcohol consumption"],
  ["skin_cancer_history", "Skin cancer history"],
  ["cancer_history", "Cancer history"],
  ["itch", "Itching"],
  ["grew", "Lesion growth"],
  ["hurt", "Pain"],
  ["changed", "Lesion changes"],
  ["bleed", "Bleeding"],
  ["elevation", "Elevated lesion"]
];

function FusionForm({ metadata, setMetadata }) {
  const change = k => e => setMetadata(
    old => ({ ...old, [k]: e.target.value })
  );

  return <div className="card">
    <h3>Clinical Metadata</h3>
    <p className="mut">
      Enter verified information for research analysis.
      Do not guess missing clinical measurements.
    </p>

    <div className="grid">
      {FUSION_NUMERIC.map(([k, label, min, max]) =>
        <label key={k}>{label}
          <input type="number" min={min} max={max}
            step={k.startsWith("diameter") ? "any" : 1}
            value={metadata[k] ?? ""}
            onChange={change(k)} required />
        </label>
      )}

      <label>Gender
        <select value={metadata.gender ?? ""}
          onChange={change("gender")}>
          <option value="">Select</option>
          <option value="MALE">Male</option>
          <option value="FEMALE">Female</option>
          <option value="UNK">Unknown</option>
        </select>
      </label>

      <label>Body region
        <select value={metadata.region ?? ""}
          onChange={change("region")}>
          <option value="">Select region</option>
          {["ABDOMEN","ARM","BACK","CHEST","EAR",
            "FACE","FOOT","FOREARM","HAND","LIP",
            "NECK","NOSE","SCALP","THIGH"].map(v =>
            <option key={v} value={v}>{v}</option>
          )}
        </select>
      </label>

      {FUSION_BOOLEAN.map(([k, label]) =>
        <label key={k}>{label}
          <select value={metadata[k] ?? ""}
            onChange={change(k)}>
            <option value="">Select</option>
            <option value="True">Yes</option>
            <option value="False">No</option>
            <option value="UNK">Unknown</option>
          </select>
        </label>
      )}
    </div>
  </div>;
}

function FusionResult({ a }) {
  return <div className="card">
    <h2>EfficientNet-B0 Fusion V2</h2>
    <h3>AI-Assisted Diagnosis (Predicted Condition)</h3>
    <div className="card">
      <div className="kpi">{a.predicted_name}</div>
      <h3>Confidence: {(a.confidence * 100).toFixed(2)}%</h3>
      <small className="mut">AI prediction, not a confirmed medical diagnosis.</small>
    </div>

    <h3>Model Evaluation Metrics</h3>
    <p className="mut">
      Observed PAD-UFES-20 test dataset performance
    </p>

    <div className="grid">
      {[
        ["Accuracy", "accuracy"],
        ["Macro F1", "macro_f1"],
        ["Balanced Accuracy", "balanced_accuracy"],
        ["MCC", "mcc"],
        ["Macro AUC", "macro_auc_ovr"]
      ].map(([label, key]) =>
        <div className="card" key={key}>
          <div className="kpi">
            {key === "mcc"
              ? Number(a.metrics[key]).toFixed(4)
              : (a.metrics[key] * 100).toFixed(2) + "%"}
          </div>
          {label}
        </div>
      )}
    </div>

    
    

    <h3>Top 3 Predictions</h3>
    {a.top3.map(([c, v]) =>
      <div key={c}>
        {c}: {(v * 100).toFixed(2)}%
        <div className="bar">
          <div style={{width: v * 100 + "%"}} />
        </div>
      </div>
    )}

    <h3>Explainability: {a.explain_method}</h3>
    <img src={a.explanation_image}
      style={{maxWidth: 400, width: "100%"}}
      alt="Research model explanation" />

    <p className="mut">
      Confidence and explanation maps do not establish
      clinical correctness.
    </p>
  </div>;
}

function CaseView({ id, back }) {
  const [c, setC] = useState(null); const [file, setFile] = useState(null); const [q, setQ] = useState(null); const [src, setSrc] = useState("gallery");
  const [res, setRes] = useState(null); const [err, setErr] = useState(""); const [busy, setBusy] = useState(false); const [hist, setHist] = useState([]); const [cmp, setCmp] = useState(null); const [notes, setNotes] = useState("");
  const [research, setResearch] = useState(
    ["admin", "researcher", "ml_engineer"].includes(session.role)
  );
  const [metadata, setMetadata] = useState({});
  const [fusionResult, setFusionResult] = useState(null);

  const researchAccess = ["admin", "researcher", "ml_engineer"].includes(session.role);

  const complete =
    FUSION_NUMERIC.every(([k, label, min, max]) =>
      metadata[k] != null &&
      String(metadata[k]).trim() !== "" &&
      Number.isFinite(Number(metadata[k])) &&
      Number(metadata[k]) >= min &&
      (max === undefined || Number(metadata[k]) <= max) &&
      (k.startsWith("diameter") || Number.isInteger(Number(metadata[k])))
    ) &&
    !!metadata.gender && !!metadata.region &&
    FUSION_BOOLEAN.every(([k]) => !!metadata[k]);

  const load = () => { api.case(id).then(x => { setC(x); setNotes(x.clinical_notes || ""); }); api.analyses(`?case_id=${id}`).then(setHist); };
  useEffect(load, [id]);
  const pick = async (e) => { const f = e.target.files[0]; setFile(f); setQ(null); setErr(""); if (f) try { setQ(await api.quality(f)); } catch (x) { setErr(x.message); } };
  const run = async (override = false) => {
    setBusy(true); setErr("");
    try {
      if (research && (!complete || !researchAccess))
        throw new Error("Complete all clinical metadata fields.");

      if (research && q?.label === "Poor")
        throw new Error("Research mode requires a better-quality image.");

      const img = await api.upload(id, file, src, override);

      if (research) {
        const a = await api.researchAnalyze(img.id, metadata);
        setFusionResult(a);
        setRes(null);
      } else {
        const a = await api.analyze(img.id, override);
        setRes(a);
        setFusionResult(null);
      }

      setFile(null);
      setQ(null);
      load();
    }
    catch (e) { setErr(e.message); } setBusy(false);
  };
  if (!c) return null;
  return <main><button className="ghost" onClick={back}>← Cases</button><h2>Case #{c.id} {c.patient_code}</h2>
    <div className="card"><h3>New image</h3><div className="mut">Workflow: capture → quality check → confirm → upload → AI analysis</div>
      {researchAccess && <label>Analysis model
        <select value={research ? "fusion" : "mobile"}
          disabled={busy}
          onChange={e => {
            setResearch(e.target.value === "fusion");
            setFusionResult(null);
            setRes(null);
          }}>
          <option value="mobile">Standard image-only model</option>
          <option value="fusion">EfficientNet-B0 Fusion V2 - Research</option>
        </select>
      </label>}

      {research && <FusionForm metadata={metadata}
        setMetadata={setMetadata} />}

      <select value={src} onChange={e => setSrc(e.target.value)}><option value="camera">Camera</option><option value="gallery">Gallery / upload</option></select>
      <input type="file" accept="image/*" capture={src === "camera" ? "environment" : undefined} onChange={pick} />
      {file && <img src={URL.createObjectURL(file)} style={{ maxWidth: 260, borderRadius: 8 }} />}
      {q && <div><b>Quality: <span className={q.label === "Poor" ? "err" : ""}>{q.label}</span></b> <span className="mut">blur {q.blur_score} · brightness {q.brightness} · contrast {q.contrast} · {q.width}×{q.height}</span>{q.issues.map(i => <div key={i} className="err">• {i}</div>)}</div>}
      {err && <p className="err">{err}</p>}
      <button disabled={!file || busy || (research && (!complete || q?.label === "Poor"))} onClick={() => run(q?.label === "Poor")}>{busy ? "Analyzing…" : q?.label === "Poor" ? "Analyze anyway (flagged for review)" : "Confirm & analyze"}</button>
      {q?.label === "Poor" && <span className="mut"> Recapturing a sharper, well-lit image is recommended; the result will be flagged for clinical review.</span>}</div>
    {res && <Result a={res} />}
    {fusionResult && <FusionResult a={fusionResult} />}
    <div className="card"><h3>Clinical notes</h3><textarea rows={3} value={notes} onChange={e => setNotes(e.target.value)} /><button className="ghost" onClick={() => api.saveNotes(id, notes)}>Save notes</button></div>
    <div className="card"><h3>Analysis history & follow-up</h3><table><tbody>{hist.map(a => <tr key={a.id}><td>#{a.id}</td><td>{new Date(a.created_at).toLocaleDateString()}</td><td>{a.predicted_class} {(a.confidence * 100).toFixed(0)}%</td><td>{a.model}</td>
      <td><button className="ghost" onClick={() => setRes(a)}>View</button> {hist.find(x => x.id < a.id) && <button className="ghost" onClick={async () => setCmp(await api.compare(hist.filter(x => x.id < a.id)[0].id, a.id))}>vs previous</button>}</td></tr>)}</tbody></table>
      {cmp && <div className="grid" style={{ marginTop: 12 }}>{[cmp.previous, cmp.current].map((a, i) => <div key={i}><img src={media(a.original_url)} /><div>{i ? "Current" : "Previous"}: {a.predicted_class} {(a.confidence * 100).toFixed(0)}% · {a.risk}</div></div>)}
        <div>Class changed: {String(cmp.changes.class_changed)}<br />Confidence Δ: {(cmp.changes.confidence_delta * 100).toFixed(1)} pts<br />Lesion area Δ: {(cmp.changes.lesion_area_ratio_delta * 100).toFixed(1)} pts<br /><span className="mut">{cmp.changes.note}</span></div></div>}</div>
  </main>;
}

function Cases() {
  const [list, setList] = useState([]); const [open, setOpen] = useState(null); const [f, setF] = useState({ patient_code: "", age: "", gender: "", body_location: "" });
  const load = () => api.cases().then(setList); useEffect(() => { load(); }, []);
  if (open) return <CaseView id={open} back={() => { setOpen(null); load(); }} />;
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });
  return <main><h2>Cases</h2>
    <div className="card"><h3>New case</h3><div className="grid"><input placeholder="Case code (de-identified)" value={f.patient_code} onChange={set("patient_code")} /><input placeholder="Age" type="number" value={f.age} onChange={set("age")} />
      <input placeholder="Gender" value={f.gender} onChange={set("gender")} /><input placeholder="Body location" value={f.body_location} onChange={set("body_location")} /></div>
      <button onClick={async () => { const c = await api.createCase({ ...f, age: f.age ? +f.age : null }); setOpen(c.id); }}>Create case</button></div>
    <div className="card"><table><tbody>{list.map(c => <tr key={c.id}><td>#{c.id}</td><td>{c.patient_code}</td><td>{c.body_location}</td><td>{c.n_images} images</td><td><button className="ghost" onClick={() => setOpen(c.id)}>Open</button></td></tr>)}</tbody></table></div></main>;
}

function Models() {
  const [m, setM] = useState([]); const load = () => api.models().then(setM); useEffect(() => { load(); }, []);
  const canEdit = ["admin", "ml_engineer"].includes(session.role);
  return <main><h2>Model registry</h2><div className="card"><table><thead><tr><th>Name</th><th>Version</th><th>Arch</th><th>Macro-F1</th><th>Bal. acc</th><th></th></tr></thead><tbody>
    {m.map(x => <tr key={x.id}><td>{x.name}</td><td>{x.version}</td><td>{x.arch}</td><td>{x.metrics?.macro_f1?.toFixed(3)}</td><td>{x.metrics?.balanced_accuracy?.toFixed(3)}</td>
      <td>{x.metrics?.research_only ? <span className="tag">Research default</span> : x.is_active ? <span className="tag">active</span> : canEdit && <button className="ghost" onClick={async () => { await api.activate(x.id); load(); }}>Deploy / roll back</button>}</td></tr>)}</tbody></table></div></main>;
}

function Notifications() {
  const [n, setN] = useState([]); useEffect(() => { api.notifications().then(setN); }, []);
  return <main><h2>Notifications</h2><div className="card">{n.length === 0 && "Nothing yet."}{n.map(x => <div key={x.id} style={{ padding: "6px 0", borderBottom: "1px solid var(--bd)" }}><span className="tag">{x.level}</span> <b>{x.title}</b> — {x.message}<div className="mut">{String(x.created_at).slice(0, 19)}</div></div>)}</div></main>;
}

export default function App() {
  const [authed, setAuthed] = useState(!!session.token); const [page, setPage] = useState("dash");
  if (!authed) return <Auth onDone={() => setAuthed(true)} />;
  const P = { dash: <Dashboard />, cases: <Cases />, models: <Models />, notes: <Notifications /> }[page];
  return <><header><b>DermAI</b><nav>{[["dash", "Dashboard"], ["cases", "Cases"], ["models", "Models"], ["notes", "Notifications"]].map(([k, l]) => <button key={k} className={"ghost " + (page === k ? "on" : "")} onClick={() => setPage(k)}>{l}</button>)}</nav>
    <span className="mut">{session.role}</span><button className="ghost" onClick={() => { session.clear(); setAuthed(false); }}>Log out</button></header>{P}</>;
}
