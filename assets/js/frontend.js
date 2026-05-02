import React, { useState, useEffect } from "react";

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [meta, setMeta] = useState(null);
  const [form, setForm] = useState({
    area: "", crop: "", rainfall: 1485, temperature: 16.37, pesticides: 121
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);

  // 1. Load Dropdowns and Metrics from Backend
  useEffect(() => {
    fetch(`${API_BASE}/metadata`)
      .then(res => res.json())
      .then(data => {
        setMeta(data);
        // Set default values for dropdowns
        setForm(f => ({ ...f, area: data.areas[0], crop: data.items[0] }));
        setLoading(false);
      })
      .catch(err => console.error("Is the Python backend running?", err));
  }, []);

  const handleChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

  const handlePredict = async () => {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form)
    });
    const data = await res.json();
    setResult(data);
  };

  if (loading) return <div style={{padding: 50}}>Connecting to ML Model...</div>;

  return (
    <div style={{ maxWidth: "800px", margin: "40px auto", fontFamily: "Georgia, serif", padding: "20px", border: "1px solid #2d6a4f22", borderRadius: "20px", backgroundColor: "#fffcf5" }}>
      <h1 style={{ color: "#1d3124" }}>See likely yield before you plant.</h1>
      
      {/* Metrics Section (This is what was showing as __R2__ before) */}
      <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
        <div style={s.statCard}><span>R²</span><strong>{meta.metrics.r2.toFixed(4)}</strong></div>
        <div style={s.statCard}><span>RMSE</span><strong>{meta.metrics.rmse.toFixed(2)}</strong></div>
        <div style={s.statCard}><span>MAE</span><strong>{meta.metrics.mae.toFixed(2)}</strong></div>
      </div>
      <p style={{fontSize: "0.8rem", color: "#5f6f61"}}>
        Trained on {meta.metrics.train_rows} rows | Tested on {meta.metrics.test_rows} rows (years > {meta.metrics.split_year})
      </p>

      {/* Form Section */}
      <div style={{ display: "grid", gap: "15px", marginTop: "20px" }}>
        <div style={{display: "flex", gap: "10px"}}>
          <div style={{flex: 1}}>
            <label>Area</label>
            <select name="area" value={form.area} onChange={handleChange} style={s.input}>
              {meta.areas.map(a => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>
          <div style={{flex: 1}}>
            <label>Crop</label>
            <select name="crop" value={form.crop} onChange={handleChange} style={s.input}>
              {meta.items.map(i => <option key={i} value={i}>{i}</option>)}
            </select>
          </div>
        </div>

        <div style={{ display: "flex", gap: "10px" }}>
          <label style={{flex: 1}}>Rainfall (mm)
            <input name="rainfall" type="number" value={form.rainfall} onChange={handleChange} style={s.input} />
          </label>
          <label style={{flex: 1}}>Temp (°C)
            <input name="temperature" type="number" value={form.temperature} onChange={handleChange} style={s.input} />
          </label>
        </div>

        <label>Pesticides (tonnes)
          <input name="pesticides" type="number" value={form.pesticides} onChange={handleChange} style={s.input} />
        </label>

        <button onClick={handlePredict} style={s.button}>Predict Yield</button>
      </div>

      {result && (
        <div style={s.resultBox}>
          <h3 style={{margin: 0, fontSize: "1rem", color: "#5f6f61"}}>Predicted Yield</h3>
          <div style={{fontSize: "2.5rem", fontWeight: "bold", color: "#1d3124"}}>{result.predicted_yield} <span style={{fontSize: "1rem"}}>hg/ha</span></div>
          <p style={{fontSize: "0.8rem", marginTop: "10px"}}>Model used reference year {result.reference_year}.</p>
        </div>
      )}
    </div>
  );
}

const s = {
  statCard: { flex: 1, padding: "15px", background: "#fff", border: "1px solid #2d6a4f1a", borderRadius: "12px", textAlign: "center" },
  input: { display: "block", width: "100%", padding: "12px", marginTop: "5px", borderRadius: "10px", border: "1px solid #ddd", fontSize: "1rem" },
  button: { padding: "15px", background: "#2d6a4f", color: "white", border: "none", borderRadius: "30px", cursor: "pointer", fontWeight: "bold", fontSize: "1rem", marginTop: "10px" },
  resultBox: { marginTop: "30px", padding: "20px", backgroundColor: "rgba(45, 106, 79, 0.05)", borderRadius: "15px", border: "1px dashed #2d6a4f44" }
};