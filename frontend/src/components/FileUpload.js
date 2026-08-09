import React, { useState } from "react";
import { validateCsv, predictCsv } from "../api";

export default function FileUpload() {
  const [file, setFile] = useState(null);
  const [validation, setValidation] = useState(null);
  const [validating, setValidating] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = async (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setValidation(null);
    setError(null);
    if (!selected) return;

    setValidating(true);
    try {
      const result = await validateCsv(selected);
      setValidation(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setValidating(false);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    setPredicting(true);
    setError(null);
    try {
      const blob = await predictCsv(file);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "churn_predictions.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div className="card">
      <h2>Batch CSV Upload</h2>
      <p className="muted">
        Upload a CSV of customers. We validate the file format first, then
        run predictions and give you back the same CSV with three extra
        columns: <code>churn_prediction</code>, <code>churn_probability</code>,{" "}
        <code>risk_level</code>.
      </p>

      <label className="dropzone">
        <input type="file" accept=".csv" onChange={handleFileChange} hidden />
        {file ? file.name : "Click to choose a .csv file"}
      </label>

      {validating && <p className="muted">Validating file format...</p>}

      {validation && (
        <div className={`alert ${validation.valid ? "success" : "error"}`}>
          {validation.valid ? (
            <>
              ✅ Format looks good — {validation.row_count} rows detected.
            </>
          ) : (
            <>
              ❌ Invalid format:
              <ul>
                {validation.errors.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}

      {error && <div className="alert error">{error}</div>}

      <button
        className="primary-btn"
        onClick={handlePredict}
        disabled={!validation?.valid || predicting}
      >
        {predicting ? "Running predictions..." : "Predict & Download CSV"}
      </button>

      <details className="hint">
        <summary>What columns does the CSV need?</summary>
        <code>
          gender, senior_citizen, partner, dependents, tenure_months, contract,
          internet_service, online_security, tech_support, streaming_tv,
          paperless_billing, payment_method, monthly_charges, total_charges,
          num_support_calls
        </code>
      </details>
    </div>
  );
}
