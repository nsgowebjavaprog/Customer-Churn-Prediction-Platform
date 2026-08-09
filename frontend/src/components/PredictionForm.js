import React, { useState } from "react";
import { predictSingle } from "../api";

const DEFAULTS = {
  gender: "Female",
  senior_citizen: 0,
  partner: "Yes",
  dependents: "No",
  tenure_months: 5,
  contract: "Month-to-month",
  internet_service: "Fiber optic",
  online_security: "No",
  tech_support: "No",
  streaming_tv: "Yes",
  paperless_billing: "Yes",
  payment_method: "Electronic check",
  monthly_charges: 89.5,
  total_charges: 450.0,
  num_support_calls: 3,
};

const SELECT_FIELDS = {
  gender: ["Male", "Female"],
  senior_citizen: [0, 1],
  partner: ["Yes", "No"],
  dependents: ["Yes", "No"],
  contract: ["Month-to-month", "One year", "Two year"],
  internet_service: ["DSL", "Fiber optic", "No"],
  online_security: ["Yes", "No", "No internet service"],
  tech_support: ["Yes", "No", "No internet service"],
  streaming_tv: ["Yes", "No", "No internet service"],
  paperless_billing: ["Yes", "No"],
  payment_method: ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
};

const NUMBER_FIELDS = ["tenure_months", "monthly_charges", "total_charges", "num_support_calls"];

export default function PredictionForm() {
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = { ...form };
      NUMBER_FIELDS.forEach((f) => (payload[f] = Number(payload[f])));
      payload.senior_citizen = Number(payload.senior_citizen);
      const res = await predictSingle(payload);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const riskColor = { Low: "#3ddc97", Medium: "#f5b942", High: "#ff5c5c" };

  return (
    <div className="card">
      <h2>Predict Churn for a Single Customer</h2>
      <p className="muted">
        Fill in the customer profile. This calls <code>POST /predict/</code> and
        saves the result to history automatically.
      </p>

      <form onSubmit={handleSubmit} className="form-grid">
        {Object.keys(DEFAULTS).map((field) => (
          <div className="form-field" key={field}>
            <label>{field.replace(/_/g, " ")}</label>
            {SELECT_FIELDS[field] ? (
              <select
                value={form[field]}
                onChange={(e) => handleChange(field, e.target.value)}
              >
                {SELECT_FIELDS[field].map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                step="any"
                value={form[field]}
                onChange={(e) => handleChange(field, e.target.value)}
              />
            )}
          </div>
        ))}

        <button type="submit" className="primary-btn" disabled={loading}>
          {loading ? "Predicting..." : "Predict Churn"}
        </button>
      </form>

      {error && <div className="alert error">{error}</div>}

      {result && (
        <div className="result-panel">
          <div className="result-row">
            <span>Prediction</span>
            <strong>{result.churn_prediction}</strong>
          </div>
          <div className="result-row">
            <span>Probability</span>
            <strong>{(result.churn_probability * 100).toFixed(1)}%</strong>
          </div>
          <div className="result-row">
            <span>Risk Level</span>
            <strong style={{ color: riskColor[result.risk_level] }}>
              {result.risk_level}
            </strong>
          </div>
          <div className="result-row">
            <span>Model Used</span>
            <strong>{result.model_used}</strong>
          </div>
        </div>
      )}
    </div>
  );
}
