import React, { useEffect, useState, useCallback } from "react";
import { fetchHistory, deleteHistoryItem, fetchStats } from "../api";

export default function ResultTable() {
  const [data, setData] = useState({ items: [], total: 0, page: 1, page_size: 10 });
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async (page = 1) => {
    try {
      const [historyRes, statsRes] = await Promise.all([
        fetchHistory(page, 10),
        fetchStats(),
      ]);
      setData(historyRes);
      setStats(statsRes);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    load(1);
  }, [load]);

  const handleDelete = async (id) => {
    await deleteHistoryItem(id);
    load(data.page);
  };

  const totalPages = Math.max(1, Math.ceil(data.total / data.page_size));

  return (
    <div className="card">
      <h2>Prediction History</h2>
      <p className="muted">
        Full CRUD over past predictions — <code>GET / PATCH / DELETE
        /history/{"{id}"}</code>.
      </p>

      {stats && (
        <div className="stats-row">
          <div className="stat">
            <span>Total predictions</span>
            <strong>{stats.total_predictions}</strong>
          </div>
          <div className="stat">
            <span>Predicted churners</span>
            <strong>{stats.predicted_churn_count}</strong>
          </div>
          <div className="stat">
            <span>Predicted churn rate</span>
            <strong>{(stats.predicted_churn_rate * 100).toFixed(1)}%</strong>
          </div>
        </div>
      )}

      {error && <div className="alert error">{error}</div>}

      <table className="history-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Customer</th>
            <th>Prediction</th>
            <th>Probability</th>
            <th>Risk</th>
            <th>Model</th>
            <th>Created</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {data.items.map((item) => (
            <tr key={item.id}>
              <td>{item.id}</td>
              <td>{item.customer_id || "—"}</td>
              <td>{item.churn_prediction}</td>
              <td>{(item.churn_probability * 100).toFixed(1)}%</td>
              <td>{item.risk_level}</td>
              <td>{item.model_used}</td>
              <td>{new Date(item.created_at).toLocaleString()}</td>
              <td>
                <button className="link-btn danger" onClick={() => handleDelete(item.id)}>
                  Delete
                </button>
              </td>
            </tr>
          ))}
          {data.items.length === 0 && (
            <tr>
              <td colSpan={8} className="muted center">
                No predictions yet — try the Single Prediction tab.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      <div className="pagination">
        <button disabled={data.page <= 1} onClick={() => load(data.page - 1)}>
          Prev
        </button>
        <span>
          Page {data.page} of {totalPages}
        </span>
        <button disabled={data.page >= totalPages} onClick={() => load(data.page + 1)}>
          Next
        </button>
      </div>
    </div>
  );
}
