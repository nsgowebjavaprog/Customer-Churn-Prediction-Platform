/**
 * api.js
 * ------
 * Thin wrapper around fetch() for talking to the FastAPI backend.
 * Base URL comes from an env var so the same build works locally
 * and inside Docker (see docker-compose.yml -> REACT_APP_API_BASE_URL).
 */

const BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

async function handleResponse(res) {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
    } catch (_) {
      /* ignore parse errors */
    }
    throw new Error(detail);
  }
  return res;
}

export async function predictSingle(payload) {
  const res = await fetch(`${BASE_URL}/predict/?save_to_history=true`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await handleResponse(res);
  return res.json();
}

export async function validateCsv(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE_URL}/upload/validate-csv`, {
    method: "POST",
    body: formData,
  });
  return res.json(); // this endpoint always returns 200 with a `valid` flag
}

export async function predictCsv(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE_URL}/upload/predict-csv`, {
    method: "POST",
    body: formData,
  });
  await handleResponse(res);
  return res.blob(); // CSV file blob, ready to download
}

export async function fetchHistory(page = 1, pageSize = 10) {
  const res = await fetch(`${BASE_URL}/history/?page=${page}&page_size=${pageSize}`);
  await handleResponse(res);
  return res.json();
}

export async function deleteHistoryItem(id) {
  const res = await fetch(`${BASE_URL}/history/${id}`, { method: "DELETE" });
  await handleResponse(res);
}

export async function fetchStats() {
  const res = await fetch(`${BASE_URL}/history/stats/summary`);
  await handleResponse(res);
  return res.json();
}
