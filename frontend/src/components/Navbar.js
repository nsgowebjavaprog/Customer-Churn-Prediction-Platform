import React from "react";

export default function Navbar({ activeTab, onTabChange }) {
  const tabs = [
    { id: "single", label: "Single Prediction" },
    { id: "upload", label: "Batch CSV Upload" },
    { id: "history", label: "Prediction History" },
  ];

  return (
    <header className="navbar">
      <div className="brand">
        <span className="brand-mark">CG</span>
        <div>
          <div className="brand-name">ChurnGuard</div>
          <div className="brand-sub">Customer Churn Prediction Platform</div>
        </div>
      </div>
      <nav className="tabs">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={`tab-btn ${activeTab === t.id ? "active" : ""}`}
            onClick={() => onTabChange(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
