import React, { useState } from "react";
import "./App.css";
import Navbar from "./components/Navbar";
import PredictionForm from "./components/PredictionForm";
import FileUpload from "./components/FileUpload";
import ResultTable from "./components/ResultTable";

export default function App() {
  const [activeTab, setActiveTab] = useState("single");

  return (
    <div className="app-shell">
      <Navbar activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="main-content">
        {activeTab === "single" && <PredictionForm />}
        {activeTab === "upload" && <FileUpload />}
        {activeTab === "history" && <ResultTable />}
      </main>
      <footer className="footer">
        ChurnGuard — FastAPI + scikit-learn + MLflow + React + Docker
      </footer>
    </div>
  );
}
