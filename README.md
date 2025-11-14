
# 🧩 Project README — ArchViz: LLM based Architectural Evolution Analyzer

## 📘 Overview

**ArchViz** analyzes the **architectural evolution** of Python repositories using static-analysis metrics and LLM-driven insights.
It extracts release history, builds dependency graphs, tracks structural drift, visualizes results, and now generates AI-based summaries.

✅ **Completion Stages:** Stages 1 – 6

* **Stage 1 – Scope & Success Criteria** (*Project_Scope.docx*)
* **Stage 2 – Version History Extraction** (tags / commits / changed files)
* **Stage 3 – Dependency Graphs & Metrics** (imports, fan-in/out, complexity)
* **Stage 4 – Evolution Differencing** (added / removed modules & edges)
* **Stage 5 – Visualization MVP** (interactive Streamlit dashboard)
* **Stage 6 – LLM Insights & Summaries** (LLM-based analysis & PDF reports)

---

## ⚙️ Environment Setup

```bash
# 1. Clone (skip if already local)
git clone <your-project-repo-url>
cd <your-project-folder>

# 2. Create and activate a virtual environment
python -m venv environment
source environment/bin/activate      # Windows: environment\Scripts\activate

# 3. Install dependencies
pip install -U pip
pip install pandas pydriller GitPython tqdm pyarrow networkx radon streamlit plotly pyvis openai python-dotenv reportlab
```

---

## 🔑 Before Running — Set Up Your OpenAI API Key

1. Create an API key at **[https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)**
2. Add it as an variable in the app.py python file (recommended)

   * **Stage5_app/app.py**

     ```powershell
     OPENAI_API_KEY "sk-yourkeyhere"
     ```

3. ⚠️ **Usage note:** Each run uses API credits. Ensure your OpenAI account has funds or a subscription before using the Insights tab.

---

## 📂 Folder Structure

```
project-root/
├── external/fastapi/                # Target repo clone
├── data/fastapi/
│   ├── curated/                     # CSV & Parquet outputs + insights
│   ├── logs/                        # JSON run logs
│   └── raw/                         # Stage 2 metadata
├── scripts/
│   ├── stage2_extract.py
│   ├── stage3_build_graphs.py
│   ├── stage4_diff.py
│   └── load_dataset.py
├── stage5_app/
│   ├── app.py                       # Streamlit dashboard
│   ├── load_data.py                 # Cached loaders
│   ├── viz_utils.py                 # Graph helpers
└── README.md
```

---

## 🏗️ Stage 2 — Version History Extraction

Extract tags, commits, and changed files using **PyDriller** → CSVs.

```bash
git clone https://github.com/fastapi/fastapi.git external/fastapi
python scripts/stage2_extract.py
```

**Outputs**

```
data/fastapi/curated/{versions,commits,files_changed}.csv
data/fastapi/raw/run_metadata.json
```

---

## 🧠 Stage 3 — Dependency Graphs & Metrics

Parses imports per tag, builds module graphs, and computes metrics.

```bash
python scripts/stage3_build_graphs.py
```

**Outputs**

```
data/fastapi/curated/{modules,edges,metrics}.csv
data/fastapi/logs/stage3_run.json
```

---

## 🔍 Stage 4 — Evolution Differencing

Compares successive versions → detects added/removed modules & edges, metric deltas, and drift.

```bash
python scripts/stage4_diff.py
```

**Outputs**

```
data/fastapi/curated/
  ├── changes_modules.csv
  ├── changes_edges.csv
  ├── changes_metrics.csv
  └── drift_summary.csv
```

---

## 💡 Stage 5 — Interactive Visualization (App)

Explore architecture and evolution via a **Streamlit dashboard**.

```bash
cd stage5_app
streamlit run app.py
```

**Features**

* Tag selector + diff view
* Interactive PyVis dependency graph
* Metrics tables & histograms
* Drift summary and metric deltas
* Auto-selected “most changed” release pair

---

## 🤖 Stage 6 — AI Insights & Summaries

**Purpose:** Leverage LLMs to create human-readable analytical reports.

**Access:** Tab 4 (Insights) inside the Streamlit app.

**Provides**

1. **Non-technical overview** (plain-language summary, 6–7 lines)
2. **Technical architecture summary** (metrics and structure insight, 6–7 lines)
3. **Evolution insights** (per-pair LLM bullets + charts)
4. **Comparative context & maintainability** (benchmark vs common frameworks)
5. **Visualizations:** growth over time, fan-in hotspots, complexity trends, and release churn
6. **📄 Export as PDF** button to save AI report for presentations or papers

**Runs with:** OpenAI GPT-4o-mini (default) or any model set via `OPENAI_MODEL`.

---

## ✅ Validation Checklist

| Check               | Expected Result                           |
| ------------------- | ----------------------------------------- |
| `versions.csv`      | 1 row per tag                             |
| `edges.csv`         | Internal imports only                     |
| `metrics.csv`       | Non-negative metrics                      |
| `drift_summary.csv` | Valid per-pair counts                     |
| Streamlit App       | All tabs load without error               |
| Insights Tab        | Shows LLM summaries + charts + PDF export |

---

## 🚀 Future Extensions

* Auto-generate `insights.csv` for auditing LLM outputs.
* Optional Docker packaging (Stage 7, future).
* Support for local LLMs (Ollama / LM Studio) to reduce API costs.

---

> **Tip 💡** The Insights tab uses your OpenAI credits per request (typically a few cents each).
> Re-runs are cached to avoid repeat charges.
