
**FinIntel-Multimodal** is an enterprise-grade financial intelligence platform that fuses structured time-series quantitative forecasting with unstructured semantic Retrieval-Augmented Generation (RAG). Designed to eliminate generative AI mathematical hallucinations during corporate analysis, the system integrates robust data pipelines, dense vector stores, statistical time-series models, and LLM orchestration into a unified Streamlit web application.

---

## 🏗️ System Architecture & Modular Breakdown

The project is engineered following a clean, modular multi-track architecture, where each script handles a distinct phase of the data intelligence lifecycle:

| Module File | Component Track | Primary Tech Stack | Core Responsibility |
| :--- | :--- | :--- | :--- |
| **`financial_data_pipepline.py`** | **Data Engineering Pipeline** | `Pandas`, `NumPy`, `KaggleHub` | Ingests raw financial CSVs from Kaggle or local paths, standardizes headers into uniform `snake_case`, handles group-based missing value imputation (`ffill`/`bfill`), and converts rows into semantic text chunks. |
| **`vector_indexing.py`** | **Vector Store & Indexing** | `ChromaDB`, `Sentence-Transformers` | Encodes textual corporate profiles into dense high-dimensional vectors (`all-MiniLM-L6-v2`), manages persistent local collection storage, and enforces strict company metadata isolation filters. |
| **`analytics.py`** | **Analytics & Forecasting** | `Prophet`, `Plotly`, `Scikit-Learn` | Extracts time-series vectors, converts annual entries into explicit datetime indices, models longitudinal business growth via Facebook Prophet, and renders interactive predictive graphics. |
| **`rag.py`** | **GenAI RAG Orchestrator** | `OpenAI API` | Fuses historical vector text chunks with statistical forecast projections into structured prompts, enforcing low-temperature bounds (`0.2`) and strict guardrails to prevent hallucinations. |
| **`streamlit_visualization_rewritten.py`** | **MLOps & Full-Stack UI** | `Streamlit`, `Python OS` | Orchestrates end-to-end execution flow inside an interactive web UI featuring a sidebar configuration panel, dynamic dataset caching, and a split-screen analytics viewport. |

---

## 🚀 Key Features

* **Zero-Hallucination RAG Guardrails:** Combines rigid statistical forecasting metrics directly with semantic vector lookups to ground LLM responses in factual corporate data.
* **Tenant Data Privacy Isolation:** Employs strict metadata filter locks (`where={"company": target}`) during ChromaDB vector queries to eliminate cross-company data leakage.
* **Sparse Data Time-Series Convergence:** Utilizes Facebook Prophet tuned specifically for multi-year corporate reporting records.
* **Interactive Split-Screen Dashboards:** Instantly contrasts generative AI textual executive summaries directly against reactive Plotly quantitative forecast trajectories.
* **Dynamic Dataset Caching:** Utilizes Streamlit's `@st.cache_data` and `@st.cache_resource` decorators for high-performance session handling.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/finintel-multimodal.git
cd finintel-multimodal
```

### 2. Install Dependencies
Ensure you have Python 3.9 or higher installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables (Optional)
If you are using the OpenAI API cloud gateway, set your API key as an environment variable:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```
*(Note: You can also input your API key directly into the secure sidebar configuration panel within the Streamlit UI).*

---

## 🏃‍♂️ Running the Application

Launch the unified Streamlit application interface:
```bash
streamlit run streamlit_visualization_rewritten.py
```

### Usage Workflow:
1. **Upload Dataset:** Upload a corporate financial statement CSV file via the sidebar file uploader.
2. **Select Parameters:** Choose your target enterprise company profile and the financial metric variable you wish to track (e.g., `revenue`, `net_income`).
3. **Ask a Question:** Input custom analytical queries (e.g., *"Summarize revenue trends, profitability, and financial outlook."*).
4. **Execute Evaluation:** Click **"Run Financial Analysis"** to render the AI synthesis and quantitative forecast chart side-by-side.

---

## 📂 Code Verification Gates

Each core script includes a verification execution block (`if __name__ == "__main__":`) equipped with mock data schemas. You can test individual modules independently:

```bash
python financial_data_pipepline.py  # Tests data ingestion & chunk generation
python vector_indexing.py           # Tests vector embedding & semantic querying
python analytics.py                 # Tests Prophet time-series convergence & plotting
python rag.py                       # Tests RAG prompt structuring & LLM synthesis
```

---

## 📜 License

Distributed under the **Apache License 2.0**. See `financial_statements.csv` or project documentation for details.ading github_repository_readme.md…]()
