# AI Data Analyst

A Streamlit app that lets you upload a CSV and ask questions about it in plain
English. Gemini plans which analysis to run (sum, average, group-by, filter,
top-N, etc.); Python (pandas) is the only thing that ever touches the actual
numbers, so the AI can't "hallucinate" a result — it can only choose and
explain a calculation that real code executed.

```
User question
      -> Gemini planner (chooses an operation + columns)
      -> Plan validated against the real dataset schema
      -> pandas executes the calculation
      -> Gemini explains the real result in plain English
      -> Answer + table/chart shown in the UI
```

## Setup

1. **Clone and enter the project**
   ```
   git clone <your-fork-url>
   cd AI_DATA_ANALYST
   ```

2. **Create a virtual environment** (recommended)
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Configure your Gemini API key**
   ```
   copy .env.example .env       # Windows
   cp .env.example .env         # macOS / Linux
   ```
   Then edit `.env` and set `GEMINI_API_KEY` to a key from
   [Google AI Studio](https://aistudio.google.com/apikey).

   > **Note:** the default model in `.env.example` / `analyst.py`
   > (`GEMINI_MODEL`, default `gemini-3.6-flash`) should be double-checked
   > against the current model list in Google AI Studio before you rely on
   > it — model names change over time. Override it in `.env` if needed.

5. **Run the app**
   ```
   streamlit run app.py
   ```
   Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Using it

1. Upload a CSV from the sidebar.
2. **📋 Raw Data** — sanity-check the first 50 rows loaded correctly.
3. **📈 Data Profile** — see column types, missing values, numeric summaries,
   duplicate rows, etc. (powered by `data_profile.py`).
4. **💬 Ask AI Analyst** — type a question (e.g. *"total revenue by city"*,
   *"top 5 products by units sold"*, *"average revenue where city is
   Delhi"*), click **Run Analysis**, and get an explanation, a table/chart,
   and the underlying execution plan (expandable) for transparency.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — upload, profile, ask, chart |
| `analyst.py` | Core pipeline: Gemini planning, plan validation, execution, explanation |
| `analysis_tools.py` | The actual pandas operations (sum, average, group-by, filters, top-N, ...) |
| `data_profile.py` | Dataset/column profiling used by the Profile tab and the planner |
| `ai_agent.py`, `query_engine.py`, `Validation.py` | Earlier prototypes, kept for reference/tests; not used by `app.py` |
| `test_*.py` | Pytest suite (`pytest -q` — 186 passed, 1 skipped as of this build) |

## Running tests

```
pytest -q
```

No API key is required to run the tests — Gemini calls are mocked/faked
throughout the suite.
