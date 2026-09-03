# Scopus Affiliation and Keyword Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FAIR](https://img.shields.io/badge/FAIR-compliant-green.svg)](https://www.go-fair.org/fair-principles/)
[![DOI](https://zenodo.org/badge/1356203841.svg)](https://doi.org/10.5281/zenodo.22284926)

A formal academic Streamlit dashboard for querying Scopus metadata, retrieving full text via the Elsevier Article Retrieval API (with automatic Scopus abstract fallback on paywalls or missing access), classifying author affiliations (Academia vs. Industry), and analyzing keyword prevalence across institutional structures and time.

---

## Table of Contents

- [Features](#features)
- [Architecture & Data Pipeline](#architecture--data-pipeline)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Data Export & Provenance](#data-export--provenance)
- [Project Structure](#project-structure)
- [FAIR Compliance](#fair-compliance)
- [License & Citation](#license--citation)

---

## Features

- **Advanced Query Ingestion**: Direct integration with the Elsevier Scopus Search API.
- **Affiliation Classification**: Heuristic classification of author affiliations into Academia, Industry, Mixed, or Unknown.
- **DOI Full-Text Retrieval & Fallback**: Queries Elsevier Article Retrieval API (`https://api.elsevier.com/content/article/doi/{doi}`). Automatically and gracefully falls back to Scopus abstracts on HTTP 401, 403, 404, 429, or missing DOIs.
- **Keyword Prevalence Engine**: Case-insensitive word-boundary frequency matching for user-defined keywords across full texts and fallback abstracts.
- **Top-Row KPI Metrics**: Displays Total Papers, Full-Text Success Rate, Keyword Match Rate, and Total Keyword Hits.
- **Tabbed Analytical Visualizations**:
  - **Affiliation Analysis**: Donut chart of overall distributions and line chart of temporal institutional shares.
  - **Keyword Prevalence**: Grouped bar chart (by affiliation) and temporal dynamics (interactive Heatmap Matrix or Trend Lines).
  - **Dataset & Provenance Export**: Data viewer with RFC 4180 CSV export embedding complete provenance metadata.
  - **Methodology & Documentation**: Complete definitions and protocol transparency.
- **Strict Academic Aesthetic**: Completely devoid of emojis across all UI elements, labels, charts, and metrics.

---
## Architecture & Data Pipeline

```mermaid
graph TD
    subgraph Frontend ["Streamlit UI (Multi-Page)"]
        UI_Input[User Configurations: Search, Filters, Full-Text Toggle]
        UI_Dash[Interactive Dashboards: Plotly Visualizations]
        UI_Export[Export Engine: RIS & LaTeX ZIP]
    end

    subgraph Data_Layer ["API Extraction Layer"]
        Scopus[Scopus Search API]
        Cond_FT{Full-Text Toggle}
        Ext_Meta[Extract Metadata, DOIs, & Abstracts]
        Elsevier[Elsevier Article Retrieval API]
        Fallback[Strict Fallback to Abstract]
    end

    subgraph Processing_Layer ["Categorization & Logic"]
        Parse_Affil[Affiliation Heuristics: Academia vs. Industry]
        Parse_Geo[Geo-Classification: EU/EEC vs. Non-EU/EEC]
        Match_Key[Keyword Frequency Matcher]
    end

    %% Flow of Execution
    UI_Input --> Scopus
    Scopus --> Cond_FT
    Cond_FT -- "False" --> Ext_Meta
    Cond_FT -- "True" --> Elsevier
    Elsevier -- "401/403/404 Error" --> Fallback
    Fallback --> Ext_Meta
    Elsevier -- "200 OK (Full Text)" --> Match_Key
    Ext_Meta --> Match_Key
    
    Scopus --> Parse_Affil
    Scopus --> Parse_Geo

    %% Output Routing
    Parse_Affil --> UI_Dash
    Parse_Geo --> UI_Dash
    Match_Key --> UI_Dash
    
    UI_Dash --> UI_Export
```
---

## Installation

### Prerequisites
- Python >= 3.10
- Elsevier Scopus API key (obtain from [dev.elsevier.com](https://dev.elsevier.com/myapikey.html))

### Setup
```bash
git clone https://github.com/OWNER/scopus-affiliation-analyzer.git
cd scopus-affiliation-analyzer

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and specify your SCOPUS_API_KEY
```

---

## Usage

Launch the dashboard:
```bash
source .venv/bin/activate
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Verification & Testing

Execute the automated test suite:
```bash
.venv/bin/python -m unittest discover -s tests -v
```

All 23 test suites verify:
- Affiliation classification (Academia, Industry, Mixed, Unknown, word-boundary regexes).
- Full-text retrieval, HTML/XML tag scrubbing, and abstract fallback under HTTP 200, 401, 403, 404, 429.
- Keyword frequency counting, phrase boundaries, and dataframe feature additions.

---

## Data Export & Provenance

CSV exports contain full provenance metadata columns:
- `provenance_query`: The original Scopus query string.
- `provenance_keywords`: The comma-separated target keywords evaluated.
- `provenance_year_start` & `provenance_year_end`: Temporal bounds applied.
- `provenance_exported_at`: ISO 8601 timestamp.
- `provenance_tool_version`: Software version identifier (`0.2.0`).

---

## License & Citation

Licensed under the [MIT License](LICENSE). Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).
