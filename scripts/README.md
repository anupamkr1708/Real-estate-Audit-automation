# Agentic Audit Automation

This project streamlines the audit workflow for lease documents using a modular, agent‑driven automation pipeline. Instead of manually opening Excel sheets, checking PDFs, validating numbers, and preparing reports, the system breaks the entire audit into specialized components that work together like a team of individual audit agents.

Each script has a clear and focused role — extraction, validation, calculations, reporting — which makes the whole workflow easier to extend, debug, and maintain.

---

# Key Features

1. Automated Document Discovery
The system identifies relevant audit files automatically:
- Prior AE Excel sheets  
- Owner AE Excel sheets  
- Rent Roll PDFs  
- Stacking Plan PDFs  

Handled by: `auto_discovery_agent.py`

---

2. Intelligent Data Extraction
Specialized extractors handle multiple document types:
- `combined_extractor.py` – Coordinates extraction across all excel sources  
- `pdf_extractor.py` – Extracts text/tables and all the relevant contents from the PDFs present in the directory  
 

Extracted files go into:
```
results/extracted_xls/
results/extracted_pdf/
```

---

3. Data Validation
The `validator.py` module performs structural and business-rule validation:
- Required field checks  
- Consistency checks  
- Rent/area mismatches  
- Schema anomalies  

Validated outputs are stored in `results/validated/`.

---

4. Calculations
All audit‑related computations (variance, comparisons, rollups) are done in:
- `calculator.py`

---

5. Reporting
`reporter.py` produces:
- Clean spreadsheets  
- Summary reports  
- Audit findings  

Results are placed in:
```
results/reports/
results/findings/
```

---

6. Orchestration
The heart of the system is:
- `master_orchestrator.py`

It runs the entire pipeline:
1. Discover →  
2. Extract →  
3. Validate →  
4. Calculate →  
5. Report  

Run it with:
```
python master_orchestrator.py
```

---

7. One‑Click Setup
The `setup.py` script handles:
- Python version checks  
- Package installation  
- Directory creation  
- Required file checks  

Use it before running anything else:
```
python setup.py
```

---

#Project Structure

```
scripts/
│
├── auto_discovery_agent.py
├── calculator.py
├── combined_extractor.py
├── complete_pipeline.py
├── extract_prior_ae.py
├── master_orchestrator.py
├── pdf_extractor.py
├── quick_fix.py
├── rent_roll_extractor.py
├── reporter.py
├── setup.py
└── validator.py
```

---

# USAGE

1. Run setup (first time)
```
python setup.py
```

2. Add documents
Place your input files in the `data/` directory:
- Rent Roll PDFs  
- Stacking Plan PDFs  
- Prior AE Excel files  
- Owner AE Excel files  

3. Run the automation
```
python master_orchestrator.py
```

4. View results
Check the `results/` folder for:
- Extracted files  
- Validated outputs  
- Reports  
- Findings  

---


