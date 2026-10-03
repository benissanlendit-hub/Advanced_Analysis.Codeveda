## Task 1: Predictive Modeling (Classification)
Customer churn prediction on the BigML Telecom dataset (2,666 train / 667 test 
records), comparing Logistic Regression, Decision Tree, and Random Forest, 
tuned via GridSearchCV.

**Key result:** Decision Tree best model — 94.6% accuracy, 0.80 F1-score.

Run: `python task1_classification_analysis.py`

## Task 2: Building Dashboards with Power BI
Interactive churn dashboard: geographic map (churn rate by US state), KPI 
cards, bar/line charts, and slicers for International plan, Voice mail plan, 
and Area code.

**Note:** published online sharing requires a Power BI Pro / work account, 
which isn't available here — `PowerBI_Dashboard_Export.pdf` is included as a 
static export, and the live interactive demo is shown in the video walkthrough.

## Files
- `task1_classification_analysis.py` — source code
- `churn-bigml-80.csv` / `churn-bigml-20.csv` — official train/test split
- `Codveda_Level3_Task1_Classification_Showcase.pptx` — Task 1 deck
- `Codveda_Level3_Task2_PowerBI_Showcase.pptx` — Task 2 deck
- `PowerBI_Dashboard_Export.pdf` — dashboard export
- `Guide_Code_Classification.pdf` — line-by-line code walkthrough

**Tools:** Python, pandas, scikit-learn, matplotlib, Power BI Desktop
