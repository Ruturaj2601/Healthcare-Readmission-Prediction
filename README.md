# 🏥 Healthcare Readmission Prediction
## Predicting 30-Day Hospital Readmissions for Diabetes Patients

An end-to-end healthcare analytics project using the UCI Diabetes 130-US Hospitals dataset (1999–2008). Built a machine learning pipeline to predict 30-day readmissions, stratify patient risk, and quantify potential cost savings from early intervention.

---

## 💡 Key Results

| Metric | Result |
|--------|--------|
| 📋 Total Patient Records | **101,766** |
| 🔁 30-Day Readmission Rate | **11.16% (11,357 readmissions)** |
| 🏆 Best Model | **Random Forest** |
| 🎯 Test Accuracy | **75.54%** |
| 📈 ROC AUC Score | **0.6529** |
| 🔍 Recall (Readmissions Caught) | **38.09%** |
| 💰 Potential Net Cost Savings | **$4,644,500** |
| 🚨 Very High Risk Patients | **265 patients (41.5% readmission rate)** |

---

## 🎯 Business Objectives

1. **Risk Prediction** — Identify diabetes patients likely to be readmitted within 30 days of discharge
2. **Risk Stratification** — Classify patients into Low / Medium / High / Very High risk tiers
3. **Cost Impact Analysis** — Quantify potential savings from targeted intervention programs
4. **Clinical Recommendations** — Provide actionable guidance for care teams at discharge

---

## 📁 Project Structure

```
healthcare-readmission-prediction/
│
├── diabetes_data/
│   ├── diabetic_data.csv              # Raw UCI dataset (101,766 records)
│   └── IDS_mapping.csv                # Diagnosis code mappings
│
├── diabetic_data_cleaned.csv          # Preprocessed dataset (ready for modeling)
├── Capstone3_Healthcare_Analytics.ipynb  # Complete analysis notebook
├── EXECUTIVE_SUMMARY.txt              # Full executive report with findings
├── HealthcareAnalytics_Project.pdf    # Project documentation
│
├── visualizations/                    # 12 professional visualizations
│   ├── 01_target_distribution.png
│   ├── 02_numeric_distributions.png
│   ├── 03_categorical_distributions.png
│   ├── 04_numeric_vs_readmission.png
│   ├── 05_categorical_vs_readmission.png
│   ├── 06_correlation_matrix.png
│   ├── 07_model_comparison.png
│   ├── 08_confusion_matrices.png
│   ├── 09_feature_importance.png
│   ├── 10_probability_distribution.png
│   ├── 11_risk_stratification.png
│   └── 12_FINAL_DASHBOARD.png
│
└── README.md
```

---

## 🔬 Analytics Pipeline

### Phase 1 — Data Acquisition & Understanding
- Loaded UCI Diabetes dataset: **101,766 records**, 50 features
- Assessed data quality — missing values in `weight` and `specialty` fields
- Defined target variable: 30-day readmission (binary classification)

### Phase 2 — Data Preprocessing & Feature Engineering
- Missing value imputation and outlier handling
- Diagnosis code mapping using `IDS_mapping.csv`
- Categorical encoding for 18 clinical and demographic variables
- Stratified train/test split: **81,412 training | 20,354 test records**

### Phase 3 — Exploratory Data Analysis
- Univariate and bivariate analysis across clinical variables
- Key finding: elderly patients (70+) show significantly elevated readmission rates
- Circulatory and respiratory diagnoses correlated with higher readmission risk
- Emergency department utilization identified as strong readmission predictor

### Phase 4 — Model Development & Evaluation

| Model | Accuracy | ROC AUC | Recall |
|-------|----------|---------|--------|
| Logistic Regression | baseline | — | — |
| Random Forest ✅ | **75.54%** | **0.6529** | **38.09%** |
| Gradient Boosting | compared | — | — |

**Confusion Matrix (Random Forest on test set):**
- True Negatives: **14,510** — correctly identified low-risk patients
- True Positives: **865** — high-risk patients correctly flagged
- False Negatives: **1,406** — missed readmissions (minimizing this is clinical priority)
- False Positives: **3,573** — unnecessary interventions

### Phase 5 — Risk Stratification

| Risk Tier | Patients | Readmission Rate |
|-----------|----------|-----------------|
| Low Risk | 3,070 | 4.9% |
| Medium Risk | 12,846 | 9.8% |
| High Risk | 4,173 | 18.1% |
| Very High Risk | 265 | **41.5%** |

### Phase 6 — Business Impact Analysis

Using conservative assumptions ($15,000 avg readmission cost, $1,000 intervention cost, 70% intervention effectiveness):

- Patients flagged for intervention: **4,438**
- Estimated readmissions prevented: **606**
- Total intervention cost: **$4,438,000**
- Potential cost savings: **$9,082,500**
- **Net benefit: $4,644,500**

---

## 🔑 Key Risk Factors Identified

1. **Prior inpatient admissions** — strongest predictor of readmission
2. **Emergency department utilization** — higher visits strongly correlate with readmission
3. **Diagnosis category** — circulatory and respiratory conditions show elevated risk
4. **Age** — patients 70+ have significantly higher readmission rates
5. **Diabetes management metrics** — medication changes and glucose levels

---

## 📊 Key Visualizations

| Visualization | Description |
|--------------|-------------|
| `07_model_comparison.png` | Performance comparison across all models |
| `09_feature_importance.png` | Top clinical predictors of readmission |
| `11_risk_stratification.png` | Patient distribution across risk tiers |
| `12_FINAL_DASHBOARD.png` | Executive summary dashboard |

---

## 💻 Code Highlights

### Data Preprocessing
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('diabetic_data_cleaned.csv')

# 18 clinical and demographic features
features = ['age', 'time_in_hospital', 'num_lab_procedures',
            'num_procedures', 'num_medications', 'number_outpatient',
            'number_emergency', 'number_inpatient', ...]

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df['readmitted'],
    test_size=0.2, stratify=df['readmitted'], random_state=42
)
# Training: 81,412 records | Test: 20,354 records
```

### Random Forest Model
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # handles class imbalance
    random_state=42
)
rf_model.fit(X_train, y_train)

# Results: Accuracy 75.54% | ROC AUC 0.6529 | Recall 38.09%
```

### Risk Stratification
```python
# Assign risk tiers based on predicted probability
churn_prob = rf_model.predict_proba(X_test)[:, 1]

def assign_risk(prob):
    if prob < 0.10: return 'Low Risk'
    elif prob < 0.20: return 'Medium Risk'
    elif prob < 0.40: return 'High Risk'
    else: return 'Very High Risk'

# Very High Risk: 265 patients at 41.5% readmission rate
```

---

## 🏥 Clinical Recommendations

**Immediate Actions**
- Implement risk scoring at discharge for all diabetes patients
- Target intensive case management for the **265 Very High Risk** patients
- Enhance discharge planning protocols for High/Very High tiers

**Resource Allocation**
- Prioritize follow-up appointments for High + Very High Risk patients
- Allocate transitional care resources based on model risk scores
- Consider home health services for highest-risk patients

**Model Deployment Path**
- Integrate model into Electronic Health Record (EHR) system
- Provide real-time risk scores to care teams at point of discharge
- Retrain model quarterly with new patient data

---

## ⚠️ Limitations

- Dataset covers 1999–2008; clinical practices have evolved since
- Class imbalance (11.16% positive rate) affects precision
- Socioeconomic factors not included in current model
- Missing data in `weight` and `specialty` fields limits some analyses
- External validation needed before clinical deployment

---

## 🛠️ Technical Specifications

- **Language:** Python 3.x
- **Libraries:** pandas, scikit-learn, matplotlib, seaborn
- **Dataset:** UCI ML Repository — Diabetes 130-US Hospitals
- **Features:** 18 clinical and demographic variables
- **Validation:** Stratified 80/20 train-test split

---

## 🚀 Future Enhancements

- [ ] LSTM/deep learning model for sequential patient data
- [ ] Include socioeconomic and social determinants of health features
- [ ] Real-time prediction API (FastAPI) for EHR integration
- [ ] Streamlit dashboard for clinical staff
- [ ] Fairness analysis across demographic groups
- [ ] External validation on post-2008 dataset

---

*Built with Python | Pandas | Scikit-learn | Matplotlib | Seaborn*
