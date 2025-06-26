# Salary Prediction App

##  Business Understanding

This project aims to assist individuals worldwide in understanding job market trends and predicting expected salary ranges based on personal and professional characteristics. By analyzing real-world salary survey data, this system provides valuable insights into how factors like job title, location, experience, and education impact compensation.

---

##  Business Problems

This project addresses the following key challenges:

- Lack of salary transparency across industries, roles, and countries.
- Difficulty for job seekers and professionals to estimate fair compensation.
- Limited data-driven insights for career planning and salary negotiation.

---

## Project Scope

The project focuses on:

- Data cleaning and normalization for high-cardinality fields like country, job title, and industry.
- Preprocessing and encoding features for machine learning input.
- Clustering salary ranges using KMeans instead of binning.
- Classification model to predict salary clusters.
- Deployment of the app using **Streamlit** for user interaction.

---

##  Preparation

### Data Source:

- Ask A Manager Salary Survey 2021 (Free-text responses)

### Environment Setup

```bash
# Clone the repository
git clone ttps://github.com/nath-nael/finalprojectprediction.git

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

##  File Overview

| File                       | Description                                                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `cleaning_dataset_1.ipynb` | Normalizes free-text fields: `Country`, `Industry`, `JobTitle` using regex, `pycountry`, and fuzzy matching. Outputs `CleanedDataset.csv`. |
| `processing.ipynb`         | Performs full preprocessing: encoding, normalization, clustering, and model training.                                                      |
| `app.py`                   | The deployed **Streamlit** app allowing interactive user input and salary prediction.                                                      |

---

##  Data Preprocessing Workflow
---
##  Exploratory Data Analysis (EDA)

### 1. Dataset Snapshot
| Aspect | Value |
|--------|-------|
| Observations | ± 5 k – 10 k individual salary records<sup>*</sup> |
| Features | 20 + demographic & job-related columns |
| Target | **TotalCompensation** (Salary + Additional Comp.) |

<sup>*</sup>Exact row count depends on the cleaning step; the notebook prints this with `df_raw.shape`.

---

### 2. Variable Overview

| Feature Type | Key Columns | Quick Notes |
|--------------|-------------|-------------|
| **Numeric**  | `Salary`, `AdditionalComp`, `TotalCompensation`, `ExperienceOverall`, `Age` | Right-skewed salary distribution; long tail of very high earners |
| **Categorical (high-cardinality)** | `Country`, `JobTitle`, `Industry` | Normalised & grouped in *cleaning_dataset_1.ipynb* |
| **Categorical (ordinal)** | `Education`, `ExperienceLevel` | Ordered encoding preserves hierarchy (e.g., *HS < BA < MA < PhD*) |
| **Categorical (nominal)** | `Gender`, `Race` | One-hot/target encoded |

---

### 3. Univariate Insights
- **Salary Distribution**  
  Histogram +kde shows a heavy right tail; median well below mean.  
  &nbsp;↳ Motivated log-scaling and later *TotalComp × 5* weighting before clustering.

- **Demographic Counts**  
  Count-plots indicate:
  - Gender responses are skewed toward one dominant group (survey bias).  
  - Highest share of respondents hold a **Bachelor’s** or **Master’s** degree.  
  - Experience peaks around the **5-10 year** bracket.

- **Top Locations**  
  Bar-plot of `Country_cleaned` reveals **United States**, **Canada**, **United Kingdom**, and **Germany** as the four largest respondent bases, followed by a long tail of single-digit-count countries.

---

### 4. Bivariate Insights
- **Salary vs. Education**  
  Box-plots show a clear upward trend: each successive education tier shifts the median salary higher and widens the inter-quartile range.

- **Salary vs. Gender**  
  Median salaries differ only slightly between groups; however, the *spread* is narrower for some, hinting at possible pay-variance disparities.

---

### 5. Data-Quality Checks
| Issue | Action |
|-------|--------|
| Missing numeric values | Filled with **0** (treated as “no response” in compensation fields). |
| Missing categorical values | Imputed with the **mode** per column. |
| Extreme outliers | Retained but down-weighted in clustering |

---

### 6. Key Takeaways for Modelling
1. **Positive skew** in salary → scale/weight before clustering.  
2. **High-cardinality text columns** require grouping & target encoding.  
3. **Imbalanced classes** (e.g., smaller salary clusters) will need attention during classification.

> These EDA findings directly informed the feature-engineering choices and the decision to use **KMeans (10 clusters, Silhouette 0.79)** followed by a **Ridge Classifier** to avoid overfitting.


---

### 1. Cleaning (in `cleaning_dataset_1.ipynb`)
- **Renaming columns**: Simplifying features from survey response.
- **Salary conversion**: Applying salary conversion to USD.
- **Handling imbalance data**: Implementing downsample to solve imbalance data.
- **Country normalization**: Standardized variants like "U.S.", "USA", "United States" → "United States".
- **Industry generalization**: Merged similar industries (e.g., "NGO", "Volunteering") into broad categories like "Non-Profit".
- **Job title standardization**: Mapped variants like "Sr. Data Analyst", "chief executive officer" → common roles.
- **Handling missing values**: Missing values imputation with (0) for numericals and (mode) for categoricals
Output: `CleanedDataset.csv`

---

### 2. Processing (in `processing.ipynb`)

- **Target Encoding**: Applied to categorical variables (Industry, JobTitle, Country, Race) based on `TotalCompensation`.
- **Ordinal Encoding**: For Age, Education, and Experience levels.
- **Gender Mapping**: Simplified gender responses into `Male`, `Female`, `Other`.

---

##  Clustering Instead of Binning

- Removed salary binning.
- **Created Clusters** using `KMeans` based on scaled features.
- Gave higher weight to `TotalCompensation` by multiplying it by 5.
- Chose `n_clusters=10` based on performance.

**Clustering Score** (Silhouette): `0.79`
Achieved the best Silhouette score with 4 features : Gender, Country, Industry, Job
Therefore, these 4 features will be used in model training.

Clusters were then treated as **target classes** for classification.

---

##  Model Selection and Training

###  Tested Classifiers:

- `LogisticRegression`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `KNeighborsClassifier`
- `XGBClassifier`
- `ExtraTreesClassifier`
- `MLPClassifier`
- `SVC`
- `GaussianNB`
- `RidgeClassifier` 

### Observation:

Most models achieved **>99% accuracy**, indicating **overfitting**.  
**Only RidgeClassifier** provided consistent results without overfitting.

---

##  Final Model: Ridge Classifier

### Why RidgeClassifier?

- Performs well with high-dimensional data.
- Handles multicollinearity and avoids overfitting.
- Simple and robust linear model, interpretable.

---

## Model Evaluation

| Metric           | Score                          |
| ---------------- | ------------------------------ |
| Accuracy         | ~87.1%                         |
| Model Robustness | High (no overfitting observed) |

          precision    recall  f1-score   support

       0       0.99      0.69      0.81       225
       1       0.98      1.00      0.99       108
       2       0.84      1.00      0.91       198
       3       0.84      1.00      0.92       605
       5       0.85      1.00      0.92        71
       6       0.00      0.00      0.00         1
       7       0.00      0.00      0.00        40
       8       0.00      0.00      0.00        44
       9       0.00      0.00      0.00        13

### Interpretation:

- **Strong performance** on Clusters 1–5 (especially clusters 1 to 3), with precision and recall above 0.84.
- **Weak performance** on Clusters 6–9 — likely due to:
  - Very low support (few samples).
  - Possible class imbalance and poor representation during training.
- **Macro average F1-score** is 0.50, indicating imbalanced performance across all classes.
- **Weighted average F1-score** is 0.83, reflecting the model's effectiveness in majority clusters.

### Recommendation:

To further improve the performance on underrepresented clusters:

- Consider **oversampling** (e.g. SMOTE) for minority clusters.
- Or group smaller clusters into broader salary bands (e.g., merging cluster 6–9).
- Investigate potential noise or inconsistencies in minority-class examples.

Ridge Classifier is a reliable, low-variance choice that maintains performance across most salary clusters without overfitting.

---

##  Streamlit Application

The deployed web app allows users to:

1. Input job-related details (title, country, experience, education, etc.).
2. View predicted salary cluster.
3. Gain insights into feature importance and salary trends.

To run:

```bash
streamlit run app.py
```
https://finalprojectprediction-bukxcnzjz42gxqe6qe7789.streamlit.app/

---

##  Summary

- A cleaned, preprocessed dataset with normalized country, job title, and industry fields.
- Unsupervised clustering used to create interpretable salary groups.
- Multiple classifiers evaluated—**RidgeClassifier chosen** for its generalization ability.
- Deployment-ready Streamlit app for public use.

---

##  Conclusion

This project delivers a realistic, interpretable salary prediction system based on survey data. It empowers:

- **Professionals** to assess salary expectations.
- **Job Seekers** to negotiate more confidently.
- **HR departments** to benchmark salary offerings fairly.

---

##  Recommended Action Items

- **Career Planning**: Suggest suitable roles based on income potential.
- **Negotiation Tools**: Leverage predictions for compensation discussions.
- **HR Benchmarking**: Identify salary gaps across roles and locations.
- **Education Advice**: Show return on investment for degree choices.
