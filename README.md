# Feature Relationship and Correlation Analyzer

## Overview

Feature Relationship and Correlation Analyzer is an interactive data visualization application built using Python and Streamlit. It allows users to upload any CSV dataset, analyze relationships between numerical features using Pearson and Spearman correlation, visualize the results through heatmaps and pairplots, and generate a downloadable correlation summary report.

**Live Demo:** https://syntecxappfeaturerelationshipcorrelationanalyzer-gfdxkappecg4u.streamlit.app (Interactive Web Application)

---

## Features

- Upload any CSV dataset
- View dataset overview and summary statistics
- Generate Pearson and Spearman correlation heatmaps
- Explore pairwise relationships between numerical features
- Identify the strongest positive and negative correlations
- Download correlation heatmap
- Download pairplot
- Generate and download correlation summary report

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit

---

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Project Structure

```text
Feature_Relationship_Correlation_Analyzer/
│
├── app.py
├── README.md
├── requirements.txt
├── titanic.csv
├── winequality-red.csv
└── images/
```

---

## Author

**Janmeshree Gaokar**
