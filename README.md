# 🔍 AI/ML Engineering Internship Tasks – DevelopersHub Corporation

This repository contains my submission for 3 advanced tasks completed as part of the DevelopersHub Corporation AI/ML Engineering Internship (Due: 24th July 2025).

---

## ✅ Task 1: News Topic Classifier Using BERT

### 🔹 Objective

Fine-tune a transformer model (BERT) to classify news headlines into one of four categories: **World, Sports, Business, Sci/Tech** using the AG News dataset.

---

### 🔹 Dataset

- **AG News Dataset**  
  - Source: [AG News on Hugging Face](https://huggingface.co/datasets/ag_news)  
  - Loaded from CSV format (`train.csv`, `test.csv`)

---

### 🔹 Methodology / Approach

1. **Data Preparation**
   - Loaded CSV files into Pandas DataFrames
   - Adjusted labels from 1–4 to 0–3
   - Converted to Hugging Face `Dataset` objects

2. **Tokenization**
   - Used `bert-base-uncased` tokenizer with max length of 128
   - Tokenized only the `text` field, padded and truncated

3. **Model Fine-Tuning**
   - Used `AutoModelForSequenceClassification` with 4 output labels
   - Trained using Hugging Face `Trainer` and `TrainingArguments`
   - Computed `eval_accuracy` and `eval_f1` with a custom metrics function
   - Tracked training logs via [Weights & Biases](https://wandb.ai)

4. **Evaluation**
   - Evaluated on test set using accuracy and weighted F1-score

5. **Deployment**
   - Built a Gradio web app for real-time headline classification
   - Inference function supports GPU/CPU device placement

---

### 🔹 Key Results

- **Accuracy**: 93.5%  
- **F1 Score**: 93.5%  
- Fine-tuned BERT model performs robustly on short news headlines across four categories.

---

### 🔹 Demo

> 🧠 **Gradio App:**  
![alt text](<task 1 a.png>)  
![alt text](<task 1 b.png>)

---

## ✅ Task 2: End-to-End ML Pipeline with Scikit-learn

### 🔹 Objective

Build a reusable and production-ready machine learning pipeline for predicting **customer churn** using the **Telco Customer Churn** dataset.

---

### 🔹 Dataset

- **Telco Customer Churn**  
  - Source: [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
  - Loaded using `kagglehub` in Google Colab

---

### 🔹 Methodology / Approach

1. **Data Ingestion**
   - Downloaded using `kagglehub`
   - Loaded `.csv` with Pandas
   - Dropped empty rows and converted `TotalCharges` to numeric

2. **Preprocessing**
   - Applied `StandardScaler` to numeric columns: `tenure`, `MonthlyCharges`, `TotalCharges`
   - Applied `OneHotEncoder` to all categorical columns
   - Combined using `ColumnTransformer` and `Pipeline`

3. **Modeling**
   - Built and trained two full pipelines:
     - Logistic Regression
     - Random Forest
   - Used `GridSearchCV` for hyperparameter tuning (5-fold)

4. **Evaluation**
   - Assessed performance on a held-out test set
   - Reported accuracy and weighted F1-score
   - Selected the best pipeline for deployment

5. **Export**
   - Exported the complete pipeline using `joblib` as `telco_churn_pipeline.pkl`

---

### 🔹 Key Results

- **Best Model**: RandomForestClassifier (`max_depth=10`, `random_state=42`)
- **Test Accuracy**: **81%**
- **F1 Score (Weighted)**: **81%**
- **Class-wise Performance**:
  - **No Churn (Class 0)**: Precision 84%, Recall 90%, F1 Score 87%
  - **Churn (Class 1)**: Precision 71%, Recall 59%, F1 Score 64%

- The model performs strongly for predicting non-churned customers and reasonably well for detecting churned customers, which tend to be harder to classify. The final pipeline is saved and ready for deployment or batch inference.


---

### 🔹 Pipeline Visualization

> 🧠 **Best Estimator (Random Forest)**  
![Pipeline Output](<paste-your-pipeline-screenshot-path-or-url-here>)

---

## 🚧 Task 5: Auto-Tagging Support Tickets Using LLM (WIP)

### 🔹 Objective

Automatically tag free-text support tickets into categories using a large language model (LLM), applying:
- **Zero-shot classification**
- **Few-shot learning**
- **Fine-tuning (optional)**

_(Details and results will be added upon completion)_

---

## ✅ Submission Notes

- This repository contains 3 selected advanced tasks:
- ✅ Task 1: News Classifier with BERT
- ✅ Task 2: End-to-End ML Pipeline
- 🚧 Task 5: LLM Auto-Tagging for Support Tickets
- All tasks are:
- Modular and reproducible
- Evaluated using standard metrics
- Aligned with production-readiness practices

---

**📅 Final Submission Due**: July 24th, 2025  
**💼 Intern Name**: y/N  
**🏷️ GitHub Repo**: [github.com/your-username/internship-tasks](https://github.com/your-username/internship-tasks)
