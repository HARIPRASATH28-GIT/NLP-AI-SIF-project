# 🚨 AI-Powered SIF Precursor Detection & Prioritization System

> **AI/NLP Engine to Detect Serious Injury & Fatality (SIF) Precursors in Oil & Gas HSE Reports**

An AI-powered Natural Language Processing (NLP) system designed to analyze workplace incident and HSE reports, identify **Serious Injury & Fatality (SIF) precursors**, and prioritize high-risk incidents for safety teams.

The system transforms unstructured incident descriptions into actionable safety insights using **NLP, Machine Learning, and risk prioritization techniques**.

---

## 🎯 Problem Statement

Workplace incident reports often contain valuable safety information in the form of unstructured text. Traditional manual analysis of these reports can be:

* ⏳ Time-consuming
* 👨‍💼 Dependent on manual review
* ⚠️ Difficult to scale across thousands of reports
* 🔍 Prone to missing hidden SIF precursors
* 📊 Difficult to prioritize based on potential severity

A serious injury or fatality may be preceded by identifiable warning signs such as:

* Unsafe acts
* Unsafe conditions
* Human factors
* Equipment failures
* Hazardous environments
* Inadequate procedures
* Near-miss events
* High-energy events

### 💡 Our Solution

We propose an **AI-powered SIF Precursor Detection System** that automatically:

```text
Incident Report
       ↓
Text Preprocessing
       ↓
NLP Feature Extraction
       ↓
SIF Precursor Detection
       ↓
Risk Classification
       ↓
Severity / Priority Score
       ↓
Safety Dashboard & Alerts
```

---

## 🚀 Key Features

### 🧠 1. NLP-Based Incident Analysis

Processes unstructured HSE incident descriptions and extracts meaningful safety information.

### 🚨 2. SIF Precursor Detection

Identifies incident characteristics associated with potential serious injury or fatality events.

### 📊 3. Risk Classification

Classifies incidents according to their potential severity and risk level.

Example:

| Risk Level  | Meaning                                |
| ----------- | -------------------------------------- |
| 🟢 Low      | Low potential for serious consequences |
| 🟡 Medium   | Requires safety review                 |
| 🟠 High     | Significant SIF potential              |
| 🔴 Critical | Immediate safety attention required    |

### 🔎 4. Safety Factor Extraction

The system can identify important factors such as:

* Human factors
* Environmental factors
* Unsafe conditions
* Unsafe actions
* Equipment-related issues
* Work activities
* Incident types
* Nature of injury
* Body part affected

### 📈 5. Incident Prioritization

High-risk incidents can be ranked so that safety teams can focus on the most critical cases first.

### 📋 6. AI-Assisted Safety Insights

The system can provide interpretable information about why an incident was classified as high risk.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   HSE Incident      │
                    │      Reports        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Cleaning &   │
                    │  Preprocessing      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    NLP Pipeline     │
                    │                     │
                    │ Tokenization        │
                    │ Feature Extraction  │
                    │ Text Representation │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ML Classification │
                    │       Model         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ SIF Precursor       │
                    │ Detection           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Risk Prioritization │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Safety Dashboard    │
                    │ & Recommendations   │
                    └─────────────────────┘
```

---

# 🤖 Machine Learning Pipeline

The proposed ML pipeline consists of the following stages:

### 1. Data Collection

HSE and workplace incident datasets are collected from publicly available sources and relevant safety reports.

### 2. Data Preprocessing

The raw data is cleaned by:

* Removing duplicate records
* Handling missing values
* Normalizing text
* Removing irrelevant characters
* Standardizing categorical variables
* Preparing incident descriptions

### 3. Text Processing

Incident descriptions are converted into machine-readable representations using NLP techniques.

Possible approaches include:

* TF-IDF
* Word embeddings
* Sentence embeddings
* Transformer-based embeddings

### 4. Feature Engineering

Relevant features are extracted from incident reports, including:

```text
Incident Description
Event Type
Nature of Injury
Part of Body
Environmental Factors
Human Factors
Task Assigned
Construction End Use
Project Type
Building Information
```

### 5. SIF Classification

The ML model predicts whether an incident contains characteristics associated with SIF potential.

Example:

```text
Input:
"Worker fell from an elevated platform while performing
maintenance without adequate fall protection."

Output:
SIF Potential → HIGH
Risk Priority → CRITICAL
```

### 6. Risk Prioritization

Incidents are ranked based on their predicted SIF potential and relevant risk factors.

---

# 🧰 Technology Stack

## Backend

* Python
* Flask
* REST API
* python-dotenv

## Machine Learning

* Scikit-learn
* Pandas
* NumPy
* NLP techniques
* Machine Learning classification

## Frontend

* HTML / CSS / JavaScript
* Dashboard-based visualization

## Data

* HSE Incident Data
* OSHA Accident and Injury Data
* Incident descriptions
* Safety and injury-related attributes

## Development Tools

* Git
* GitHub
* VS Code / Cursor
* Python Virtual Environment

---

# 📁 Project Structure

```text
SIF_AI_Project/
│
├── backend/
│   ├── app.py
│   ├── model/
│   │   └── model.pkl
│   ├── requirements.txt
│   ├── .env
│   └── venv/
│
├── dataset/
│   └── incident_data.csv
│
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── README.md
├── .gitignore
└── requirements.txt
```

> **Note:** Do not upload your `venv/`, `.env`, API keys, passwords, or other secrets to GitHub.

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/SIF_AI_Project.git
cd SIF_AI_Project
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```powershell
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `python-dotenv` is missing:

```bash
pip install python-dotenv
```

## 4. Configure Environment Variables

Create a `.env` file:

```env
# Add required environment variables here
```

Never commit `.env` to GitHub.

## 5. Run the Backend

```bash
python app.py
```

The Flask API will start locally.

---

# 🔌 API Example

### Prediction Endpoint

```http
POST /predict
```

Example request:

```json
{
    "incident_description": "Worker fell from an elevated platform during maintenance work."
}
```

Example response:

```json
{
    "sif_prediction": "HIGH",
    "risk_level": "CRITICAL",
    "priority_score": 0.91
}
```

> The exact endpoint and response format may change as development continues.

---

# 📊 Example Workflow

### Input

```text
Worker was performing maintenance at height.
The worker lost balance and fell from the platform.
Fall protection was not properly used.
```

### AI Analysis

```text
Hazard Detected:
Working at Height

Potential Precursor:
Fall from Height

Safety Factor:
Inadequate Fall Protection

SIF Potential:
HIGH
```

### Prioritization

```text
Risk Level: 🔴 CRITICAL
Priority: Immediate Review
```

---

# 📈 Model Evaluation

The model will be evaluated using standard classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC

For SIF detection, **Recall is particularly important**, because missing a potentially serious incident can have significant safety consequences.

---

# 🔬 Future Enhancements

Future versions of the system may include:

* 🤖 Transformer-based NLP models
* 🧠 BERT-based incident classification
* 🔍 Explainable AI (XAI)
* 📊 Interactive safety dashboard
* 🚨 Real-time risk alerts
* 📈 SIF trend analysis
* 🗺️ Location-based risk visualization
* 🔄 Continuous model learning
* 🌐 Multilingual incident analysis
* 📱 Mobile safety application
* 📄 Automatic HSE report summarization
* 🎯 Recommended preventive actions

---

# 🛡️ Intended Impact

The goal of this project is to help organizations move from a **reactive safety approach** toward a more **proactive safety management system**.

```text
Traditional Approach

Incident
   ↓
Investigation
   ↓
Corrective Action


AI-Assisted Approach

Incident Data
   ↓
AI/NLP Analysis
   ↓
SIF Precursor Detection
   ↓
Risk Prioritization
   ↓
Early Safety Intervention
```

By identifying potential SIF precursors earlier, safety teams can prioritize high-risk incidents and take preventive action before more severe consequences occur.

---

# ⚠️ Disclaimer

This project is intended as a **research and decision-support system**.

AI predictions should not replace qualified safety professionals, engineering controls, organizational safety procedures, or regulatory requirements.

Model outputs should be reviewed by appropriate HSE personnel before being used for operational decisions.

---

# 👥 Project

**Project:** AI-Powered SIF Precursor Detection & Prioritization System

**Problem Statement:** SIH26165

**Domain:** Artificial Intelligence / Machine Learning / NLP / Occupational Safety

**Application Area:** Oil & Gas / Industrial Safety / HSE

---

# ⭐ Acknowledgements

This project uses publicly available workplace safety and incident datasets for research and development.

We acknowledge the organizations and researchers who make safety-related data and research available for improving occupational safety.

---

## 📜 License

This project is intended for educational, research, and hackathon purposes.

Add an appropriate open-source license before distributing the project publicly.
