# Forecasting & Nowcasting with Text as Data

This repository contains teaching materials for learning how to build text-based indicators and apply them in forecasting and nowcasting tasks. Through practical, hands-on examples, it covers zero- and few-shot learning for text classification, event detection with limited labels, and mixed-frequency data techniques for time series modeling.

## 📦 Repository Structure

```
BSE-FORECASTNLP/
├── session1/
├── session2/
├── session3/
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

* **Session 1**: Learning with limited supervision: zero and few-shot learning
* **Session 2**: Fine-tuning and policy-oriented evaluation
* **Session 3**: Classic MIDAS and machine learning extensions

## ✅ Setup Instructions

For this course, the recommended setup is a standard Python virtual environment using `requirements.txt`.

Before starting, make sure you have:

- **Git** installed: check with `git --version`
- **Python 3.11** installed
- **VSCode** installed, preferably with the Python and Jupyter extensions

1. **Check Python 3.11**

   ```bash
   python3.11 --version
   ```

   If Python 3.11 is not installed, download it from the [official Python website](https://www.python.org/downloads/).

2. **Clone the repository**

   ```bash
   git clone https://github.com/RenatoVassallo/BSE-ForecastNLP.git
   cd BSE-ForecastNLP
   ```

3. **Create and activate a virtual environment**

   On macOS / Linux:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  
   ```

   On Windows: 
   
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   This step may take a few minutes depending on your system. Please be patient.

5. **Select the environment in VSCode**

   Open a notebook in VSCode and select the Python interpreter associated with:

   ```plain
   .venv

## 🚀 Optional: uv setup

If you are familiar with uv, you can instead run:

```bash
uv sync
```
This uses the project configuration and lock file. 


## Notes

* Materials are designed for teaching and illustration, not as production-ready pipelines.
* Please use the latest requirements.txt from this repository.
* If installation issues arise, first make sure your Python version is 3.11 and that your virtual environment is activated.
