# Nowcasting and Forecasting with Text as Data

This repository contains teaching materials — including slides, code, and datasets — for learning how to build text-based indicators and apply them in forecasting and nowcasting tasks. Through practical, hands-on examples, it covers zero- and few-shot learning for text classification, event detection with limited labels, and mixed-frequency data techniques for time series modeling.

## 📦 Repository Structure

```
BSE-FORECASTNLP/
├── session1/
├── session2/
├── session3/
├── .gitignore
├── README.md
└── requirements.txt
```

* **Session 1**: Zero- and Few-Shot Learning for Text Classification
* **Session 2**: Nowcasting Political Events with Limited Labeled Data
* **Session 3**: Mixed Data Sampling Approaches for Time Series Nowcasting and Forecasting

## ✅ Setup Instructions

1. **Ensure Python 3.11 is installed:**

   Before proceeding, verify that Python 3.11 is installed:

   ```bash
   python3.11 --version
   ```

   If not, download and install it from the [official Python website](https://www.python.org/downloads/).

2. **Clone the repository:**

   ```bash
   git clone https://github.com/RenatoVassallo/BSE-ForecastNLP.git
   cd BSE-ForecastNLP
   ```

3. **Create a virtual environment with Python 3.11:**

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate   # On Windows: .\.venv\Scripts\activate
   ```

4. **Install the required libraries using the latest requirements.txt:**

   Ensure that you only use the `requirements.txt` file from the GitHub repository. Avoid using any previous versions.

   ```bash
   pip install -r requirements.txt
   ```

5. **You are ready to go!**

   Navigate to the `session1` folder and try running any notebook.

Feel free to open issues or reach out if you have any questions. 🚀
