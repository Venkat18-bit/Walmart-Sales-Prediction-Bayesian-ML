# Walmart Sales Prediction Using Bayesian Machine Learning

A data science and machine learning project analyzing Walmart weekly sales using probability theory, Bayes' theorem, and Naive Bayes classification.

The project investigates how holidays, seasonal patterns, weather conditions, fuel prices, unemployment, and store size relate to unusually strong sales weeks.

## Project Overview

Retailers need to anticipate changes in weekly sales to plan staffing and inventory effectively.

This project analyzes historical sales from 45 Walmart stores and builds a probabilistic classification model to identify strong sales weeks.

A strong sales week is defined as a week when sales are at least 10% above that store's average weekly sales.

## Dataset

**Source:** [Walmart Dataset – Kaggle](https://www.kaggle.com/datasets/yasserh/walmart-dataset)

- 6,435 store-week observations
- 45 Walmart stores
- 143 weeks
- February 2010 to October 2012
- 8 original dataset columns

The dataset is not redistributed in this repository.

### Dataset Setup

Download `Walmart.csv` from Kaggle and place it in a folder named `data` inside the repository.

The expected structure is:

```text
Walmart-Sales-Prediction-Bayesian-ML/
├── data/
│   └── Walmart.csv
├── Walmart_Sales_Bayesian_Analysis.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The `data/` folder and CSV are excluded from Git tracking.

## Technologies Used

- Python
- NumPy
- pandas
- scikit-learn
- Jupyter Notebook
- Probability and Bayesian statistics

## Methodology

### 1. Exploratory Probability Analysis

Defined and analyzed seven binary events:

- S: Strong sales week
- H: Holiday week
- Q: November–December
- C: Temperature below 40°F
- F: Fuel price at least $3.50
- J: Unemployment at least 8%
- G: Average store sales at least $1 million per week

Calculated marginal, joint, and conditional probabilities, with independent verification.

### 2. Bayesian Analysis

Applied:

- Conditional probability
- Bayes' theorem
- Law of total probability
- Independence checks
- Base-rate sensitivity analysis

Compared holiday-only and November–December staffing strategies using precision and recall.

### 3. Naive Bayes Classification

Implemented probabilistic classification using six binary features:

`H, Q, C, F, J, G`

Training and testing were separated chronologically:

- Training: 2010–2011
- Testing: 2012

Store averages used to define the target and store-size feature were calculated using training data only.

Calculated class priors and conditional likelihoods manually, applied Laplace smoothing, and verified the results against scikit-learn's `CategoricalNB(alpha=1)`.

### 4. Model Evaluation

Compared three strategies:

1. Categorical Naive Bayes
2. Always predict non-strong
3. Predict strong during holiday weeks

Evaluated confusion matrices, accuracy, precision, and recall.

Also examined classification thresholds of 0.1, 0.2, 0.3, and 0.5.

## Key Findings

- Strong weeks account for approximately 12.65% of all store-week observations.
- Thanksgiving has the highest observed strong-week probability among the holiday groups, at 82.22%.
- November–December weeks have a 41.11% strong-week probability.
- The holiday-only staffing rule has 27.11% precision and 14.99% recall.

### Test-Set Model Performance

| Model | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| Naive Bayes | 86.82% | 0.00% | 0.00% |
| Never Strong | 86.82% | 0.00% | 0.00% |
| Holiday Rule | 84.55% | 25.56% | 9.02% |

At the default classification threshold, Naive Bayes predicts every test observation as non-strong.

Its high accuracy therefore does not translate into successful strong-week detection.

Precision is reported as zero under the chosen evaluation convention when no positive predictions are made.

### Threshold Analysis

Lowering the classification threshold increases the number of strong-week predictions.

At threshold 0.1, Naive Bayes achieves 15.20% precision and 7.45% recall.

These results highlight the importance of class imbalance, threshold selection, and metrics beyond accuracy.

## Limitations

- The 2012 test data ends in October and does not include Thanksgiving or Christmas weeks.
- Some input features are conditionally dependent, violating the simplifying Naive Bayes assumption.
- The model's default threshold fails to identify strong weeks.
- Historical associations do not establish causal relationships.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Venkat18-bit/Walmart-Sales-Prediction-Bayesian-ML.git
cd Walmart-Sales-Prediction-Bayesian-ML
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the dataset

Download `Walmart.csv` from the Kaggle link above.

Create a local `data/` directory and place the dataset inside it.

### 5. Run the notebook

Open:

`Walmart_Sales_Bayesian_Analysis.ipynb`

Select the Python environment and run all cells sequentially.

## Conclusion

This project demonstrates the application of probability theory, Bayesian inference, and Naive Bayes classification to retail sales analysis.

It emphasizes leakage-free data preparation, independent mathematical verification, baseline comparison, and honest model evaluation.