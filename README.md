# House Price Prediction MLOps Project

**Student Name:** Rabbi Awan  
**Student ID:** 23L-2554

This project demonstrates a basic MLOps workflow using Git, GitHub, VS Code, and Scikit-Learn. It trains a Random Forest regression model to predict house prices.

## Project Structure

```text
mlops-project-23L-2554/
├── data/
│   └── house_prices_23L-2554.csv
├── model/
│   └── house_price_model_23L-2554.pkl
├── src/
│   ├── train.py
│   └── train_23L-2554.py
├── .gitignore
├── requirements.txt
└── README.md
```

The `data/` and `model/` directories are excluded from Git because datasets and trained model artifacts should not be committed.

## Dataset

The project uses the Housing Prices Dataset from Kaggle:

https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

Download `Housing.csv`, rename it to `house_prices_23L-2554.csv`, and place it inside the `data/` directory.

## Setup Instructions

Open a terminal in the project root and create a virtual environment:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
pip install -r requirements.txt
```

## Run the Training Script

From the project root, run:

```powershell
python src/train.py
```

The trained model will be saved as:

```text
model/house_price_model_23L-2554.pkl
```