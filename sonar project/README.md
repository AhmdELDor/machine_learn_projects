# Sonar — Rock vs. Mine Prediction

A binary classification project that predicts whether an object detected by sonar is a **rock (R)** or a **mine (M)**, based on 60 numerical features representing sonar signal strengths at different frequencies.

## Dataset

- `sonar.csv` — 208 samples, 60 feature columns (no header) and 1 label column (`R` or `M`).

## Model

- **Algorithm:** Logistic Regression (`scikit-learn`)
- **Split:** 90% training / 10% testing, stratified on the label.
- Reports accuracy on both training and test data, then runs a prediction on a single sample input.

## How to run

```bash
pip install numpy pandas scikit-learn
python sonar_mine_project.py
```

Run the script from inside this folder, since it reads `./sonar.csv`.
