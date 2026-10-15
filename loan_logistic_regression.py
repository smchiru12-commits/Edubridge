"""Loan eligibility with Logistic Regression.
x-axis = civil (credit) score, y-axis = loan amount required.
NOTE: no loan dataset was supplied, so a SYNTHETIC dataset is generated. Replace
`make_data()` with pd.read_csv('your_file.csv') (columns: civil_score, loan_amount, eligible)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def make_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    score = rng.integers(300, 901, n)                 # civil score 300-900
    amount = rng.uniform(1, 50, n).round(1)           # loan amount in lakhs
    z = 0.02 * (score - 650) - 0.12 * (amount - 20)   # higher score helps, bigger loan hurts
    eligible = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame(dict(civil_score=score, loan_amount=amount, eligible=eligible))

df = make_data(); df.to_csv("loan_dataset.csv", index=False)
X, y = df[["civil_score", "loan_amount"]], df.eligible
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(Xtr)
model = LogisticRegression().fit(sc.transform(Xtr), ytr)
pred = model.predict(sc.transform(Xte))
print("Accuracy:", round(accuracy_score(yte, pred), 3))
print(confusion_matrix(yte, pred)); print(classification_report(yte, pred, target_names=["Not eligible", "Eligible"]))

def check(score, amount):
    p = model.predict_proba(sc.transform(pd.DataFrame([[score, amount]], columns=X.columns)))[0, 1]
    return f"Score {score}, loan {amount}L -> {'ELIGIBLE' if p >= .5 else 'NOT ELIGIBLE'} (probability {p:.0%})"
for s, a in [(780, 10), (700, 25), (550, 15), (620, 40)]: print(check(s, a))

# ---- plot: x = civil score, y = loan amount, with decision boundary (p = 0.5) ----
gx, gy = np.meshgrid(np.linspace(300, 900, 300), np.linspace(0, 52, 300))
grid = pd.DataFrame(np.c_[gx.ravel(), gy.ravel()], columns=X.columns)
prob = model.predict_proba(sc.transform(grid))[:, 1].reshape(gx.shape)
fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(gx, gy, prob, levels=[0, .5, 1], colors=["#fde0dd", "#e5f5e0"], alpha=.8)
ax.contour(gx, gy, prob, levels=[.5], colors="k", linewidths=2)
ax.scatter(df.civil_score[y == 1], df.loan_amount[y == 1], c="green", s=14, label="Eligible")
ax.scatter(df.civil_score[y == 0], df.loan_amount[y == 0], c="red", s=14, label="Not eligible")
ax.set_xlabel("Civil (credit) score"); ax.set_ylabel("Loan amount required (lakhs)")
ax.set_title("Loan eligibility: Logistic Regression (black line = decision boundary)"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig("loan_eligibility_plot.png", dpi=120)
