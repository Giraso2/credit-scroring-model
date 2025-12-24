# credit-scroring-model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# -------------------------------
# 1️⃣ Load or Create CSV Dataset
# -------------------------------
try:
    df = pd.read_csv("credit_data.csv")
    required_columns = ["name","age","income","debts","payment_history","loan_amount","credit_history","credit_score"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing in CSV")
    print("Existing CSV loaded!\n")
except (FileNotFoundError, KeyError):
    data = {
        "name": ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Henry"],
        "age": [25, 35, 45, 23, 52, 40, 29, 48],
        "income": [3000, 6000, 8000, 2000, 10000, 7000, 3500, 9000],
        "debts": [500, 2000, 1000, 200, 2500, 1800, 800, 2200],
        "payment_history": [0, 1, 0, 1, 0, 0, 1, 0],
        "loan_amount": [1000, 2000, 1500, 800, 3000, 2500, 1200, 2800],
        "credit_history": [1, 1, 1, 0, 1, 1, 0, 1],
        "credit_score": [1, 1, 1, 0, 1, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv("credit_data.csv", index=False)
    print("CSV created successfully!\n")

# -------------------------------
# 2️⃣ Features and Labels
# -------------------------------
X = df[["age", "income", "debts", "payment_history", "loan_amount", "credit_history"]]
y = df["credit_score"]

# -------------------------------
# 3️⃣ Split Data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -------------------------------
# 4️⃣ Train Models
# -------------------------------
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

models = {
    "Logistic Regression": log_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model
}

# -------------------------------
# 5️⃣ Evaluate Models
# -------------------------------
print("\n--- Model Evaluation ---")
for name, model in models.items():
    pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("ROC-AUC:", roc_auc_score(y_test, pred))
    print(classification_report(y_test, pred))

# -------------------------------
# 6️⃣ Interactive Loop - User Inputs Important History
# -------------------------------
print("\n--- Add New Applicants ---")
new_applicants = []  # Store new applicant names for visualization
while True:
    name_input = input("Enter Applicant Name: ")
    age = int(input("Enter Age: "))
    income = float(input("Enter Monthly Income: "))
    debts = float(input("Enter Total Debts: "))
    payment_history = int(input("Enter Payment History (0=Good, 1=Late): "))

    # Auto feature engineering
    loan_amount = 0.3 * income
    credit_history_score = 1 if payment_history == 0 else 0

    new_person = np.array([[age, income, debts, payment_history, loan_amount, credit_history_score]])

    # Predict for each model
    for model_name, model in models.items():
        result = model.predict(new_person)
        status = "Good Credit Risk" if result[0] == 1 else "Bad Credit Risk"
        print(f"{model_name} predicts for {name_input}: {status}")

    # Ask user to save to CSV
    save = input(f"Do you want to save {name_input} to CSV? (y/n): ").lower()
    if save == 'y':
        new_row = {
            "name": name_input,
            "age": age,
            "income": income,
            "debts": debts,
            "payment_history": payment_history,
            "loan_amount": loan_amount,
            "credit_history": credit_history_score,
            "credit_score": int(result[0])
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv("credit_data.csv", index=False)
        new_applicants.append(name_input)
        print(f"{name_input} saved to CSV!\n")

    # Continue or exit
    cont = input("Do you want to add another applicant? (y/n): ").lower()
    if cont != 'y':
        break

# -------------------------------
# 7️⃣ Final Visualization with Names
# -------------------------------
plt.figure(figsize=(10,6))

# Color coding: newly added applicants green, old ones red/blue
colors = []
for i, row in df.iterrows():
    if row["name"] in new_applicants:
        colors.append("green")  # newly added
    else:
        colors.append("red" if row["credit_score"]==0 else "blue")  # old applicants

scatter = plt.scatter(df["income"], df["loan_amount"], c=colors, s=100)

# Annotate each point with name
for i, row in df.iterrows():
    plt.text(row["income"]+100, row["loan_amount"]+50, row["name"], fontsize=9)

plt.xlabel("Income")
plt.ylabel("Loan Amount")
plt.title("Income vs Loan Amount (Credit Score & New Applicants Highlighted)")
plt.show()
