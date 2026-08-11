import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Titanic-Dataset.csv")

print("Original Dataset Shape:", df.shape)


# ==========================================
# 2. DATA CLEANING
# ==========================================

# Fill missing Age values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Remove duplicate records
df = df.drop_duplicates()

# Convert text values into numbers
encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])

print("Cleaned Dataset Shape:", df.shape)


# ==========================================
# 3. SELECT FEATURES AND TARGET
# ==========================================

X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

y = df["Survived"]


# ==========================================
# 4. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))


# ==========================================
# 5. TRAIN DECISION TREE MODEL
# ==========================================

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

model.fit(X_train, y_train)


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. MODEL ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Did Not Survive", "Survived"]
    )
)


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Did Not Survive", "Survived"]
)

display.plot(ax=ax)

ax.set_title("Confusion Matrix - Titanic Survival Prediction")
ax.set_xlabel("Predicted Outcome")
ax.set_ylabel("Actual Outcome")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(7, 5))

feature_importance.plot(kind="barh")

plt.title("Feature Importance in Survival Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()


# ==========================================
# PROJECT COMPLETE
# ==========================================

print("\nMachine Learning Project Completed Successfully!")