import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluation_metrics(y_test, y_pred, model_name):
    print(f"\n---- {model_name} ----")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
    print("F1 Score :", f1_score(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
# Dataset Cleaning
df = pd.read_csv('loan_approval_dataset.csv')
df = df.drop(columns=['loan_id'])
df.columns = df.columns.str.strip()
df['loan_status'] = (
    df['loan_status']
    .str.strip()
    .map({'Approved': 1, 'Rejected': 0})
)

# One-hot Encoding
df = pd.get_dummies(df, columns=['education','self_employed'],drop_first=True)

# Train Test Split of  Data
X = df.drop(columns=['loan_status'])
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling - Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled,y_train)
y_pred_lr = lr.predict(X_test_scaled)
evaluation_metrics(y_test=y_test,y_pred=y_pred_lr,model_name="Logistic Regression")

df.boxplot(column='cibil_score', by='loan_status')
plt.title("CIBIL Score vs Loan Status")
plt.suptitle("")
plt.xlabel("Loan Status")
plt.ylabel("CIBIL Score")
plt.show()

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train) 
y_pred_dt = dt.predict(X_test)
evaluation_metrics(y_test=y_test,y_pred=y_pred_dt,model_name="Decision Tree Classifier")
importances = dt.feature_importances_
features = X.columns

plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Decision Tree Feature Importance")
plt.show()
