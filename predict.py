import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from protein_embedding import SupConEncoder

df = pd.read_csv("protein_features.csv")
X = df.drop(columns=["pdb_id", "label", "subclass"]).values
y = df["label"].values
subclass = df["subclass"].values
feature_names = df.drop(columns=["pdb_id", "label", "subclass"]).columns.tolist()

X_train, X_test, y_train, y_test, sub_train, sub_test = train_test_split(
    X, y, subclass, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SupConEncoder(input_dim=X.shape[1], embed_dim=64).to(device)
encoder.load_state_dict(torch.load("protein_embedding.pt", map_location=device))
encoder.eval()

with torch.no_grad():
    X_train_embed = encoder(torch.tensor(X_train_scaled, dtype=torch.float32).to(device)).cpu()
    X_test_embed = encoder(torch.tensor(X_test_scaled, dtype=torch.float32).to(device)).cpu()

def calculate_metrics(y_true, y_pred, model_name, feature_type):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else "N/A"
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)
    return {
        "Model": model_name,
        "Features": feature_type,
        "Accuracy": acc,
        "AUC": auc,
        "Precision": precision,
        "F1": f1
    }

def record_misclassified_subclasses(preds, y_true, subclass_list, model_name, feature_type):
    error_counts = {}
    for pred, true, sub in zip(preds, y_true, subclass_list):
        if pred != true:
            error_counts[sub] = error_counts.get(sub, 0) + 1

    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    filename = f"misclassified_{model_name.lower().replace(' ', '_')}_{feature_type.lower()}.md"
    with open(filename, "w") as f:
        f.write(f"# ❌ Misclassified Subclasses for {model_name} ({feature_type})\n\n")
        f.write("| Subclass | Misclassified Count |\n|----------|---------------------|\n")
        for sub, count in sorted_errors:
            f.write(f"| {sub} | {count} |\n")
    print(f"📉 Saved subclass errors to: {filename}")

if __name__ == "__main__":
    results = []

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    for feature_type, (Xtr, Xte) in {
        "Raw": (X_train_scaled, X_test_scaled),
        "Projected": (X_train_embed, X_test_embed)
    }.items():
        for model_name, model in models.items():
            model.fit(Xtr, y_train)
            preds = model.predict(Xte)
            results.append(calculate_metrics(y_test, preds, model_name, feature_type))
            record_misclassified_subclasses(preds, y_test, sub_test, model_name, feature_type)

    # === Save summary results as markdown ===
    markdown_table = "| Model           | Features   | Accuracy | AUC   | Precision | F1    |\n"
    markdown_table += "|-----------------|------------|----------|-------|-----------|-------|\n"
    for result in results:
        auc_str = f"{result['AUC']:.4f}" if isinstance(result["AUC"], float) else result["AUC"]
        markdown_table += f"| {result['Model']} | {result['Features']} | {result['Accuracy']:.4f} | {auc_str} | {result['Precision']:.4f} | {result['F1']:.4f} |\n"

    with open("results.md", "w") as f:
        f.write("# 📊 Model Comparison Summary\n\n" + markdown_table)

    print("\n✅ Results saved to results.md")
