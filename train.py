import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from protein_embedding import SupConEncoder
from loss import MultiSupConLoss
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from multi_class_dataset import MultiLabelDataset
df = pd.read_csv("protein_features.csv")
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df["subclass"], random_state=42)


le_subclass = LabelEncoder()
df_train["subclass_encoded"] = le_subclass.fit_transform(df_train["subclass"])
df_test["subclass_encoded"] = le_subclass.transform(df_test["subclass"])

X_train = df_train.drop(columns=["pdb_id", "label", "subclass", "subclass_encoded"]).values
y_train_label = df_train["label"].values
y_train_sub = df_train["subclass_encoded"].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_label_tensor = torch.tensor(y_train_label, dtype=torch.long)
y_subclass_tensor = torch.tensor(y_train_sub, dtype=torch.long)

batch_size = 24
dataset = MultiLabelDataset(X_tensor, y_label_tensor, y_subclass_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SupConEncoder(input_dim=X_tensor.shape[1], embed_dim=64).to(device)
criterion = MultiSupConLoss(temperature=0.07, lambda_label=1.0, lambda_subclass=1.0)
optimizer = optim.Adam(model.parameters(), lr=5e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

for epoch in range(1000):
    model.train()
    total_loss = 0
    for batch_x, batch_y_label, batch_y_sub in dataloader:
        batch_x, batch_y_label, batch_y_sub = batch_x.to(device), batch_y_label.to(device), batch_y_sub.to(device)
        embeddings = model(batch_x)
        loss = criterion(embeddings, batch_y_label, batch_y_sub)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

torch.save(model.state_dict(), "protein_embedding.pt")
cols_to_use = [col for col in df_test.columns if col not in ["pdb_id", "label", "subclass", "subclass_encoded"]]
X_test_scaled = scaler.transform(df_test[cols_to_use])
pd.DataFrame(X_test_scaled, columns=cols_to_use).to_csv("X_test_scaled.csv", index=False)
df_test[["pdb_id", "label", "subclass"]].to_csv("y_test_info.csv", index=False)