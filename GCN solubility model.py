import pandas as pd
from rdkit import Chem
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# === Load Data === #
df = pd.read_csv("data/data_curated.csv")
df = df[["SMILES", "Solubility"]].dropna()


# === Atom Feature Engineering === #
def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetHybridization().real,
        int(atom.GetIsAromatic()),
    ]


# === Convert SMILES to Graph === #
def smiles_to_graph(smiles, y):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atoms = mol.GetAtoms()
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    x = torch.tensor([atom_features(atom) for atom in atoms], dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    data.y = torch.tensor([[y]], dtype=torch.float)
    return data


# === Train/Test Split === #
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# === Build Dataset === #
def build_dataset(df):
    dataset = []
    for row in df.itertuples(index=False):
        graph = smiles_to_graph(row.SMILES, row.Solubility)
        if graph:
            dataset.append(graph)
    return dataset


train_dataset = build_dataset(train_df)
test_dataset = build_dataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training on {len(train_dataset)} | Testing on {len(test_dataset)}")


# === Define 3-layer GCN === #
class GCN(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.lin = Linear(64, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# === Train the Model === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(input_dim=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for epoch in range(30):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d}, Loss: {total_loss:.4f}")

# === Evaluate === #
model.eval()
true_vals, pred_vals = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index, batch.batch)
        true_vals.extend(batch.y.squeeze().tolist())
        pred_vals.extend(preds.squeeze().tolist())

# === Metrics === #
r2 = r2_score(true_vals, pred_vals)
mae = mean_absolute_error(true_vals, pred_vals)
print(f"R²: {r2:.3f}, MAE: {mae:.3f}")

# === Plot === #
plt.figure(figsize=(6, 6))
plt.scatter(true_vals, pred_vals, alpha=0.6)
plt.xlabel("True Solubility")
plt.ylabel("Predicted Solubility")
plt.show()
