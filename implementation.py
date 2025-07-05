# JEPA-Based Multi-Modal Logistics Optimization Framework (Edge Scoring + Path Decoding)
# Phase 1: System Setup and Data Simulation

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn import BCEWithLogitsLoss

# ---- Configuration ----
NUM_CITIES = 10
NUM_MODES = 3  # Aerial, Road, Rail
TIME_STEPS = 12
EMBED_DIM = 64
MAX_PAYLOAD = [20, 1000, 5000]  # Max for UAV, Truck, Rail
ALLOWED_TRANSITIONS = [(0, 1), (1, 2), (2, 1), (1, 0)]
MAX_TIME = 10.0  # hours
MODE_SWITCH_PENALTY = 0.5
COST_WEIGHTS = {'dist': 1.0, 'time': 1.0, 'emission': 1.0}

# ---- Simulate Cities and Transportation Graph ----
np.random.seed(42)
torch.manual_seed(42)
city_coords = np.random.rand(NUM_CITIES, 2)

distances = torch.tensor([[[np.linalg.norm(city_coords[i] - city_coords[j])
                             for m in range(NUM_MODES)]
                             for j in range(NUM_CITIES)]
                             for i in range(NUM_CITIES)], dtype=torch.float32)

edge_attrs = {
    'cost': distances.clone(),
    'time': torch.rand(NUM_CITIES, NUM_CITIES, NUM_MODES),
    'emission': torch.rand(NUM_CITIES, NUM_CITIES, NUM_MODES)
}

normative_prior = torch.rand(NUM_CITIES, NUM_CITIES, NUM_MODES, TIME_STEPS)

# ---- Delivery Request ----
class DeliverySample:
    def __init__(self, src, dst, demand, time):
        self.src = src
        self.dst = dst
        self.demand = demand
        self.time = time

requests = [DeliverySample(np.random.randint(NUM_CITIES),
                           np.random.randint(NUM_CITIES),
                           np.random.uniform(10, 100),
                           np.random.randint(TIME_STEPS)) for _ in range(1000)]

# ---- Constraints ----
def is_feasible_mode(m, demand):
    return demand <= MAX_PAYLOAD[m]

def allowed_mode_transition(m1, m2):
    return (m1, m2) in ALLOWED_TRANSITIONS

# ---- Dataset ----
class LogisticsDataset(Dataset):
    def __init__(self, requests):
        self.samples = requests

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r = self.samples[idx]
        return {
            'src': r.src,
            'dst': r.dst,
            'demand': r.demand,
            'time': r.time,
            'N_slice': normative_prior[:, :, :, r.time],
            'edge_attrs': edge_attrs
        }

# ---- Expert Label Generator ----
def generate_expert_labels(src, dst, demand):
    labels = torch.zeros(NUM_CITIES, NUM_CITIES, NUM_MODES)
    G = nx.DiGraph()
    for i in range(NUM_CITIES):
        for j in range(NUM_CITIES):
            for m in range(NUM_MODES):
                if is_feasible_mode(m, demand):
                    dist = edge_attrs['cost'][i, j, m].item()
                    time = edge_attrs['time'][i, j, m].item()
                    emission = edge_attrs['emission'][i, j, m].item()
                    cost = COST_WEIGHTS['dist'] * dist + COST_WEIGHTS['time'] * time + COST_WEIGHTS['emission'] * emission
                    G.add_edge(i, j, weight=cost, mode=m)
    try:
        path = nx.shortest_path(G, source=src, target=dst, weight='weight')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            m = G[u][v]['mode']
            labels[u, v, m] = 1
    except:
        pass
    return labels

# ---- Model: Edge Score Predictor ----
class JEPA(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc_input = nn.Linear(3, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.gcn1 = GCNConv(1, embed_dim)
        self.gcn2 = GCNConv(embed_dim, embed_dim)
        self.prior_conv = nn.Sequential(
            nn.Conv2d(NUM_MODES, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, embed_dim, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, NUM_CITIES * NUM_CITIES * NUM_MODES)
        )

    def forward(self, src, dst, demand, edge_feat, N_slice):
        x = torch.tensor([src, dst, demand], dtype=torch.float32).unsqueeze(0)
        z_d = self.fc_input(x.unsqueeze(0))
        z_d = self.transformer(z_d).mean(dim=1).squeeze(0)

        node_x = torch.rand(NUM_CITIES, 1)
        edge_index = torch.nonzero(torch.ones(NUM_CITIES, NUM_CITIES)).T
        z_G = self.gcn2(self.gcn1(node_x, edge_index), edge_index).mean(dim=0)

        z_N = self.prior_conv(N_slice.permute(2, 0, 1).unsqueeze(0)).view(-1)

        z = torch.cat([z_d, z_G, z_N], dim=0)
        return self.fusion(z)

# ---- Decode Scores to Path ----
def decode_path_scores(out, src, dst, demand, threshold=0.3):
    plan = torch.sigmoid(out).view(NUM_CITIES, NUM_CITIES, NUM_MODES)
    G = nx.DiGraph()
    for i in range(NUM_CITIES):
        for j in range(NUM_CITIES):
            for m in range(NUM_MODES):
                if plan[i, j, m] > threshold and is_feasible_mode(m, demand):
                    dist = edge_attrs['cost'][i, j, m].item()
                    time = edge_attrs['time'][i, j, m].item()
                    emission = edge_attrs['emission'][i, j, m].item()
                    cost = COST_WEIGHTS['dist'] * dist + COST_WEIGHTS['time'] * time + COST_WEIGHTS['emission'] * emission
                    G.add_edge(i, j, weight=(1 - plan[i, j, m].item()) + cost, mode=m)

    print(f"Total scored edges: {G.number_of_edges()}")
    try:
        path = nx.shortest_path(G, source=src, target=dst, weight='weight')
        modes = [G[path[i]][path[i+1]]['mode'] for i in range(len(path)-1)]
        total_penalty = sum([MODE_SWITCH_PENALTY if modes[i] != modes[i-1] else 0 for i in range(1, len(modes))])
        print(f"Predicted Path: {path}")
        print(f"Modes: {modes} | Switch Penalty: {total_penalty:.2f}")
        return path, modes, plan
    except:
        print("No path found.")
        return [], [], plan

# ---- Visualize ----
def visualize_path(plan, path, src, dst, modes):
    G = nx.DiGraph()
    pos = {i: city_coords[i] for i in range(NUM_CITIES)}
    for i in range(NUM_CITIES):
        for j in range(NUM_CITIES):
            for m in range(NUM_MODES):
                if plan[i, j, m] > 0.01:
                    G.add_edge(i, j, weight=plan[i, j, m].item(), label=f"mode={m}")
    print(f"Total visual edges: {G.number_of_edges()}")
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700)
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    if path:
        edgelist = [(path[i], path[i+1]) for i in range(len(path)-1)]
        edge_colors = ['red'] * len(edgelist)
        nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color=edge_colors, width=2)
    plt.title(f"Predicted Path with Modes: {src} → {dst}")
    plt.show()

# ---- Train Model to Score Edges ----
dataset = LogisticsDataset(requests)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
model = JEPA(EMBED_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = BCEWithLogitsLoss()

for epoch in range(5):
    epoch_loss = 0
    for batch in dataloader:
        for i in range(batch['src'].shape[0]):
            src = batch['src'][i].item()
            dst = batch['dst'][i].item()
            demand = batch['demand'][i].item()
            N_slice = batch['N_slice'][i]
            edge_feat = {k: v for k, v in batch['edge_attrs'].items()}

            label = generate_expert_labels(src, dst, demand)
            out = model(src, dst, demand, edge_feat, N_slice)
            loss = loss_fn(out.view(-1), label.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
