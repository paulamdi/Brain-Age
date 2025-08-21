# shap_contrastive_learning for full fine tuned model
# 

"""
SHAP Contrastive Learning — Phase 1:
1. Load SHAP vectors (z-scored)
2. Generate positive/negative pairs via top-k similarity
3. Build Dataset and DataLoader
4. Define projection head + contrastive loss (NT-Xent)
5. Train and save embeddings
"""

# === Step 1: Load SHAP vectors ===
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Load SHAP vectors
df_shap_z = pd.read_csv("FULL_FINE_TUNING_shap_global_features_zscored.csv")
subject_ids = df_shap_z["Subject_ID"].values
shap_vectors = df_shap_z.drop(columns=["Subject_ID"]).values

# Map Subject_ID to SHAP vector
id_to_vector = {subj: vec for subj, vec in zip(subject_ids, shap_vectors)}
print(id_to_vector[2110])  # example


print("SHAP input dimension:", shap_vectors.shape[1])



# === Step 2: Load triplets ===
triplet_df = pd.read_csv("FULL_FINE_TUNING_shap_triplets_topk.csv")

# Convert to list of tuples for easier use
triplets = list(zip(triplet_df["anchor"], triplet_df["positive"], triplet_df["negative"]))





# === Step 3: TripletDataset and DataLoader for SHAP Contrastive Learning ===

import torch
from torch.utils.data import Dataset, DataLoader

# --- Define a custom Dataset class that handles SHAP triplets ---
class TripletDataset(Dataset):
    def __init__(self, triplets, id_to_vector):
        """
        triplets: list of (anchor, positive, negative) subject IDs
        id_to_vector: dictionary mapping subject ID → SHAP z-scored vector
        """
        self.triplets = triplets                      # Store the triplets
        self.id_to_vector = id_to_vector              # Store the SHAP z-scored vectors

    def __len__(self):
        # Return the total number of triplets
        return len(self.triplets)

    def __getitem__(self, idx):
        # Get subject IDs for this triplet
        anchor_id, pos_id, neg_id = self.triplets[idx]

        # Convert SHAP vectors from numpy to PyTorch tensors
        anchor = torch.tensor(self.id_to_vector[anchor_id], dtype=torch.float32)
        positive = torch.tensor(self.id_to_vector[pos_id], dtype=torch.float32)
        negative = torch.tensor(self.id_to_vector[neg_id], dtype=torch.float32)

        # Return the triplet
        return anchor, positive, negative

# --- Create the Dataset instance ---
# triplets: a list of (anchor, positive, negative) from your earlier CSV
# id_to_vector: dictionary mapping Subject_ID → SHAP z-scored vector (from FULL_FINE_TUNING_shap_global_features_zscored.csv)

dataset = TripletDataset(triplets, id_to_vector)  # Create dataset object

# --- Create the DataLoader ---
# Batching and shuffling for training
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # Use batch size = 64






# === Step 4: SHAP Contrastive Model + NT-Xent Loss ===

# =====================
# SHAP Contrastive Learning: Projection & NT-Xent Loss
# =====================

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================
# Projection Head Module
# =====================

input_dim = shap_vectors.shape[1]

class ShapProjectionHead(nn.Module):
    def __init__(self, input_dim=input_dim, hidden_dim=64, output_dim=32):
        super(ShapProjectionHead, self).__init__()
        
        # Linear layer to reduce dimensionality from input_dim to hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Linear layer to project from hidden_dim to output_dim (final embedding size)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Apply first linear layer + ReLU activation
        x = F.relu(self.fc1(x))  # Shape: (batch_size, hidden_dim)
        
        # Apply second linear layer (no activation)
        x = self.fc2(x)  # Shape: (batch_size, output_dim)
        
        # Normalize the output embeddings to unit length (L2 norm)
        return F.normalize(x, p=2, dim=1)  # Shape: (batch_size, output_dim)
    
            # p= 2 Indica que se usa la norma L2 (Euclídea) para la normalización
            #dim=1 Se normaliza cada fila del tensor x (es decir, cada vector del batch)
        
        # Lentgh 1 Normalize
        # Cosine similarity compares the direction between two vectors, not their magnitude. 
        # If you don’t normalize them, the model might learn to produce “larger” embeddings instead of “more aligned” ones.
        # With this line, you ensure that the model learns only differences in orientation (direction), not in scale

# =====================
# NT-Xent Loss for Triplets
# =====================

class TripletNTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(TripletNTXentLoss, self).__init__()
        self.temperature = temperature  # Temperature parameter controls softness of distribution

    def forward(self, anchor, positive, negative):
        # anchor, positive, negative: all have shape (batch_size, embed_dim)

        # Compute cosine similarity (dot product since vectors are normalized)
        # Similarity anchor–positive (sim_ap) and anchor–negative (sim_an).
        sim_ap = torch.sum(anchor * positive, dim=1)  # Shape: (batch_size,)
        sim_an = torch.sum(anchor * negative, dim=1)  # Shape: (batch_size,)

        # Scale similarities by temperature
        # Lower T -> separates more the differences
        # To improve contrastive learninge, more reliable
        sim_ap = sim_ap / self.temperature  # Shape: (batch_size,)
        sim_an = sim_an / self.temperature  # Shape: (batch_size,)

        # Concatenate positive and negative similarities to form logits
        # Creats a tensor : logits
        logits = torch.stack([sim_ap, sim_an], dim=1)  # Shape: (batch_size, 2)

        # Targets: 0 means anchor is most similar to the first input (positive)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)  # Shape: (batch_size,)

        # Compute cross-entropy loss over (positive vs. negative)
        # Penalize the model if it does not asign more similariy to the positive than the negative
        loss = F.cross_entropy(logits, labels)  # Scalar loss
        return loss
    
    
    # This function implements a contrastive loss for training with triplets:
    #
    # - Rewards the model when the anchor is more similar to the positive than to the negative.
    # - Uses cosine similarity implicitly via the dot product between normalized embeddings.
    # - Applies temperature scaling to sharpen similarity differences.
    # - Uses cross-entropy loss to compare [sim(anchor, positive), sim(anchor, negative)] against the target label 0,
    #   meaning the model should consider the first input (positive) as the most similar.






# Phase 2: Train the Embedding Model ===

import torch
import torch.nn as nn
import torch.optim as optim

# --- Step 1: Instantiate the model ---
# Uses the projection head defined earlier, with input_dim equal to the SHAP vector dimension
model = ShapProjectionHead(input_dim=shap_vectors.shape[1], hidden_dim=64, output_dim=32)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Step 2: Define the contrastive loss (NT-Xent style) ---
criterion = TripletNTXentLoss(temperature=0.5)  

# --- Step 3: Define optimizer ---
# Adam is a standard optimizer for contrastive learning setups
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # Learning rate and regularization

# --- Step 4: Training Loop ---
n_epochs = 100  # Number of epochs to train
model.train()   # Set model to training mode

for epoch in range(n_epochs):
    total_loss = 0.0  # Accumulator for epoch loss

    for anchor, positive, negative in dataloader:
        # Move data to device (GPU or CPU)
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        # Zero out gradients from previous step
        optimizer.zero_grad()

        # Get embeddings using the projection head
        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)

        # Compute contrastive loss
        loss = criterion(anchor_embed, positive_embed, negative_embed)

        # Backpropagate the loss and update weights
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # Add batch loss to total

    # Print average loss per epoch
    print(f"Epoch {epoch+1}/{n_epochs} | Loss: {total_loss / len(dataloader):.4f}")

# --- Step 5: Save the trained model and embeddings ---

# Save model weights to file
torch.save(model.state_dict(), "FULL_FINE_TUNING_shap_projection_head_trained.pt")
print("Model saved as FULL_FINE_TUNING_shap_projection_head_trained.pt")

# Optional: save final SHAP embeddings for all subjects
model.eval()  # Set model to evaluation mode
all_embeddings = {}

with torch.no_grad():
    for subj_id, shap_vec in id_to_vector.items(): 
        tensor_input = torch.tensor(shap_vec, dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, dim)
        embedding = model(tensor_input).squeeze(0).cpu().numpy()  # Shape: (output_dim,)
        all_embeddings[subj_id] = embedding

# Convert to DataFrame and save
embedding_df = pd.DataFrame.from_dict(all_embeddings, orient="index")
embedding_df.index.name = "Subject_ID"
embedding_df.columns = [f"embed_{i}" for i in range(embedding_df.shape[1])]
embedding_df.to_csv("FULL_FINE_TUNING_shap_embeddings.csv")
print("SHAP embeddings saved as FULL_FINE_TUNING_shap_embeddings.csv")
