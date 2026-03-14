# validation_text_classification.py
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def main():
    # Basic version sanity prints (useful in logs)
    print("Python validation script running...")
    print("Torch version:", torch.__version__)
    print("NumPy version:", np.__version__)

    # Device (CPU is fine for validation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Simple text-classification-like model: embedding + average + linear
    vocab_size = 5000
    embed_dim = 64
    num_classes = 2
    seq_len = 20
    batch_size = 32

    class SimpleTextClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.fc = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            # x: (batch, seq_len)
            emb = self.embedding(x)             # (batch, seq_len, embed_dim)
            pooled = emb.mean(dim=1)           # (batch, embed_dim)
            out = self.fc(pooled)              # (batch, num_classes)
            return out

    model = SimpleTextClassifier(vocab_size, embed_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic "text" data: random token ids
    torch.manual_seed(0)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    y = torch.randint(0, num_classes, (batch_size,), dtype=torch.long, device=device)

    # TensorBoard writer
    log_dir = os.path.join("runs", f"validation_{datetime.utcnow().isoformat()}")
    writer = SummaryWriter(log_dir=log_dir)

    model.train()
    initial_loss = None
    for epoch in range(3):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        print(f"Epoch {epoch} - loss: {loss_value:.4f}")
        writer.add_scalar("train/loss", loss_value, epoch)

        if initial_loss is None:
            initial_loss = loss_value

    writer.close()

    # Basic sanity checks
    assert np.isfinite(loss_value), "Final loss is not finite"
    # Allow for noise; we just want to ensure training is doing something reasonable
    assert loss_value < initial_loss * 1.2, "Loss did not improve meaningfully over epochs"

    print("Validation completed successfully.")


if __name__ == "__main__":
    main()
