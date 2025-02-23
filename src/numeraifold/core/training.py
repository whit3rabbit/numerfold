import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr

def train_numeraifold_model(model, train_loader, val_loader, epochs=10, lr=0.001,
                             weight_decay=1e-5, device='cuda', patience=3):
    """
    Train a NumerAIFold model with a fixed training loop that includes error handling and proper tensor shape management.
    
    Args:
        model (torch.nn.Module): The NumerAIFold model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int, optional): Number of training epochs. Default is 10.
        lr (float, optional): Learning rate for the optimizer. Default is 0.001.
        weight_decay (float, optional): L2 regularization factor. Default is 1e-5.
        device (str, optional): Device to perform training on. Default is 'cuda'.
        patience (int, optional): Number of epochs to wait for improvement before early stopping.
        
    Returns:
        torch.nn.Module: The trained model with its best state loaded and training history attached.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model = None
    no_improve_epochs = 0
    history = {'train_loss': [], 'val_loss': [], 'val_corr': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                data, target = data.to(device), target.to(device)

                # Adjust target shape if it has multiple columns
                if len(target.shape) > 1 and target.shape[1] > 1:
                    target = target[:, 0]  # Use only the first target column
                target = target.float()

                optimizer.zero_grad()
                predictions, _ = model(data)

                # Ensure predictions and targets have matching shapes
                predictions = predictions.view(-1)
                target = target.view(-1)

                loss = criterion(predictions, target)
                loss.backward()

                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            except Exception as e:
                print(f"Warning in batch {batch_idx}: {str(e)}")
                continue

        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                try:
                    data, target = data.to(device), target.to(device)

                    # Adjust target shape if necessary
                    if len(target.shape) > 1 and target.shape[1] > 1:
                        target = target[:, 0]
                    target = target.float()

                    predictions, _ = model(data)
                    predictions = predictions.view(-1)
                    target = target.view(-1)

                    loss = criterion(predictions, target)
                    val_loss += loss.item()

                    val_preds.extend(predictions.cpu().numpy())
                    val_targets.extend(target.cpu().numpy())

                except Exception as e:
                    print(f"Warning in validation: {str(e)}")
                    continue

        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Calculate Pearson correlation as a performance metric
        try:
            val_corr, _ = pearsonr(val_preds, val_targets)
            if np.isnan(val_corr):
                val_corr = 0
        except Exception:
            val_corr = 0

        # Step the learning rate scheduler
        scheduler.step()

        # Save training history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_corr'].append(val_corr)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, Val Corr: {val_corr:.6f}")

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping if no improvement for 'patience' epochs
        if no_improve_epochs >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

    # Load the best model weights
    if best_model is not None:
        model.load_state_dict(best_model)

    model.history = history
    return model

def calculate_confidence_scores(model, data_loader, device='cuda'):
    """
    Calculate pLDDT-like confidence scores for the model predictions.
    
    Confidence is derived from the consistency of attention patterns across layers.
    Higher consistency (i.e., lower variance in attention) implies higher confidence.
    
    Args:
        model (torch.nn.Module): The trained NumerAIFold model.
        data_loader (DataLoader): DataLoader providing input data batches.
        device (str, optional): Device to perform computations on. Default is 'cuda'.
        
    Returns:
        tuple: Two numpy arrays containing the confidence scores and corresponding predictions.
    """
    model.eval()
    confidences = []
    predictions = []

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            preds, attentions = model(data)

            # Compute the standard deviation of attention values as a measure of variability
            attention_std = torch.stack([attn.std(dim=1).mean() for attn in attentions])
            # Compute confidence: lower variability results in higher confidence
            confidence = 1 / (1 + attention_std.mean(dim=0))

            confidences.extend(confidence.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    return np.array(confidences), np.array(predictions)
