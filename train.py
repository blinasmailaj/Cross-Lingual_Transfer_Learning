import torch
from tqdm import tqdm

def train(config, model, train_loader, val_loader, optimizer, device):
    model.train()
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids, attention_mask, labels)
            total_loss += outputs.loss.item()
    return total_loss / len(val_loader)
