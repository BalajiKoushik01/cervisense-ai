import os
import time
import csv
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils.dataset import CervicalSupervisedDataset
from data_utils.augmentations import finetune_train_augment, eval_augment
from models.backbone import CerviSenseEncoder
from models.classifier import CerviSenseClassifier
from training.losses import FocalLoss
from rich.console import Console
from rich.table import Table

def load_encoder(checkpoint_path, device):
    encoder = CerviSenseEncoder(pretrained=False, embedding_dim=512)
    if os.path.exists(checkpoint_path):
        chkpt = torch.load(checkpoint_path, map_location=device)['model']
        enc_state = {k.replace('encoder_q.', ''): v for k, v in chkpt.items() if k.startswith('encoder_q.')}
        encoder.load_state_dict(enc_state)
    else:
        print("WARNING: checkpoint not found. using random init.")
    return encoder

def train_epoch(model, train_loader, criterion, optimizer, scaler, accum_steps, device, mixed_prec):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=mixed_prec):
            preds = model(images)
            loss = criterion(preds, labels) / accum_steps
            
        scaler.scale(loss).backward()
        
        if (i+1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        train_loss += loss.item() * accum_steps
        
    return train_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device, mixed_prec):
    from sklearn.metrics import f1_score, accuracy_score
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=mixed_prec):
                preds = model(images)
                loss = criterion(preds, labels)
            val_loss += loss.item()
            all_preds.append(preds.argmax(dim=1).cpu())
            all_targets.append(labels.cpu())
            
    val_loss /= len(val_loader)
    
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    val_acc = accuracy_score(y_true, y_pred)
    val_macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    return val_loss, val_acc, val_macro_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/finetune_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['checkpointing']['save_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['logging']['csv_log']), exist_ok=True)

    train_ds = CervicalSupervisedDataset(os.path.join(config['data']['data_dir'], 'train'), transform=finetune_train_augment)
    val_ds = CervicalSupervisedDataset(os.path.join(config['data']['data_dir'], 'val'), transform=eval_augment)
    
    num_workers = config['data']['num_workers'] if os.name != 'nt' else 0
    sampler = train_ds.get_weighted_sampler() if config['data']['use_weighted_sampler'] else None
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], 
                              sampler=sampler, shuffle=(sampler is None),
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=num_workers)

    encoder = load_encoder(config['model']['checkpoint'], device)
    model = CerviSenseClassifier(encoder, num_classes=config['model']['num_classes'], dropout=config['model']['dropout']).to(device)

    if config['loss']['type'] == 'focal':
        criterion = FocalLoss(alpha=config['loss']['alpha'], gamma=config['loss']['gamma'])
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config['training']['label_smoothing'])

    base_lr = float(config['training']['base_lr'])
    wd = float(config['training']['weight_decay'])
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': base_lr * 0.1},
        {'params': model.classifier.parameters(), 'lr': base_lr}
    ], lr=base_lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    scaler = torch.cuda.amp.GradScaler()

    epochs = config['training']['epochs']
    freeze_epochs = config['training']['freeze_encoder_epochs']
    early_stop_patience = config['training']['early_stop_patience']

    best_macro_f1 = 0.0
    early_stop_counter = 0

    with open(config['logging']['csv_log'], 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_macro_f1'])

    console = Console()
    
    for epoch in range(epochs):
        if epoch < freeze_epochs:
            for p in model.encoder.parameters(): p.requires_grad = False
        elif epoch == freeze_epochs:
            for p in model.encoder.parameters(): p.requires_grad = True
            
        mixed_prec = config['hardware'].get('mixed_precision', True)
        accum_steps = config['hardware'].get('gradient_accumulation_steps', 1)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, accum_steps, device, mixed_prec)
        val_loss, val_acc, val_macro_f1 = validate_epoch(model, val_loader, criterion, device, mixed_prec)
        
        scheduler.step(val_macro_f1)

        with open(config['logging']['csv_log'], 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, train_loss, val_loss, val_acc, val_macro_f1])

        table = Table(title=f"Epoch {epoch+1}/{epochs}")
        table.add_column("Train Loss", justify="right")
        table.add_column("Val Loss", justify="right")
        table.add_column("Val Acc", justify="right")
        table.add_column("Val F1", justify="right")
        table.add_column("Best F1", justify="right")
        
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(config['checkpointing']['save_dir'], 'best_finetune.pth'))
        else:
            early_stop_counter += 1

        table.add_row(f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{val_macro_f1:.4f}", f"{best_macro_f1:.4f}")
        console.print(table)
            
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

if __name__ == '__main__':
    main()
