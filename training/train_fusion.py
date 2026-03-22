import os
import time
import csv
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils.dataset import DualDomainDataset
from data_utils.augmentations import finetune_train_augment, eval_augment
from models.backbone import CerviSenseEncoder
from models.fusion import HCMAF
from training.losses import FocalLoss
from rich.console import Console
from rich.table import Table

def load_encoder(checkpoint_path, device):
    encoder = CerviSenseEncoder(pretrained=False, embedding_dim=512)
    current_state = encoder.state_dict()
    if os.path.exists(checkpoint_path):
        chkpt = torch.load(checkpoint_path, map_location=device)
        if 'encoder.features.0.0.weight' in chkpt:
            enc_state = {k.replace('encoder.', ''): v for k, v in chkpt.items() if k.startswith('encoder.')}
        elif 'model' in chkpt: 
            chkpt_model = chkpt['model']
            enc_state = {k.replace('encoder_q.', ''): v for k, v in chkpt_model.items() if k.startswith('encoder_q.')}
        else:
            enc_state = chkpt
        
        valid_state = {k: v for k, v in enc_state.items() if k in current_state}
        current_state.update(valid_state)
        encoder.load_state_dict(current_state)
    else:
        print(f"WARNING: checkpoint {checkpoint_path} not found. using random init.")
    return encoder

def train_epoch(model, train_loader, criterion, optimizer, scaler, accum, device, mixed_prec):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    for i, (colpo, histo, labels) in enumerate(train_loader):
        colpo, histo, labels = colpo.to(device), histo.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=mixed_prec):
            preds, _ = model(colpo, histo)
            loss = criterion(preds, labels) / accum
        scaler.scale(loss).backward()
        if (i+1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        train_loss += loss.item() * accum
    return train_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device, mixed_prec):
    from sklearn.metrics import f1_score, accuracy_score
    model.eval()
    val_loss = 0.0
    all_preds, all_targs = [], []
    with torch.no_grad():
        for c, h, labels in val_loader:
            c, h, labels = c.to(device), h.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=mixed_prec):
                preds, _ = model(c, h)
                loss = criterion(preds, labels)
            val_loss += loss.item()
            all_preds.append(preds.argmax(1).cpu())
            all_targs.append(labels.cpu())
    val_loss /= len(val_loader)
    yt = torch.cat(all_targs).numpy()
    yp = torch.cat(all_preds).numpy()
    vacc = accuracy_score(yt, yp)
    vf1 = f1_score(yt, yp, average='macro')
    return val_loss, vacc, vf1

def adjust_encoder_freezing(model, epoch, freeze_eps):
    if epoch < freeze_eps:
        for p in model.encoder_colpo.parameters(): p.requires_grad = False
        for p in model.encoder_histo.parameters(): p.requires_grad = False
    elif epoch == freeze_eps:
        for p in model.encoder_colpo.parameters(): p.requires_grad = True
        for p in model.encoder_histo.parameters(): p.requires_grad = True

def run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, config, device, console):
    epochs = config['training']['epochs']
    freeze_eps = config['training']['freeze_encoder_epochs']
    patience = config['training']['early_stop_patience']
    best_macro_f1 = 0.0
    early_stop_counter = 0

    with open(config['logging']['csv_log'], 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_macro_f1'])

    for epoch in range(epochs):
        adjust_encoder_freezing(model, epoch, freeze_eps)
            
        mixed_prec = config['hardware'].get('mixed_precision', True)
        accum = config['hardware'].get('gradient_accumulation_steps', 1)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, accum, device, mixed_prec)
        val_loss, vacc, vf1 = validate_epoch(model, val_loader, criterion, device, mixed_prec)
        
        scheduler.step(vf1)
        with open(config['logging']['csv_log'], 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, train_loss, val_loss, vacc, vf1])

        table = Table(title=f"Epoch {epoch+1}/{epochs}")
        table.add_column("Train Loss", justify="right")
        table.add_column("Val Loss", justify="right")
        table.add_column("Val F1", justify="right")
        table.add_column("Best F1", justify="right")
        
        if vf1 > best_macro_f1:
            best_macro_f1 = vf1
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(config['checkpointing']['save_dir'], 'best_fusion.pth'))
        else:
            early_stop_counter += 1

        table.add_row(f"{train_loss:.4f}", f"{val_loss:.4f}", f"{vf1:.4f}", f"{best_macro_f1:.4f}")
        console.print(table)
            
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/fusion_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config['checkpointing']['save_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['logging']['csv_log']), exist_ok=True)

    colpo_dir = config['data']['colpo_dir']
    histo_dir = config['data']['histo_dir']
    
    train_ds = DualDomainDataset(colpo_dir, histo_dir, transform=finetune_train_augment)
    val_ds = DualDomainDataset(colpo_dir, histo_dir, transform=eval_augment)
    
    nw = config['data']['num_workers'] if os.name != 'nt' else 0
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=nw, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=nw)

    encoder_colpo = load_encoder(config['model']['colpo_checkpoint'], device)
    encoder_histo = load_encoder(config['model']['histo_checkpoint'], device)

    model = HCMAF(encoder_colpo, encoder_histo, num_classes=config['model']['num_classes'], dim=config['model']['feature_dim']).to(device)

    if config['loss']['type'] == 'focal':
        criterion = FocalLoss(alpha=config['loss']['alpha'], gamma=config['loss']['gamma'])
    else:
        criterion = nn.CrossEntropyLoss()

    enc_lr = float(config['training']['encoder_lr'])
    fus_lr = float(config['training']['fusion_lr'])
    weight_decay = float(config['training'].get('weight_decay', 1e-4))
    
    optimizer = torch.optim.AdamW([
        {'params': model.encoder_colpo.parameters(), 'lr': enc_lr},
        {'params': model.encoder_histo.parameters(), 'lr': enc_lr},
        {'params': model.self_attn_colpo.parameters(), 'lr': fus_lr},
        {'params': model.self_attn_histo.parameters(), 'lr': fus_lr},
        {'params': model.cross_attn_colpo2histo.parameters(), 'lr': fus_lr},
        {'params': model.cross_attn_histo2colpo.parameters(), 'lr': fus_lr},
        {'params': model.gate.parameters(), 'lr': fus_lr},
        {'params': model.classifier.parameters(), 'lr': fus_lr}
    ], lr=fus_lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    scaler = torch.cuda.amp.GradScaler()
    console = Console()
    
    run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, config, device, console)

if __name__ == '__main__':
    main()
