import os
import time
import math
import csv
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from data_utils.dataset import CervicalSSLDataset
from data_utils.augmentations import ssl_augment
from models.backbone import CerviSenseEncoder
from models.moco import MoCov3
from rich.console import Console
from rich.table import Table

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.001):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_checkpoint(state, is_best, filepath, best_filepath):
    torch.save(state, filepath)
    if is_best:
        torch.save(state, best_filepath)

def train_epoch(model, loader, optimizer, scaler, scheduler, accum_steps, device, mixed_prec):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    for i, (x1, x2) in enumerate(loader):
        x1, x2 = x1.to(device), x2.to(device)
        with torch.cuda.amp.autocast(enabled=mixed_prec):
            loss = model(x1, x2) / accum_steps
        scaler.scale(loss).backward()
        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        epoch_loss += loss.item() * accum_steps
    return epoch_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ssl_config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("WARNING: Using CPU for training. This will be extremely slow. Use Colab T4.")

    os.makedirs(config['checkpointing']['save_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['logging']['csv_log']), exist_ok=True)

    dataset = CervicalSSLDataset(config['data']['ssl_data_dir'], transform=ssl_augment)
    num_workers = config['data']['num_workers'] if os.name != 'nt' else 0
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True,
                        num_workers=num_workers, drop_last=True, pin_memory=True)

    encoder = CerviSenseEncoder(pretrained=config['model']['pretrained'], 
                                embedding_dim=config['model']['embedding_dim'])
    
    model = MoCov3(encoder, dim=config['model']['embedding_dim'], 
                   mlp_dim=config['model']['projection_mlp_dim'],
                   T=config['model']['temperature']).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=float(config['training']['base_lr']),
                                  weight_decay=float(config['training']['weight_decay']))

    epochs = config['training']['epochs']
    steps_per_epoch = len(loader) // config['hardware']['gradient_accumulation_steps']
    total_steps = epochs * steps_per_epoch
    warmup_steps = config['training']['warmup_epochs'] * steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_loss = float('inf')
    early_stop_counter = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f"Resumed from epoch {start_epoch}")

    console = Console()
    with open(config['logging']['csv_log'], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'lr', 'gpu_mem_mb'])

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        mixed_prec = config['hardware']['mixed_precision']
        accum_steps = config['hardware']['gradient_accumulation_steps']
        
        avg_loss = train_epoch(model, loader, optimizer, scaler, scheduler, accum_steps, device, mixed_prec)
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        gpu_mem = torch.cuda.memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0

        with open(config['logging']['csv_log'], 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_loss, current_lr, gpu_mem])

        table = Table(title=f"Epoch {epoch+1}/{epochs}")
        table.add_column("Loss", justify="right")
        table.add_column("Best Loss", justify="right")
        table.add_column("LR", justify="right")
        table.add_column("Time (s)", justify="right")
        
        is_best = False
        if avg_loss < float(best_loss) - 0.01:
            best_loss = avg_loss
            is_best = True
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        table.add_row(f"{avg_loss:.4f}", f"{best_loss:.4f}", f"{current_lr:.6f}", f"{epoch_time:.1f}")
        console.print(table)

        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss
        }
        
        if (epoch + 1) % config['checkpointing']['save_every_n_epochs'] == 0:
            save_checkpoint(state, is_best, 
                            os.path.join(config['checkpointing']['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'),
                            os.path.join(config['checkpointing']['save_dir'], 'best_ssl.pth'))
        elif is_best:
            torch.save(state, os.path.join(config['checkpointing']['save_dir'], 'best_ssl.pth'))
            
        if early_stop_counter >= config['training']['early_stop_patience']:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

if __name__ == '__main__':
    main()
