import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from data_utils.dataset import CervicalSupervisedDataset
from data_utils.augmentations import eval_augment
from evaluation.metrics import compute_all_metrics
from models.backbone import CerviSenseEncoder
from models.classifier import CerviSenseClassifier
from rich.console import Console

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/combined/supervised/test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading checkpoint {args.checkpoint}")
    encoder = CerviSenseEncoder(pretrained=False, embedding_dim=512)
    model = CerviSenseClassifier(encoder, num_classes=5, dropout=0.0).to(device)
    
    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
    else:
        print("Checkpoint not found!")
        return

    model.eval()
    test_ds = CervicalSupervisedDataset(args.data_dir, transform=eval_augment)
    nw = 0 if os.name == 'nt' else 4
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=nw)

    all_preds, all_targs, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images)
            probs = torch.softmax(preds, dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.argmax(1).cpu())
            all_targs.append(labels.cpu())

    yt = torch.cat(all_targs).numpy()
    yp = torch.cat(all_preds).numpy()
    yprob = torch.cat(all_probs).numpy()

    metrics = compute_all_metrics(yt, yp, yprob)
    
    os.makedirs('outputs/reports', exist_ok=True)
    with open('outputs/reports/evaluation_report.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    Console().print(metrics)

if __name__ == '__main__':
    main()
