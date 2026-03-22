import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_ssl_curve(log_path):
    if not os.path.exists(log_path): return
    df = pd.read_csv(log_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], 'b-', label='SSL Loss')
    best_epoch = df.loc[df['loss'].idxmin()]['epoch']
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({int(best_epoch)})')
    plt.title('SSL Pre-training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/plots/ssl_loss_curve.png', dpi=300)
    plt.close()

def plot_finetune_curves(log_path):
    if not os.path.exists(log_path): return
    df = pd.read_csv(log_path)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0,0].plot(df['epoch'], df['train_loss'], label='Train')
    axs[0,0].plot(df['epoch'], df['val_loss'], label='Val')
    axs[0,0].set_title('Loss')
    axs[0,0].legend()
    
    axs[0,1].plot(df['epoch'], df['val_acc'], color='green')
    axs[0,1].set_title('Validation Accuracy')
    
    axs[1,0].plot(df['epoch'], df['val_macro_f1'], color='purple')
    axs[1,0].set_title('Validation Macro F1')
    
    axs[1,1].plot(df['epoch'], df['lr'] if 'lr' in df.columns else [0]*len(df), color='orange')
    axs[1,1].set_title('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/finetune_curves.png', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='outputs/ssl_training_log.csv')
    args = parser.parse_args()
    os.makedirs('outputs/plots', exist_ok=True)
    plot_ssl_curve(args.log)
    plot_finetune_curves('outputs/finetune_training_log.csv')

if __name__ == '__main__':
    main()
