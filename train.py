import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import wandb
from tqdm import tqdm
import time
from collections import defaultdict
import json

#  wandb offline mode
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

# Import spatial-aware BYOL
from spatial_byol import SpatialBYOL, spatial_byol_loss

class OCTDataset(Dataset):
    """Dataset for OCT images."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        class_dirs = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        
        for class_dir in class_dirs:
            class_path = os.path.join(root_dir, class_dir)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        
        print(f"Found {len(self.image_paths)} OCT images")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        class_counts = defaultdict(int)
        for path in self.image_paths:
            class_name = os.path.basename(os.path.dirname(path))
            class_counts[class_name] += 1
        print("Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count:,}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            return self.transform(image), self.transform(image)
        return image, image

class OCTAugmentations:
    """No need to creat instance of this class."""
    @staticmethod
    def get_byol_transforms(image_size=224):
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class SpatialBYOLTrainer:
    """
    Trainer for Spatial-Aware BYOL
    """
    def __init__(
        self,
        model,
        train_loader,
        device,
        learning_rate=3e-4,
        weight_decay=1e-4,
        accumulation_steps=4,
        mixed_precision=True,
        spatial_loss_weight=0.5,
        model_type='spatial',  # 'spatial'
        log_wandb=False,
        save_dir='./checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.save_dir = save_dir
        self.log_wandb = log_wandb
        self.accumulation_steps = accumulation_steps
        self.mixed_precision = mixed_precision
        self.spatial_loss_weight = spatial_loss_weight
        self.model_type = model_type
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=learning_rate * 0.01
        )
        
        # Track loss components separately
        self.train_losses = {'total': [], 'global': [], 'spatial': []}
        self.learning_rates = []
        self.epoch_times = []
        
        effective_batch = train_loader.batch_size * accumulation_steps
        print(f"Trainer configured:")
        print(f"  Model type: {model_type}")
        print(f"  Effective batch: {effective_batch}")
        print(f"  Spatial loss weight: {spatial_loss_weight}")
        print(f"  Mixed precision: {mixed_precision}")
        
        if self.log_wandb:
            try:
                wandb.init(
                    project="",
                    config={
                        "model": f"Spatial-BYOL-{model_type}",
                        "learning_rate": learning_rate,
                        "effective_batch": effective_batch,
                        "spatial_weight": spatial_loss_weight,
                        "total_images": len(train_loader.dataset)
                    },
                    mode="offline"
                )
                print("Wandb initialized (offline)")
            except:
                self.log_wandb = False
    
    def train_epoch(self, epoch):
        """Training epoch with spatial loss monitoring."""
        self.model.train()
        
        # Track loss
        epoch_losses = {'total': 0.0, 'global': 0.0, 'spatial': 0.0}
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=num_batches,
            desc=f"Epoch {epoch+1}"
        )
        
        epoch_start = time.time()
        self.optimizer.zero_grad()
        
        for batch_idx, (view1, view2) in progress_bar:
            view1 = view1.to(self.device, non_blocking=True)
            view2 = view2.to(self.device, non_blocking=True)
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    online_out, target_out = self.model(view1, view2)
                    
                    # 
                    if self.model_type == 'spatial':
                        loss, loss_dict = spatial_byol_loss(
                            online_out, target_out, 
                            spatial_weight=self.spatial_loss_weight
                        )
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                # loss 
                for key in loss_dict:
                    epoch_losses[key] += loss_dict[key] / self.accumulation_steps
            else:
                online_out, target_out = self.model(view1, view2)
                
                if self.model_type == 'spatial':
                    loss, loss_dict = spatial_byol_loss(
                        online_out, target_out,
                        spatial_weight=self.spatial_loss_weight
                    )
                
                loss = loss / self.accumulation_steps
                loss.backward()
                
                for key in loss_dict:
                    epoch_losses[key] += loss_dict[key] / self.accumulation_steps
            
            # Optimizer step
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.model.update_target_network()
            
            # Update progress
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.accumulation_steps:.4f}",
                'global': f"{loss_dict['global']:.4f}",
                'spatial': f"{loss_dict.get('spatial', loss_dict.get('pixel', 0)):.4f}",
                'lr': f"{current_lr:.6f}"
            })
            
            # Log to wandb
            if self.log_wandb and batch_idx % 100 == 0:
                log_dict = {
                    "batch_loss": loss.item() * self.accumulation_steps,
                    "batch_global_loss": loss_dict['global'],
                    "batch_spatial_loss": loss_dict.get('spatial', loss_dict.get('pixel', 0)),
                    "learning_rate": current_lr,
                    "epoch": epoch
                }
                wandb.log(log_dict)
        
        # Calculate epoch averages
        num_updates = num_batches // self.accumulation_steps
        for key in epoch_losses:
            epoch_losses[key] = (epoch_losses[key] * self.accumulation_steps) / num_updates
        
        epoch_time = time.time() - epoch_start
        
        # Store metrics
        for key in self.train_losses:
            self.train_losses[key].append(epoch_losses[key])
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        self.epoch_times.append(epoch_time)
        
        self.scheduler.step()
        
        print(f"Epoch {epoch+1} - Total: {epoch_losses['total']:.4f}, " + 
              f"Global: {epoch_losses['global']:.4f}, " +
              f"Spatial: {epoch_losses['spatial']:.4f}, " +
              f"Time: {epoch_time:.2f}s")
        
        return epoch_losses
    
    def train(self, num_epochs):
        """Full training loop."""
        print(f"\nStarting Spatial-BYOL training for {num_epochs} epochs")
        print(f"Model type: {self.model_type}")
        print(f"Spatial loss weight: {self.spatial_loss_weight}")
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_losses = self.train_epoch(epoch)
            
            if self.log_wandb:
                wandb.log({
                    "epoch_total_loss": epoch_losses['total'],
                    "epoch_global_loss": epoch_losses['global'],
                    "epoch_spatial_loss": epoch_losses['spatial'],
                    "epoch": epoch + 1
                })
            
            # Save best model
            if epoch_losses['total'] < best_loss:
                best_loss = epoch_losses['total']
                self.save_checkpoint(epoch, epoch_losses, is_best=True)
            
            # Regular checkpoints
            if (epoch + 1) % 25 == 0:
                self.save_checkpoint(epoch, epoch_losses)
                self.plot_training_curves()
        
        print(f"\nTraining completed! Best loss: {best_loss:.4f}")
        self.save_checkpoint(num_epochs-1, epoch_losses, final=True)
        self.plot_training_curves()
        
        if self.log_wandb:
            wandb.finish()
    
    def save_checkpoint(self, epoch, losses, is_best=False, final=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': losses,
            'train_losses': self.train_losses
        }
        
        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            print(f"Best model saved at epoch {epoch+1}")
        
        if final:
            path = os.path.join(self.save_dir, 'final_model.pth')
            torch.save(checkpoint, path)
        
        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, path)
    
    def plot_training_curves(self):
        """Plot training curves with loss components."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.train_losses['total']) + 1)
        
        # Total loss
        axes[0, 0].plot(epochs, self.train_losses['total'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss components
        axes[0, 1].plot(epochs, self.train_losses['global'], 'r-', linewidth=2, label='Global')
        axes[0, 1].plot(epochs, self.train_losses['spatial'], 'g-', linewidth=2, label='Spatial')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.learning_rates, 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss ratio (spatial/global)
        ratio = [s/g if g > 0 else 0 for s, g in zip(self.train_losses['spatial'], self.train_losses['global'])]
        axes[1, 1].plot(epochs, ratio, 'orange', linewidth=2)
        axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Spatial/Global Ratio')
        axes[1, 1].set_title('Loss Component Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300)
        plt.close()

def main():
    """Main training function."""
    
    config = {
        'data_dir': './oct_data',
        'batch_size': 64,
        'accumulation_steps': 4,
        'num_workers': 1,
        'num_epochs': 200,
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './oct_spatial_byol_checkpoints',
        'log_wandb': False,
        'mixed_precision': True,
        'model_type': 'spatial',
        'spatial_loss_weight': 0.5,  # can tune
        'projection_dim': 256,
        'spatial_projection_dim': 128,
    }
    
    print("=" * 60)
    print("OCT Spatial-Aware BYOL Training")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Model: {config['model_type']}")
    print(f"Spatial weight: {config['spatial_loss_weight']}")
    print(f"Data: {config['data_dir']}")
    
    # Create dataset
    print("\nLoading dataset...")
    transforms = OCTAugmentations.get_byol_transforms(224)
    train_dataset = OCTDataset(config['data_dir'], transform=transforms)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False,
        drop_last=True,
        prefetch_factor=2
    )
    
    print(f"Dataset ready: {len(train_dataset):,} images")
    print(f"Batches per epoch: {len(train_loader):,}")
    
    # Create model
    print(f"\nBuilding {config['model_type']} model...")
    
    if config['model_type'] == 'spatial':
        model = SpatialBYOL(
            encoder_arch='resnet50',
            pretrained=True,
            projection_dim=config['projection_dim'],
            spatial_projection_dim=config['spatial_projection_dim'],
            spatial_loss_weight=config['spatial_loss_weight']
        )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params:,} parameters")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = SpatialBYOLTrainer(
        model=model,
        train_loader=train_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        accumulation_steps=config['accumulation_steps'],
        mixed_precision=config['mixed_precision'],
        spatial_loss_weight=config['spatial_loss_weight'],
        model_type=config['model_type'],
        log_wandb=config['log_wandb'],
        save_dir=config['save_dir']
    )
    
    # Start training
    print("\n Starting training...")
    trainer.train(config['num_epochs'])
    
    print("\nTraining complete!")
    print(f"Results saved to: {config['save_dir']}")

if __name__ == "__main__":
    main()