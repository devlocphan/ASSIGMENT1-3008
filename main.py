import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ==================== HYPERPARAMETERS (OPTIMIZED FOR SPEED) ====================
BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 15
SUBSET_SIZE = 0.3

# ==================== PATH SETUP ====================
def get_cifar10_path():
    """Auto-detect path for Google Colab or GitHub"""
    colab_path = "/content/drive/My Drive/AS1/cifar-10-batches-py"
    if os.path.exists(colab_path):
        print(f"✓ Found: Google Drive\n")
        return colab_path
    
    github_path = "./cifar-10-batches-py"
    if os.path.exists(github_path):
        print(f"✓ Found: GitHub\n")
        return github_path
    
    raise FileNotFoundError("CIFAR-10 data not found.")

CIFAR10_PATH = get_cifar10_path()

# ==================== 1. LOAD CIFAR-10 ====================
def unpickle(file):
    """Load pickled CIFAR-10 file"""
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def load_cifar10_local():
    """Load CIFAR-10 dataset"""
    train_data = []
    train_labels = []
    
    for i in range(1, 6):
        batch = unpickle(f"{CIFAR10_PATH}/data_batch_{i}")
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.array(train_labels)
    
    test_batch = unpickle(f"{CIFAR10_PATH}/test_batch")
    test_data = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])
    
    # Use subset for faster training
    subset_idx = int(len(train_data) * SUBSET_SIZE)
    train_data = train_data[:subset_idx]
    train_labels = train_labels[:subset_idx]
    
    test_subset = int(len(test_data) * SUBSET_SIZE)
    test_data = test_data[:test_subset]
    test_labels = test_labels[:test_subset]
    
    print(f"Train: {train_data.shape} | Test: {test_data.shape}\n")
    return train_data, train_labels, test_data, test_labels

def prepare_dataloaders(train_data, train_labels, test_data, test_labels, batch_size=BATCH_SIZE):
    """Prepare data loaders with randomized train/val split"""
    train_data = train_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_data = test_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    
    train_data = (train_data - 0.5) / 0.5
    test_data = (test_data - 0.5) / 0.5
    
    train_data = torch.from_numpy(train_data)
    train_labels = torch.from_numpy(train_labels).long()
    test_data = torch.from_numpy(test_data)
    test_labels = torch.from_numpy(test_labels).long()
    
    # Create full train dataset
    full_train_dataset = TensorDataset(train_data, train_labels)
    
    # Randomized split with fixed seed (reproducible)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0)
    )
    
    test_dataset = TensorDataset(test_data, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# ==================== 2. MODELS ====================
def build_cnn(depth=2):
    """Build lightweight CNN with careful depth handling"""
    layers = []
    in_ch, out_ch = 3, 32
    pool_count = 0  # Track pooling operations
    max_pools = 3   # Maximum pooling operations (32→16→8→4)
    
    for i in range(depth):
        layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # Pool less frequently for deeper networks
        # Pool at layers: 2, 4, 8 (for depth 16)
        should_pool = (
            (i + 1) % 2 == 0 and           # Every 2 layers
            pool_count < max_pools and     # But max 3 times
            (i + 1) < depth - 2            # Not near end
        )
        
        if should_pool:
            layers.append(nn.MaxPool2d(2, 2))
            pool_count += 1
        
        in_ch = out_ch
        if i % 4 == 1:
            out_ch = min(out_ch * 2, 256)
    
    # Add global average pooling at the end (academic + safe)
    layers.append(nn.AdaptiveAvgPool2d(1))
    
    class CNN(nn.Module):
        def __init__(self, conv_layers):
            super().__init__()
            self.conv = nn.Sequential(*conv_layers)
            self.flat_size = None
            self.fc = None
        
        def _init_fc(self, x):
            """Initialize FC layer on first forward pass"""
            if self.fc is None:
                with torch.no_grad():
                    x_conv = self.conv(x)
                    self.flat_size = x_conv.view(x_conv.size(0), -1).size(1)
                    self.fc = nn.Sequential(
                        nn.Linear(self.flat_size, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)
                    ).to(device)
        
        def forward(self, x):
            self._init_fc(x)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    return CNN(layers).to(device)

class SimpleCNN(nn.Module):
    """Simple 2-layer CNN"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ==================== 3. TRAINING ====================
def train_epoch(model, train_loader, criterion, optimizer):
    """Train one epoch"""
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, data_loader, criterion):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(data_loader), correct / total

def train_model(model, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """Train and evaluate model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses, val_accs = [], [], []
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.3f} | Val: {val_loss:.3f} | Acc: {val_acc:.3f}")
    
    training_time = time.time() - start_time
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    print(f"  Test Acc: {test_acc:.4f} | Time: {training_time:.0f}s\n")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'train_time': training_time
    }

# ==================== EXPERIMENTS ====================
def experiment_depth_analysis(train_loader, val_loader, test_loader):
    """Experiment A: CNN Depth"""
    print("\n[Experiment A] CNN Depth Analysis")
    print("-" * 50)
    
    depths = [2, 8, 16, 32]
    results = {}
    
    for depth in depths:
        print(f"Depth {depth}:")
        model = build_cnn(depth=depth)
        result = train_model(model, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE)
        results[depth] = result
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    ax = axes[0, 0]
    for d in depths:
        ax.plot(results[d]['train_losses'], label=f'Depth {d}', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for d in depths:
        ax.plot(results[d]['val_losses'], label=f'Depth {d}', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for d in depths:
        ax.plot(results[d]['val_accs'], label=f'Depth {d}', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    test_accs = [results[d]['test_acc'] for d in depths]
    ax.bar([str(d) for d in depths], test_accs, color='steelblue', alpha=0.7)
    ax.set_ylabel('Test Accuracy')
    ax.set_xlabel('Depth')
    ax.set_title('Final Test Accuracy')
    ax.grid(True, axis='y', alpha=0.3)
    for i, acc in enumerate(test_accs):
        ax.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('experiment_a_depth.png', dpi=100)
    print("✓ Saved: experiment_a_depth.png")
    
    print(f"\n{'Depth':<8} {'Test Acc':<10} {'Time (s)':<10}")
    for d in depths:
        print(f"{d:<8} {results[d]['test_acc']:.4f}     {results[d]['train_time']:.0f}")

def experiment_learning_rate_analysis(train_loader, val_loader, test_loader):
    """Experiment B: Learning Rate"""
    print("\n[Experiment B] Learning Rate Analysis")
    print("-" * 50)
    
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    results = {}
    
    for lr in learning_rates:
        print(f"LR {lr}:")
        model = SimpleCNN().to(device)
        result = train_model(model, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=lr)
        results[lr] = result
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    ax = axes[0, 0]
    for lr in learning_rates:
        ax.plot(results[lr]['train_losses'], label=f'LR={lr}', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for lr in learning_rates:
        ax.plot(results[lr]['val_losses'], label=f'LR={lr}', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for lr in learning_rates:
        ax.plot(results[lr]['val_accs'], label=f'LR={lr}', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    test_accs = [results[lr]['test_acc'] for lr in learning_rates]
    ax.bar([str(lr) for lr in learning_rates], test_accs, color='coral', alpha=0.7)
    ax.set_ylabel('Test Accuracy')
    ax.set_xlabel('Learning Rate')
    ax.set_title('Final Test Accuracy')
    ax.grid(True, axis='y', alpha=0.3)
    for i, acc in enumerate(test_accs):
        ax.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('experiment_b_lr.png', dpi=100)
    print("✓ Saved: experiment_b_lr.png")
    
    print(f"\n{'LR':<10} {'Test Acc':<10} {'Time (s)':<10}")
    for lr in learning_rates:
        print(f"{lr:<10} {results[lr]['test_acc']:.4f}     {results[lr]['train_time']:.0f}")

def experiment_batch_size_study(train_data, train_labels, test_data, test_labels):
    """Experiment C: Batch Size"""
    print("\n[Experiment C] Batch Size Study")
    print("-" * 50)
    
    batch_sizes = [32, 64, 128, 256]
    results = {}
    
    for bs in batch_sizes:
        print(f"Batch Size {bs}:")
        train_loader, val_loader, test_loader = prepare_dataloaders(
            train_data, train_labels, test_data, test_labels, batch_size=bs
        )
        model = SimpleCNN().to(device)
        result = train_model(model, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE)
        results[bs] = result
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    ax = axes[0, 0]
    for bs in batch_sizes:
        ax.plot(results[bs]['train_losses'], label=f'BS={bs}', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for bs in batch_sizes:
        ax.plot(results[bs]['val_losses'], label=f'BS={bs}', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for bs in batch_sizes:
        ax.plot(results[bs]['val_accs'], label=f'BS={bs}', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    test_accs = [results[bs]['test_acc'] for bs in batch_sizes]
    ax.bar([str(bs) for bs in batch_sizes], test_accs, color='lightgreen', alpha=0.7)
    ax.set_ylabel('Test Accuracy')
    ax.set_xlabel('Batch Size')
    ax.set_title('Final Test Accuracy')
    ax.grid(True, axis='y', alpha=0.3)
    for i, acc in enumerate(test_accs):
        ax.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('experiment_c_bs.png', dpi=100)
    print("✓ Saved: experiment_c_bs.png")
    
    print(f"\n{'Batch Size':<12} {'Test Acc':<10} {'Time (s)':<10}")
    for bs in batch_sizes:
        print(f"{bs:<12} {results[bs]['test_acc']:.4f}     {results[bs]['train_time']:.0f}")

# ==================== MAIN ====================
if __name__ == "__main__":
    print(f"Loading CIFAR-10 ({SUBSET_SIZE*100:.0f}% subset)...")
    train_data, train_labels, test_data, test_labels = load_cifar10_local()
    
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_data, train_labels, test_data, test_labels, batch_size=BATCH_SIZE
    )
    
    while True:
        print("Select: (a) Depth (b) Learning Rate (c) Batch Size (q) Quit")
        choice = input("Choice: ").strip().lower()
        
        if choice == 'a':
            experiment_depth_analysis(train_loader, val_loader, test_loader)
        elif choice == 'b':
            experiment_learning_rate_analysis(train_loader, val_loader, test_loader)
        elif choice == 'c':
            experiment_batch_size_study(train_data, train_labels, test_data, test_labels)
        elif choice == 'q':
            print("Done!")
            break