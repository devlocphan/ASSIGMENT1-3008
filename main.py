import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30

# Path to CIFAR-10 local data
CIFAR10_PATH = r"D:\GRIFFITH\TRI1_2026\3008ICT DEEP LEARNING\AS1\cifar-10-batches-py"

# ==================== 1. LOAD LOCAL CIFAR-10 ====================
def unpickle(file):
    """Load pickled CIFAR-10 file"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_local():
    """Load CIFAR-10 from local directory"""
    train_data = []
    train_labels = []
    
    for i in range(1, 6):
        batch_path = f"{CIFAR10_PATH}/data_batch_{i}"
        batch = unpickle(batch_path)
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.array(train_labels)
    
    test_batch = unpickle(f"{CIFAR10_PATH}/test_batch")
    test_data = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])
    
    meta = unpickle(f"{CIFAR10_PATH}/batches.meta")
    class_names = [name.decode('utf-8') for name in meta[b'label_names']]
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    return train_data, train_labels, test_data, test_labels, class_names

def prepare_dataloaders(train_data, train_labels, test_data, test_labels, batch_size=BATCH_SIZE):
    """Convert to PyTorch tensors and create DataLoaders"""
    
    train_data = train_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_data = test_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    
    train_data = (train_data - 0.5) / 0.5
    test_data = (test_data - 0.5) / 0.5
    
    train_data = torch.from_numpy(train_data)
    train_labels = torch.from_numpy(train_labels).long()
    test_data = torch.from_numpy(test_data)
    test_labels = torch.from_numpy(test_labels).long()
    
    train_size = int(0.7 * len(train_data))
    val_size = int(0.15 * len(train_data))
    
    train_dataset = TensorDataset(train_data[:train_size], train_labels[:train_size])
    val_dataset = TensorDataset(train_data[train_size:train_size+val_size], train_labels[train_size:train_size+val_size])
    test_dataset = TensorDataset(test_data, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# ==================== 2. CNN MODELS WITH VARIABLE DEPTH ====================
def build_cnn(depth=2, use_residual=False, use_dropout=False, dropout_rate=0.3):
    """Build CNN with configurable depth"""
    layers = []
    in_channels = 3
    out_channels = 32
    
    # Build convolutional layers
    for i in range(depth):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_rate))
        
        # Add pooling every 2 layers
        if (i + 1) % 2 == 0:
            layers.append(nn.MaxPool2d(2, 2))
        
        in_channels = out_channels
        if i % 4 == 1:  # Increase channels periodically
            out_channels = min(out_channels * 2, 256)
    
    class DynamicCNN(nn.Module):
        def __init__(self, conv_layers, in_ch, use_res):
            super(DynamicCNN, self).__init__()
            self.conv_layers = nn.Sequential(*conv_layers)
            self.use_residual = use_res
            
            # Calculate flattened size
            self.flat_size = self._get_flat_size()
            self.fc1 = nn.Linear(self.flat_size, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def _get_flat_size(self):
            x = torch.randn(1, 3, 32, 32).to(device)
            x = self.conv_layers(x)
            return x.view(x.size(0), -1).size(1)
        
        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    return DynamicCNN(layers, in_channels, use_residual)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ==================== 3. TRAINING & EVALUATION ====================
def train_epoch(model, train_loader, criterion, optimizer):
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
    
    accuracy = correct / total
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_losses, val_losses, val_accs = [], [], []
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    training_time = time.time() - start_time
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    print(f"Test Accuracy: {test_acc:.4f}, Training Time: {training_time:.2f}s\n")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'train_time': training_time
    }

# ==================== EXPERIMENT (a): CNN DEPTH ANALYSIS ====================
def experiment_depth_analysis(train_loader, val_loader, test_loader):
    """Train CNNs with different depths: 2, 8, 16, 32"""
    print("\n" + "="*60)
    print("EXPERIMENT (a): CNN DEPTH ANALYSIS")
    print("="*60)
    
    depths = [2, 8, 16, 32]
    results_depth = {}
    
    for depth in depths:
        print(f"\n--- Training CNN with {depth} convolutional layers ---")
        model = build_cnn(depth=depth).to(device)
        results = train_model(model, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE)
        results_depth[depth] = results
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for depth in depths:
        ax.plot(results_depth[depth]['train_losses'], label=f'Depth {depth}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss vs Epoch (Different Depths)')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for depth in depths:
        ax.plot(results_depth[depth]['val_losses'], label=f'Depth {depth}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Epoch (Different Depths)')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Validation Accuracy
    ax = axes[1, 0]
    for depth in depths:
        ax.plot(results_depth[depth]['val_accs'], label=f'Depth {depth}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy vs Epoch (Different Depths)')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Test Accuracy Comparison
    ax = axes[1, 1]
    test_accs = [results_depth[d]['test_acc'] for d in depths]
    ax.bar([str(d) for d in depths], test_accs, color='steelblue')
    ax.set_xlabel('CNN Depth')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Final Test Accuracy vs Depth')
    ax.grid(True, axis='y')
    
    for i, acc in enumerate(test_accs):
        ax.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('experiment_a_depth_analysis.png', dpi=150)
    print("\n✓ Saved: experiment_a_depth_analysis.png")
    
    # Summary table
    print("\n" + "="*60)
    print("DEPTH ANALYSIS SUMMARY")
    print("="*60)
    print(f"{'Depth':<10} {'Test Acc':<12} {'Train Time (s)':<15}")
    print("-"*60)
    for depth in depths:
        print(f"{depth:<10} {results_depth[depth]['test_acc']:.4f}      {results_depth[depth]['train_time']:.2f}")

# ==================== EXPERIMENT (b): LEARNING RATE ANALYSIS ====================
def experiment_learning_rate_analysis(train_loader, val_loader, test_loader):
    """Train CNN with different learning rates"""
    print("\n" + "="*60)
    print("EXPERIMENT (b): LEARNING RATE ANALYSIS")
    print("="*60)
    
    learning_rates = [0.000001, 0.0001, 0.001, 0.01, 1.0]
    results_lr = {}
    
    for lr in learning_rates:
        print(f"\n--- Training with Learning Rate: {lr} ---")
        model = SimpleCNN().to(device)
        results = train_model(model, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=lr)
        results_lr[lr] = results
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for lr in learning_rates:
        ax.plot(results_lr[lr]['train_losses'], label=f'LR={lr}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss vs Epoch (Different LR)')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for lr in learning_rates:
        ax.plot(results_lr[lr]['val_losses'], label=f'LR={lr}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Epoch (Different LR)')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Validation Accuracy
    ax = axes[1, 0]
    for lr in learning_rates:
        ax.plot(results_lr[lr]['val_accs'], label=f'LR={lr}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy vs Epoch (Different LR)')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Test Accuracy Comparison
    ax = axes[1, 1]
    test_accs = [results_lr[lr]['test_acc'] for lr in learning_rates]
    ax.bar([str(lr) for lr in learning_rates], test_accs, color='coral')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Final Test Accuracy vs Learning Rate')
    ax.grid(True, axis='y')
    
    for i, acc in enumerate(test_accs):
        ax.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('experiment_b_learning_rate_analysis.png', dpi=150)
    print("\n✓ Saved: experiment_b_learning_rate_analysis.png")
    
    # Summary table
    print("\n" + "="*60)
    print("LEARNING RATE ANALYSIS SUMMARY")
    print("="*60)
    print(f"{'LR':<15} {'Test Acc':<12} {'Train Time (s)':<15}")
    print("-"*60)
    for lr in learning_rates:
        print(f"{lr:<15} {results_lr[lr]['test_acc']:.4f}      {results_lr[lr]['train_time']:.2f}")

# ==================== EXPERIMENT (c): MINI-BATCH SIZE STUDY ====================
def experiment_batch_size_study(train_data, train_labels, test_data, test_labels):
    """Train CNN with different batch sizes"""
    print("\n" + "="*60)
    print("EXPERIMENT (c): MINI-BATCH SIZE STUDY")
    print("="*60)
    
    batch_sizes = [1, 8, 16, 64, 256]
    results_bs = {}
    
    for bs in batch_sizes:
        print(f"\n--- Training with Batch Size: {bs} ---")
        train_loader, val_loader, test_loader = prepare_dataloaders(
            train_data, train_labels, test_data, test_labels, batch_size=bs
        )
        model = SimpleCNN().to(device)
        results = train_model(model, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE)
        results_bs[bs] = results
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for bs in batch_sizes:
        ax.plot(results_bs[bs]['train_losses'], label=f'BS={bs}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss vs Epoch (Different Batch Sizes)')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for bs in batch_sizes:
        ax.plot(results_bs[bs]['val_losses'], label=f'BS={bs}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Epoch (Different Batch Sizes)')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Validation Accuracy
    ax = axes[1, 0]
    for bs in batch_sizes:
        ax.plot(results_bs[bs]['val_accs'], label=f'BS={bs}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy vs Epoch (Different Batch Sizes)')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Test Accuracy Comparison
    ax = axes[1, 1]
    test_accs = [results_bs[bs]['test_acc'] for bs in batch_sizes]
    ax.bar([str(bs) for bs in batch_sizes], test_accs, color='lightgreen')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Final Test Accuracy vs Batch Size')
    ax.grid(True, axis='y')
    
    for i, acc in enumerate(test_accs):
        ax.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('experiment_c_batch_size_study.png', dpi=150)
    print("\n✓ Saved: experiment_c_batch_size_study.png")
    
    # Summary table
    print("\n" + "="*60)
    print("BATCH SIZE STUDY SUMMARY")
    print("="*60)
    print(f"{'Batch Size':<15} {'Test Acc':<12} {'Train Time (s)':<15}")
    print("-"*60)
    for bs in batch_sizes:
        print(f"{bs:<15} {results_bs[bs]['test_acc']:.4f}      {results_bs[bs]['train_time']:.2f}")

# ==================== MAIN ====================
if __name__ == "__main__":
    # Load data once
    print("Loading CIFAR-10 from local path...")
    train_data, train_labels, test_data, test_labels, class_names = load_cifar10_local()
    
    print("\nPreparing DataLoaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_data, train_labels, test_data, test_labels, batch_size=BATCH_SIZE
    )
    
    # User menu
    while True:
        print("Select an experiment:")
        print("  (a) CNN Depth Analysis")
        print("  (b) Learning Rate Analysis")
        print("  (c) Mini-batch Size Study")
        
        choice = input("Enter your choice (a/b/c/q): ")
        
        if choice == 'a':
            experiment_depth_analysis(train_loader, val_loader, test_loader)
        elif choice == 'b':
            experiment_learning_rate_analysis(train_loader, val_loader, test_loader)
        elif choice == 'c':
            experiment_batch_size_study(train_data, train_labels, test_data, test_labels)
        else:
            print("Invalid choice. Please try again.")