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
print(f"Using device: {device}\n")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30

# ==================== PATH SETUP ====================
def get_cifar10_path():
    """Auto-detect path for local or Google Colab"""
    # Try Colab path first
    colab_path = "/content/drive/My Drive/AS1/cifar-10-batches-py"
    if os.path.exists(colab_path):
        print(f"Found Colab path: {colab_path}")
        return colab_path
   
    # Try GitHub clone path (if cloned locally)
    github_path = "./cifar-10-batches-py"
    if os.path.exists(github_path):
        print(f"Found GitHub path: {github_path}")
        return github_path
    
    raise FileNotFoundError("CIFAR-10 data not found. Please check paths.")

CIFAR10_PATH = get_cifar10_path()

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
    
    print(f"Training data: {train_data.shape}, Test data: {test_data.shape}\n")
    
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

# ==================== 2. CNN MODELS ====================
def build_cnn(depth=2, use_residual=False, use_dropout=False, dropout_rate=0.3):
    """Build CNN with configurable depth"""
    layers = []
    in_channels = 3
    out_channels = 32
    
    for i in range(depth):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_rate))
        
        if (i + 1) % 2 == 0:
            layers.append(nn.MaxPool2d(2, 2))
        
        in_channels = out_channels
        if i % 4 == 1:
            out_channels = min(out_channels * 2, 256)
    
    class DynamicCNN(nn.Module):
        def __init__(self, conv_layers, in_ch, use_res):
            super(DynamicCNN, self).__init__()
            self.conv_layers = nn.Sequential(*conv_layers)
            self.use_residual = use_res
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
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f}")
    
    training_time = time.time() - start_time
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    print(f"  Final Test Accuracy: {test_acc:.4f} | Time: {training_time:.1f}s\n")
    
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
    print("\n[Experiment A] CNN Depth Analysis")
    print("-" * 50)
    
    depths = [2, 8, 16, 32]
    results_depth = {}
    
    for depth in depths:
        print(f"Training CNN (depth={depth})...")
        model = build_cnn(depth=depth).to(device)
        results = train_model(model, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE)
        results_depth[depth] = results
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    for depth in depths:
        ax.plot(results_depth[depth]['train_losses'], label=f'Depth {depth}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for depth in depths:
        ax.plot(results_depth[depth]['val_losses'], label=f'Depth {depth}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for depth in depths:
        ax.plot(results_depth[depth]['val_accs'], label=f'Depth {depth}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    test_accs = [results_depth[d]['test_acc'] for d in depths]
    ax.bar([str(d) for d in depths], test_accs, color='steelblue', alpha=0.7)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Final Test Accuracy')
    ax.grid(True, axis='y', alpha=0.3)
    for i, acc in enumerate(test_accs):
        ax.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('experiment_a_depth_analysis.png', dpi=150)
    print("✓ Saved: experiment_a_depth_analysis.png")
    
    print("\nResults Summary:")
    print(f"{'Depth':<10} {'Test Acc':<12} {'Time (s)':<10}")
    for depth in depths:
        print(f"{depth:<10} {results_depth[depth]['test_acc']:.4f}      {results_depth[depth]['train_time']:.1f}")

# ==================== EXPERIMENT (b): LEARNING RATE ANALYSIS ====================
def experiment_learning_rate_analysis(train_loader, val_loader, test_loader):
    """Train CNN with different learning rates"""
    print("\n[Experiment B] Learning Rate Analysis")
    print("-" * 50)
    
    learning_rates = [0.000001, 0.0001, 0.001, 0.01, 1.0]
    results_lr = {}
    
    for lr in learning_rates:
        print(f"Training with learning rate={lr}...")
        model = SimpleCNN().to(device)
        results = train_model(model, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=lr)
        results_lr[lr] = results
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    for lr in learning_rates:
        ax.plot(results_lr[lr]['train_losses'], label=f'LR={lr}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for lr in learning_rates:
        ax.plot(results_lr[lr]['val_losses'], label=f'LR={lr}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for lr in learning_rates:
        ax.plot(results_lr[lr]['val_accs'], label=f'LR={lr}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    test_accs = [results_lr[lr]['test_acc'] for lr in learning_rates]
    ax.bar([str(lr) for lr in learning_rates], test_accs, color='coral', alpha=0.7)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Final Test Accuracy')
    ax.grid(True, axis='y', alpha=0.3)
    for i, acc in enumerate(test_accs):
        ax.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('experiment_b_learning_rate_analysis.png', dpi=150)
    print("✓ Saved: experiment_b_learning_rate_analysis.png")
    
    print("\nResults Summary:")
    print(f"{'LR':<15} {'Test Acc':<12} {'Time (s)':<10}")
    for lr in learning_rates:
        print(f"{lr:<15} {results_lr[lr]['test_acc']:.4f}      {results_lr[lr]['train_time']:.1f}")

# ==================== EXPERIMENT (c): MINI-BATCH SIZE STUDY ====================
def experiment_batch_size_study(train_data, train_labels, test_data, test_labels):
    """Train CNN with different batch sizes"""
    print("\n[Experiment C] Mini-batch Size Study")
    print("-" * 50)
    
    batch_sizes = [1, 8, 16, 64, 256]
    results_bs = {}
    
    for bs in batch_sizes:
        print(f"Training with batch size={bs}...")
        train_loader, val_loader, test_loader = prepare_dataloaders(
            train_data, train_labels, test_data, test_labels, batch_size=bs
        )
        model = SimpleCNN().to(device)
        results = train_model(model, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE)
        results_bs[bs] = results
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    for bs in batch_sizes:
        ax.plot(results_bs[bs]['train_losses'], label=f'BS={bs}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for bs in batch_sizes:
        ax.plot(results_bs[bs]['val_losses'], label=f'BS={bs}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for bs in batch_sizes:
        ax.plot(results_bs[bs]['val_accs'], label=f'BS={bs}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    test_accs = [results_bs[bs]['test_acc'] for bs in batch_sizes]
    ax.bar([str(bs) for bs in batch_sizes], test_accs, color='lightgreen', alpha=0.7)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Final Test Accuracy')
    ax.grid(True, axis='y', alpha=0.3)
    for i, acc in enumerate(test_accs):
        ax.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('experiment_c_batch_size_study.png', dpi=150)
    print("✓ Saved: experiment_c_batch_size_study.png")
    
    print("\nResults Summary:")
    print(f"{'Batch Size':<15} {'Test Acc':<12} {'Time (s)':<10}")
    for bs in batch_sizes:
        print(f"{bs:<15} {results_bs[bs]['test_acc']:.4f}      {results_bs[bs]['train_time']:.1f}")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("Loading CIFAR-10...")
    train_data, train_labels, test_data, test_labels, class_names = load_cifar10_local()
    
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_data, train_labels, test_data, test_labels, batch_size=BATCH_SIZE
    )
    
    while True:
        print("\nSelect experiment: (a) Depth (b) Learning Rate (c) Batch Size (q) Quit")
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
        else:
            print("Invalid choice.")