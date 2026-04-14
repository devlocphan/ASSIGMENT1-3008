import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== HYPERPARAMETERS ====================
BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 15
SUBSET_SIZE = 0.3

# ==================== 1. LOAD CIFAR-10 ====================
def load_data_cifar10(subset_size=0.3):
    """Load CIFAR-10 using torchvision"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Use subset for speed
    train_size = int(len(trainset) * subset_size)
    test_size = int(len(testset) * subset_size)
    
    trainset, _ = random_split(
        trainset, [train_size, len(trainset) - train_size],
        generator=torch.Generator().manual_seed(0)
    )
    testset, _ = random_split(
        testset, [test_size, len(testset) - test_size],
        generator=torch.Generator().manual_seed(0)
    )
    
    # Manual validation split (0.2)
    val_size = int(len(trainset) * 0.2)
    train_ds, val_ds = random_split(
        trainset, [len(trainset) - val_size, val_size],
        generator=torch.Generator().manual_seed(0)
    )
    
    return train_ds, val_ds, testset

def load_batches(train_ds, val_ds, test_ds, batch_size=128):
    """Create DataLoaders"""
    train_iter = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_iter = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_iter = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_iter, val_iter, test_iter

# ==================== 2. MODELS ====================
class Softmax(nn.Module):
    """Softmax Regression"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 10)
    
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

def create_cnn(depth=2, use_dropout=False, use_residual=False):
    """Create CNN with variable depth"""
    layers = []
    in_ch, out_ch = 3, 32
    pool_count = 0
    max_pools = 3
    
    for i in range(depth):
        layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        layers.append(nn.ReLU())
        
        if use_dropout:
            layers.append(nn.Dropout2d(0.25))
        
        should_pool = (
            (i + 1) % 2 == 0 and 
            pool_count < max_pools and 
            (i + 1) < depth - 2
        )
        
        if should_pool:
            layers.append(nn.MaxPool2d(2, 2))
            pool_count += 1
        
        in_ch = out_ch
        if i % 4 == 1:
            out_ch = min(out_ch * 2, 256)
    
    layers.append(nn.AdaptiveAvgPool2d(1))
    
    model = nn.Sequential(*layers)
    return model

class CNN(nn.Module):
    """CNN with variable depth"""
    def __init__(self, depth=2, use_dropout=False, use_residual=False):
        super().__init__()
        self.conv = create_cnn(depth, use_dropout, use_residual)
        
        with torch.no_grad():
            x = torch.randn(1, 3, 32, 32).to(device)
            x = self.conv(x)
            flat_size = x.view(1, -1).size(1)
        
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SimpleCNN(nn.Module):
    """Simple CNN: 2 Conv layers, 1 MaxPool, 1 Linear with ReLU"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))

# ==================== 3. TRAINING ====================
def accuracy(y_hat, y):
    """Compute accuracy"""
    return float((torch.argmax(y_hat, dim=1) == y).sum()) / len(y)

def train_epoch(net, train_iter, loss_fn, optimizer):
    """Train one epoch"""
    net.train()
    total_loss = 0
    num_batches = 0
    
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def evaluate(net, data_iter, loss_fn):
    """Evaluate model"""
    net.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            
            total_loss += loss.item()
            total_acc += accuracy(y_hat, y)
            num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches

def train(net, train_iter, val_iter, test_iter, epochs, lr, weight_decay=0):
    """Train model"""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_losses, val_losses, val_accs = [], [], []
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss = train_epoch(net, train_iter, loss_fn, optimizer)
        val_loss, val_acc = evaluate(net, val_iter, loss_fn)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.3f} | Val: {val_loss:.3f} | Acc: {val_acc:.3f}")
    
    elapsed = time.time() - start_time
    test_loss, test_acc = evaluate(net, test_iter, loss_fn)
    print(f"  Test Acc: {test_acc:.4f} | Time: {elapsed:.0f}s\n")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'train_time': elapsed
    }

# ==================== EXPERIMENTS ====================
def experiment_depth_analysis(train_iter, val_iter, test_iter):
    """(a) CNN Depth Analysis: depths 2, 8, 16, 32"""
    print("\n[Experiment A] CNN Depth Analysis")
    print("-" * 50)
    
    depths = [2, 8, 16, 32]
    results = {}
    
    for depth in depths:
        print(f"Depth {depth}:")
        net = CNN(depth=depth).to(device)
        result = train(net, train_iter, val_iter, test_iter, EPOCHS, LEARNING_RATE)
        results[depth] = result
    
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

def experiment_learning_rate_analysis(train_iter, val_iter, test_iter):
    """(b) Learning Rate Analysis: LRs 0.000001, 0.0001, 0.001, 0.01, 1.0"""
    print("\n[Experiment B] Learning Rate Analysis")
    print("-" * 50)
    
    learning_rates = [0.000001, 0.0001, 0.001, 0.01, 1.0]
    results = {}
    
    for lr in learning_rates:
        print(f"LR {lr}:")
        net = SimpleCNN().to(device)
        result = train(net, train_iter, val_iter, test_iter, EPOCHS, lr)
        results[lr] = result
    
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
    ax.bar([f'{lr:.6f}' for lr in learning_rates], test_accs, color='coral', alpha=0.7)
    ax.set_ylabel('Test Accuracy')
    ax.set_xlabel('Learning Rate')
    ax.set_title('Final Test Accuracy')
    ax.grid(True, axis='y', alpha=0.3)
    for i, acc in enumerate(test_accs):
        ax.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('experiment_b_lr.png', dpi=100)
    print("✓ Saved: experiment_b_lr.png")
    
    print(f"\n{'LR':<12} {'Test Acc':<10} {'Time (s)':<10}")
    for lr in learning_rates:
        print(f"{lr:<12.6f} {results[lr]['test_acc']:.4f}     {results[lr]['train_time']:.0f}")

def experiment_batch_size_study(train_ds, val_ds, test_ds):
    """(c) Mini-batch Size Study: batch sizes 1, 8, 16, 64, 256"""
    print("\n[Experiment C] Mini-batch Size Study")
    print("-" * 50)
    
    batch_sizes = [1, 8, 16, 64, 256]
    results = {}
    
    for bs in batch_sizes:
        print(f"Batch Size {bs}:")
        train_iter, val_iter, test_iter = load_batches(train_ds, val_ds, test_ds, batch_size=bs)
        net = SimpleCNN().to(device)
        result = train(net, train_iter, val_iter, test_iter, EPOCHS, LEARNING_RATE)
        results[bs] = result
    
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
    print(f"Loading CIFAR-10 ({SUBSET_SIZE*100:.0f}% subset)...\n")
    train_ds, val_ds, test_ds = load_data_cifar10(SUBSET_SIZE)
    
    train_iter, val_iter, test_iter = load_batches(train_ds, val_ds, test_ds, BATCH_SIZE)
    
    while True:
        print("Select: (a) Depth (b) Learning Rate (c) Batch Size (q) Quit")
        choice = input("Choice: ").strip().lower()
        
        if choice == 'a':
            experiment_depth_analysis(train_iter, val_iter, test_iter)
        elif choice == 'b':
            experiment_learning_rate_analysis(train_iter, val_iter, test_iter)
        elif choice == 'c':
            experiment_batch_size_study(train_ds, val_ds, test_ds)
        elif choice == 'q':
            print("Done!")
            break