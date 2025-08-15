import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import struct
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DigitRecognizer(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(DigitRecognizer, self).__init__()
        self.flatten = nn.Flatten()
        
        # Deeper architecture with batch normalization and dropout
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return F.log_softmax(x, dim=1)

def read_idx(filename):
    """Read IDX file format with error handling."""
    try:
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        raise

def train_model(learning_rate=0.001, batch_size=128, epochs=20, validation_split=0.1):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load and preprocess data
        print("Loading training data...")
        train_images = read_idx('train-images.idx3-ubyte')
        train_labels = read_idx('train-labels.idx1-ubyte')
        test_images = read_idx('t10k-images.idx3-ubyte')
        test_labels = read_idx('t10k-labels.idx1-ubyte')

        # Normalize and convert to tensors
        X = torch.FloatTensor(train_images.reshape(-1, 784)) / 255.0
        y = torch.LongTensor(train_labels)
        X_test = torch.FloatTensor(test_images.reshape(-1, 784)) / 255.0
        y_test = torch.LongTensor(test_labels)

        # Split training data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize model, optimizer, and criterion
        model = DigitRecognizer().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }

        best_val_acc = 0
        
        print("Starting training...")
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            correct_train = 0
            total_train = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()

            # Validation phase
            model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += batch_y.size(0)
                    correct_val += (predicted == batch_y).sum().item()

            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            train_acc = 100 * correct_train / total_train
            val_acc = 100 * correct_val / total_val

            # Update learning rate scheduler
            scheduler.step(val_loss)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Load best model and evaluate on test set
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        print(f'Final Test Accuracy: {100 * correct / total:.2f}%')
        
        return model, history

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

if __name__ == '__main__':
    model, history = train_model()