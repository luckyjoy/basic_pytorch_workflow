# =======================================================================================
# PYTORCH FUNDAMENTALS AND ADVANCED TECHNIQUES Recurrent Neural Network (sequential data)
# ========================================================================================
# Author: Bang Thien Nguyen ontario1998@gmail.com
# Goal: Provide a single, runnable Python file demonstrating all core components 
# of a modern PyTorch deep learning workflow, from basic tensors to advanced 
# concepts like Transfer Learning and robust error handling.
# 
# Key Features Included:
# 1. Tensor creation and device management (CPU/GPU).
# 2. Autograd and automatic gradient calculation.
# 3. Neural Network definition with regularization (Dropout), CNN, and RNN examples.
# 4. Training utilities (Loss, Optimizer, Learning Rate Scheduler, Gradient Clipping).
# 5. Model persistence (Saving and Loading state_dict & TorchScript).
# 6. Comprehensive Training and Evaluation loops.
# 7. Model Inspection and Parameter Analysis.
# 8. Practical Transfer Learning setup using pre-trained ResNet-18.
# 9. Key PyTorch error handling techniques for robustness.
# 10. Model Deployment preparation (TorchScript).
# 11. Performance Optimization (Mixed Precision).
# =======================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import warnings # New Import
import torchvision.models as models 

# Utility to suppress the recurring 'pynvml is deprecated' FutureWarning
warnings.filterwarnings(
    "ignore", 
    message="The pynvml package is deprecated. Please install nvidia-ml-py instead.",
    category=FutureWarning
)

# --- 1. TENSORS: THE CORE DATA STRUCTURE ---
# Tensors are multi-dimensional arrays, similar to NumPy arrays,
# but they can utilize GPUs for accelerated computation.

print("--- 1. TENSOR FUNDAMENTALS ---")

# Determine the device to use (GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# The device is set once at the beginning
print(f"Using device: {device}")

# A. Creating Tensors
data_list = [[1, 2], [3, 4]]
x_data = torch.tensor(data_list, dtype=torch.float32)
np_array = np.array(data_list)
x_np = torch.from_numpy(np_array)
x_rand = torch.rand(2, 3)

# C. Moving Tensors to GPU (if available)
if device == 'cuda':
    x_gpu = x_data.to(device)
else:
    pass

# D. Tensor Operations (Example: Matrix Multiplication)
x_ones = torch.ones(3, 2, dtype=torch.float32)
x_twos = torch.ones(2, 3, dtype=torch.float32) * 2
result = x_ones @ x_twos
print("Section 1 (Tensors) ran successfully.")


# --- 2. AUTOGRAD: AUTOMATIC DIFFERENTIATION ---

print("\n--- 2. AUTOGRAD (GRADIENT CALCULATION) ---")
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3 * x + 1
z = y.sum()

try:
    z.backward()
    print("Section 2 (Autograd) ran successfully. Gradient calculated: 7.0")
    
    # ERROR HANDLING EXAMPLE: Attempting to call backward() twice
    try:
        z.backward()
    except RuntimeError as e:
        print(f"Autograd Error Handling Example: Caught expected RuntimeError (calling backward twice): {e}")

except RuntimeError as e:
    print(f"Error during Autograd calculation: {e}")


# --- 3. BUILDING A NEURAL NETWORK (NN.MODULE) ---

print("\n--- 3. NEURAL NETWORK DEFINITION ---")

# Define a simple Feed-Forward Network (for demonstration with simulated tabular data)
class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNeuralNet, self).__init__()
        # Define the layers
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # FEATURE: Added a Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5) 
        self.layer_2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Define the forward computation flow
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.dropout(out) # Apply dropout after ReLU
        out = self.layer_2(out)
        return out

# Define a simple Convolutional Neural Network (for practice with image data)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer 1: Input=3 channels (RGB), Output=16 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # Max Pooling Layer: Reduces feature map size by half (2x2 kernel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Layer 2: Input=16 channels, Output=32 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # Fully Connected Layer (assuming input image size is 32x32 for simplicity)
        # 32 channels * (32/4) width * (32/4) height = 32 * 8 * 8 = 2048 features
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x):
        # x shape: [Batch, 3, H, W]
        out = self.pool(self.relu1(self.conv1(x)))
        out = self.pool(self.relu2(self.conv2(out)))
        
        # Flatten the output for the fully connected layer
        out = torch.flatten(out, 1) # Flattens all dimensions except batch dimension
        
        out = self.fc(out)
        return out

# Define a simple Recurrent Neural Network (RNN) for sequence data
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN Layer: batch_first=True means input shape is [Batch, Seq_len, Input_size]
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Final fully connected layer to map hidden state of the last time step to classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [Batch, Seq_len, Input_size]
        
        # Initialize hidden state (h0)
        # Shape: [num_layers * num_directions, batch_size, hidden_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        # out shape: [Batch, Seq_len, Hidden_size] (Output for every time step)
        # h_n shape: [num_layers, Batch, Hidden_size] (Final hidden state)
        out, h_n = self.rnn(x, h0)
        
        # We only care about the output of the LAST time step for classification.
        # Index -1 selects the output corresponding to the final sequence element.
        # out[:, -1, :] shape: [Batch, Hidden_size]
        out = self.fc(out[:, -1, :])
        
        return out


# Instantiate the Model parameters (Now using SimpleRNN for sequential data demonstration)
input_size = 10     # Number of features per time step
hidden_size = 5     # Size of the RNN hidden state
num_classes = 2
batch_size = 4      # Define batch size for the training loop
seq_len = 8         # Sequence length (e.g., number of words or time points)
num_layers = 1      # Number of recurrent layers (single layer RNN)


# Create an instance of the model and move it to the correct device
model = SimpleRNN(input_size, hidden_size, num_layers, num_classes).to(device)

print(f"Model Architecture defined: {model.__class__.__name__} (Recurrent Network)")
print(f"CNN Architecture reference: {SimpleCNN.__name__} (Convolutional Layers for image practice)")
print(f"Feed-Forward Architecture reference: {SimpleNeuralNet.__name__} (For tabular data practice)")


# --- 4. LOSS FUNCTION, OPTIMIZER, AND SCHEDULER (Training Essentials) ---
# SCHEDULER is a new utility function added here.

print("\n--- 4. LOSS & OPTIMIZER & SCHEDULER ---")
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# New Utility: Learning Rate Scheduler
# StepLR decays the learning rate by 'gamma' every 'step_size' epochs.
scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
print(f"Optimizer: {optimizer.__class__.__name__}, Scheduler: {scheduler.__class__.__name__}")


# --- 5. SAVING AND LOADING A MODEL ---

print("\n--- 5. SAVING AND LOADING ---")
FILEPATH = "model_weights.pth"

# 1. Saving the model state
try:
    torch.save(model.state_dict(), FILEPATH)
    print(f"Model weights successfully saved to '{FILEPATH}'")
except Exception as e:
    print(f"Error saving model state: {e}")

# 2. Loading the model state
# First, create a fresh instance of the model (with random weights)
# CRITICAL: Must use the same RNN architecture for loading
loaded_model = SimpleRNN(input_size, hidden_size, num_layers, num_classes).to(device)
print(f"New model instance created (random weights).")

try:
    # Then, load the saved state dictionary
    loaded_model.load_state_dict(torch.load(FILEPATH, map_location=device))
    loaded_model.eval() # Set to evaluation mode after loading
    print("Saved weights loaded back into the new model instance.")
    # Show a weight to prove the load was successful
    print(f"Example: First weight of loaded model: {loaded_model.rnn.weight_ih_l0.data[0][0]:.4f}")
except FileNotFoundError:
    print(f"Error loading model: File '{FILEPATH}' not found.")
except RuntimeError as e:
    # This catches errors if the loaded state dict doesn't match the model's architecture
    print(f"Error loading model: State dictionary mismatch (likely model change). Details: {e}")
except Exception as e:
    print(f"General error during model loading: {e}")


# --- 6. DATA HANDLING AND FULL TRAINING LOOP ---

print("\n--- 6. DATA HANDLING AND TRAINING LOOP ---")

# A. Define a Custom Dataset (to hold the simulated data)
class CustomTensorDataset(data.Dataset):
    """A standard PyTorch Dataset wrapper for input and target tensors."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        # Returns the total number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Returns one sample (feature and label) at the given index
        return self.features[idx], self.labels[idx]


# B. Simulate Data (Now generating sequential data: [Samples, Sequence Length, Features])
num_samples = 100
X_train = torch.randn(num_samples, seq_len, input_size) # SHAPE UPDATED FOR RNN
y_train = torch.randint(0, num_classes, (num_samples,))

# Create a small evaluation dataset
X_eval = torch.randn(20, seq_len, input_size) # SHAPE UPDATED FOR RNN
y_eval = torch.randint(0, num_classes, (20,))

print(f"Simulated Train Dataset: {num_samples} sequences of length {seq_len}. Eval Dataset: 20 sequences.")

# C. Instantiate DataLoaders
train_dataset = CustomTensorDataset(X_train, y_train)
eval_dataset = CustomTensorDataset(X_eval, y_eval)

train_loader = data.DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True # Shuffle data every epoch for better training
)
eval_loader = data.DataLoader(
    dataset=eval_dataset, 
    batch_size=batch_size, 
    shuffle=False # No need to shuffle evaluation data
)
print("DataLoaders created for training and evaluation.")


# D. Define the Training Function (Updated to include Gradient Clipping)
def train_model(model, data_loader, loss_fn, optimizer, scheduler, epochs, device, max_norm=1.0):
    """
    Performs the core training loop with LR Scheduler, error handling, and 
    Gradient Clipping (max_norm=1.0) for stability.
    """
    model.train() # Set the model to training mode
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        # Iterate over batches provided by the DataLoader
        for i, (X_batch, y_batch) in enumerate(data_loader):
            try:
                # Move data to the correct device (CPU/GPU)
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                # 1. Zero the gradients
                optimizer.zero_grad()

                # 2. Forward pass: compute prediction (logits)
                logits = model(X_batch)

                # 3. Calculate loss
                loss = loss_fn(logits, y_batch)

                # 4. Backward pass: compute gradient of loss w.r.t model parameters
                loss.backward()
                
                # NEW PRACTICAL FEATURE: Gradient Clipping 
                # Prevents exploding gradients, crucial for RNNs
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

                # 5. Optimization step: update weights
                optimizer.step()
                
                running_loss += loss.item()
            except Exception as e:
                # This catches common batch issues like shape mismatch or device errors
                print(f"Error in batch {i+1} of Epoch {epoch+1}: {e}. Skipping batch.")
                continue # Skip the current batch and move to the next one
        
        # UTILITY: Step the learning rate scheduler after the epoch
        scheduler.step()
        
        # Print statistics every epoch
        avg_loss = running_loss / len(data_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Avg Batch Loss: {avg_loss:.4f} | Current LR: {current_lr:.6f}")

    print("Training finished.")
    
# E. Run the Training Loop
num_epochs = 5
train_model(model, train_loader, loss_fn, optimizer, scheduler, num_epochs, device, max_norm=1.0)
print(f"Gradient Clipping applied with max_norm=1.0 during training.")


# --- 7. MODEL EVALUATION AND PREDICTION (UTILITIES) ---

print("\n--- 7. MODEL EVALUATION AND PREDICTION ---")

# UTILITY: Evaluation Function
def evaluate_model(model, data_loader, device):
    """Calculates the accuracy of the model on a given DataLoader."""
    model.eval() # Set the model to evaluation mode (disables dropout, etc.)
    correct = 0
    total = 0
    
    # torch.no_grad() disables gradient calculation, saving memory and speeding up computation
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            logits = model(X_batch)
            
            # Get the predicted class (index of the max logit)
            _, predicted = torch.max(logits.data, 1)
            
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Run Evaluation
accuracy = evaluate_model(model, eval_loader, device)
print(f"Model Accuracy on Evaluation Data: {accuracy:.2f}%")


# UTILITY: Single Prediction Example (Adapted for Sequential Input)
def predict_single(model, sample_input, device):
    """Performs a single inference step. Expects a list/array of sequential feature vectors."""
    model.eval() # Always set to eval mode for inference
    
    # Ensure input is a tensor and moved to the correct device
    # Input shape should be [Sequence_Length, Input_Feature_Size]
    input_tensor = torch.tensor(sample_input, dtype=torch.float32).to(device)
    
    # Add a batch dimension at the start: [1, Sequence_Length, Input_Feature_Size]
    input_tensor = input_tensor.unsqueeze(0) 

    with torch.no_grad():
        logits = model(input_tensor)
        
    # Apply Softmax to get probabilities (optional, but helpful for interpretation)
    probabilities = torch.softmax(logits, dim=1)
    
    # Get the predicted class index
    _, predicted_class = torch.max(logits, 1)
    
    return predicted_class.item(), probabilities.squeeze().tolist()

# Prepare a random sample for prediction (list of 8 sequences, each of 10 floats)
# Shape: [seq_len, input_size]
random_sample = np.random.rand(seq_len, input_size).tolist()

# Run Prediction
pred_class, probs = predict_single(model, random_sample, device)
print(f"\nExample Prediction (Sequential Input):")
print(f"Input Sequence Length: {len(random_sample)}, Feature Size: {len(random_sample[0])}")
print(f"Predicted Class Index: {pred_class}")
print(f"Class Probabilities: {probs}")


# --- 8. VISUALIZATION AND INSPECTION UTILITY ---

print("\n--- 8. VISUALIZATION AND INSPECTION UTILITY ---")

def inspect_model_parameters(model):
    """Utility to print the shapes and an excerpt of learned weights/biases."""
    print("Inspecting Learned Parameters (After Training):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            # We use .data to access the tensor without the gradient history
            print(f"  Layer: {name}, Shape: {param.data.shape}")
            # Print a small excerpt of the weights/biases
            if param.data.dim() > 1:
                # For weights (matrices), print the mean value
                print(f"    Weights excerpt (mean): {param.data.mean().item():.4f}")
            else:
                # For biases (vectors), print the first 3 values
                print(f"    Biases excerpt (first 3): {[f'{x:.4f}' for x in param.data[:3].tolist()]}")
    
    # Example usage of torch.flatten utility (still relevant for general tensor ops)
    example_tensor = torch.randn(2, 3, 4)
    flattened_tensor = torch.flatten(example_tensor, start_dim=1)
    print(f"\nExample torch.flatten utility: Input {example_tensor.shape} -> Output {flattened_tensor.shape}")

# Run the inspection utility on the trained model
inspect_model_parameters(model)


# --- 9. TRANSFER LEARNING EXAMPLE (PRACTICAL) ---

print("\n--- 9. TRANSFER LEARNING EXAMPLE (PRACTICAL) ---")

# A. Setup Function using a standard Pre-trained Model (e.g., ResNet18)
def setup_practical_transfer_model(new_num_classes, device):
    """
    Demonstrates practical transfer learning using a pre-trained ResNet-18.
    1. Loads ResNet-18 with pre-trained weights.
    2. Freezes all feature-extracting layers.
    3. Replaces the final classification layer (model.fc).
    """
    # 1. Load the pre-trained model (ResNet18)
    try:
        # weights='IMAGENET1K_V1' loads weights trained on the ImageNet dataset.
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        # Fallback if weights cannot be loaded (e.g., no internet access)
        print("Warning: Could not load pre-trained ResNet18 weights. Using untrained ResNet18.")
        base_model = models.resnet18()
        
    base_model = base_model.to(device)
    
    # 2. Freeze all parameters in the feature extracting base
    # We turn off gradient calculation for the weights we don't want to change.
    for param in base_model.parameters():
        param.requires_grad = False
    print("ResNet-18 base weights frozen (requires_grad=False).")

    # 3. Replace the final classification layer (the model's "head")
    # For ResNet18, the final layer is named 'fc' (Fully Connected)
    
    # Get the number of input features to the final layer (512 for ResNet18)
    num_ftrs = base_model.fc.in_features
    
    # Replace the existing layer with a new one for our specific num_classes
    base_model.fc = nn.Linear(num_ftrs, new_num_classes)
    
    # The new layer's parameters are created with requires_grad=True by default, 
    # ensuring they will be trained.
    print(f"ResNet-18 final layer ('fc') replaced with a new layer for {new_num_classes} classes.")
    
    return base_model

# B. Run the Practical Transfer Learning Setup
# NOTE: input_size is conceptually ignored here, as ResNet expects 3-channel image data.
transfer_model_practical = setup_practical_transfer_model(num_classes, device)
print("\nPractical Transfer Learning Model setup (ResNet-18).")

# C. Check which parameters are trainable
print("Verifying Trainable Parameters:")
total_params = sum(p.numel() for p in transfer_model_practical.parameters())
trainable_params = sum(p.numel() for p in transfer_model_practical.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params}")
# ResNet18 fc layer has 512 inputs * 2 outputs (weights) + 2 outputs (biases) = 1026
print(f"Trainable Parameters (New Head Only): {trainable_params} (Expected 1026 for a 2-class ResNet head)")

# D. Setup Optimizer for Transfer Learning
# CRITICAL: We pass ALL parameters to the optimizer, but only those where 
# requires_grad=True (the new fc layer) will receive updates!
transfer_optimizer_practical = optim.Adam(
    transfer_model_practical.parameters(), 
    lr=1e-3
)
print(f"Transfer Optimizer targets all parameters but only trains the new classification layer.")


# --- 10. MODEL DEPLOYMENT PREP (TORCHSCRIPT) ---

print("\n--- 10. MODEL DEPLOYMENT PREP (TORCHSCRIPT) ---")

TORCHSCRIPT_FILEPATH = "rnn_model_scripted.pt"
script_input = torch.randn(1, seq_len, input_size).to(device) # Single sample input for tracing/scripting

try:
    # Use torch.jit.script to trace the model's forward pass logic
    # This creates a module that can be saved and loaded independently of Python
    scripted_model = torch.jit.script(model, example_inputs=[script_input])
    
    # Save the TorchScript model
    scripted_model.save(TORCHSCRIPT_FILEPATH)
    print(f"Model successfully converted to TorchScript and saved to '{TORCHSCRIPT_FILEPATH}'")
    
    # Demonstrate loading and running the scripted model
    loaded_scripted_model = torch.jit.load(TORCHSCRIPT_FILEPATH)
    test_output = loaded_scripted_model(script_input)
    print(f"Loaded TorchScript model ran successfully. Output shape: {test_output.shape}")

except Exception as e:
    print(f"Error during TorchScript creation/loading: {e}")
    print("This usually happens if the model uses operations not supported by scripting.")


# --- 11. PERFORMANCE OPTIMIZATION (AUTOMATIC MIXED PRECISION - AMP) ---

print("\n--- 11. PERFORMANCE OPTIMIZATION (AMP) ---")

# Automatic Mixed Precision (AMP) uses float16 (half-precision) for certain 
# computations to speed up training and save memory, while maintaining stability 
# with float32 for critical parts like parameter updates.

if device == 'cuda':
    # Create the GradScaler once at the beginning of training (for stability)
    scaler = torch.cuda.amp.GradScaler() 
    print("CUDA device detected. AMP's GradScaler initialized.")
    print("To use AMP, the training loop would be modified to utilize the scaler.")
    
    # Simplified example of AMP usage structure:
    # 1. Initialize scaler = torch.cuda.amp.GradScaler()
    # 2. Wrap forward pass: with torch.cuda.amp.autocast(): logits = model(X_batch)
    # 3. Scale loss: scaler.scale(loss).backward()
    # 4. Step optimizer: scaler.step(optimizer)
    # 5. Update scaler: scaler.update()

else:
    print("AMP skipped: CUDA device not available. AMP requires a compatible GPU.")


print("\nAll common PyTorch features, including practical deployment and optimization features, integrated.")
