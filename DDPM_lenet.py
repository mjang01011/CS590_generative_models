# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision import datasets, transforms
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import glob
# from tqdm import tqdm
# from collections import OrderedDict

# # Set random seeds for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)

# # Define device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Define LeNet5 architecture
# class LeNet5(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.features = nn.Sequential(OrderedDict([
#             ('conv1', nn.Conv2d(1, 6, kernel_size=5, padding=2)),
#             ('relu1', nn.ReLU()),
#             ('pool1', nn.AvgPool2d(kernel_size=2, stride=2)),
#             ('conv2', nn.Conv2d(6, 16, kernel_size=5)),
#             ('relu2', nn.ReLU()),
#             ('pool2', nn.AvgPool2d(kernel_size=2, stride=2))
#         ]))
        
#         self.classifier = nn.Sequential(OrderedDict([
#             ('fc1', nn.Linear(16 * 5 * 5, 120)),
#             ('relu3', nn.ReLU()),
#             ('fc2', nn.Linear(120, 84)),
#             ('relu4', nn.ReLU()),
#             ('fc3', nn.Linear(84, 10))
#         ]))
        
#         self.param_dim = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
    
#     def set_flat_weights(self, flat_weights):
#         """Set model weights from a flattened tensor"""
#         start = 0
#         for param in self.parameters():
#             num_params = param.numel()
#             param.data.copy_(flat_weights[start:start + num_params].view(param.shape))
#             start += num_params
    
#     def get_flat_weights(self):
#         """Get flattened weights from the model"""
#         return torch.cat([p.data.flatten() for p in self.parameters()])


# # Simple but effective time embedding for the diffusion model
# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = np.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        
#         # Pad if dimension is odd
#         if self.dim % 2 == 1:
#             embeddings = nn.functional.pad(embeddings, (0, 1, 0, 0))
#         return embeddings


# # DDPM model definition - simplified and optimized
# class DDPM(nn.Module):
#     def __init__(self, input_dim, hidden_dims=[1024, 2048, 1024], timesteps=1000, beta_schedule='cosine'):
#         super().__init__()
#         self.timesteps = timesteps
#         self.input_dim = input_dim
        
#         # Create beta schedule (noise level at each timestep)
#         self.register_buffer('beta', self._get_beta_schedule(beta_schedule, timesteps))
#         self.register_buffer('alpha', 1 - self.beta)
#         self.register_buffer('alpha_hat', torch.cumprod(self.alpha, dim=0))
        
#         # Time embedding
#         self.time_embed_dim = 128
#         self.time_embed = nn.Sequential(
#             SinusoidalPositionEmbeddings(self.time_embed_dim),
#             nn.Linear(self.time_embed_dim, self.time_embed_dim * 2),
#             nn.SiLU(),
#             nn.Linear(self.time_embed_dim * 2, self.time_embed_dim),
#         )
        
#         # Model architecture - simple MLP with residual connections
#         # Input layer
#         combined_input_dim = input_dim + self.time_embed_dim
#         self.input_proj = nn.Linear(combined_input_dim, hidden_dims[0])
        
#         # Middle layers with skip connections
#         self.middle_layers = nn.ModuleList()
#         for i in range(len(hidden_dims) - 1):
#             self.middle_layers.append(
#                 nn.Sequential(
#                     nn.Linear(hidden_dims[i], hidden_dims[i+1]),
#                     nn.LayerNorm(hidden_dims[i+1]),
#                     nn.SiLU()
#                 )
#             )
        
#         # Output projection
#         self.output_proj = nn.Linear(hidden_dims[-1], input_dim)
    
#     def _get_beta_schedule(self, schedule_type, timesteps):
#         """Get different beta schedules for the diffusion process"""
#         if schedule_type == 'linear':
#             return torch.linspace(1e-4, 2e-2, timesteps)
#         elif schedule_type == 'cosine':
#             # Cosine schedule as proposed in Improved DDPM paper
#             steps = timesteps + 1
#             x = torch.linspace(0, timesteps, steps)
#             alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
#             alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#             betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#             return torch.clip(betas, 0.0001, 0.9999)
#         else:
#             raise ValueError(f"Unknown beta schedule: {schedule_type}")
    
#     def forward(self, x, t):
#         """Predict the noise added to x at timestep t"""
#         t_embed = self.time_embed(t)
#         x_combined = torch.cat([x, t_embed], dim=1)
#         h = self.input_proj(x_combined)

#         # Apply residuals only when shapes match
#         for layer in self.middle_layers:
#             h_new = layer(h)
#             if h_new.shape == h.shape:
#                 h = h + h_new
#             else:
#                 h = h_new

#         return self.output_proj(h)

    
#     def add_noise(self, x_0, t):
#         """Add noise to the input according to the diffusion process"""
#         # Extract alpha_t at timestep t
#         alpha_hat_t = self.alpha_hat[t].view(-1, 1)
        
#         # Sample noise
#         epsilon = torch.randn_like(x_0)
        
#         # Forward diffusion process q(x_t | x_0)
#         x_t = torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * epsilon
        
#         return x_t, epsilon
    
#     def sample(self, n_samples, shape, guidance_scale=1.0):
#         """Generate samples by denoising from random noise"""
#         # Start from pure noise
#         x = torch.randn(n_samples, shape).to(self.beta.device)
#         original_noise = x.clone()
        
#         # Iteratively denoise
#         for t in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling"):
#             t_batch = torch.full((n_samples,), t, device=self.beta.device, dtype=torch.long)
            
#             with torch.no_grad():
#                 # Predict noise
#                 predicted_noise = self(x, t_batch)
                
#                 # Apply classifier-free guidance if requested
#                 if guidance_scale != 1.0:
#                     predicted_noise = guidance_scale * predicted_noise
                
#                 # Get parameters for the reverse process
#                 alpha_t = self.alpha[t]
#                 alpha_hat_t = self.alpha_hat[t]
#                 beta_t = self.beta[t]
                
#                 # For the last step, we don't add noise
#                 if t > 0:
#                     noise = torch.randn_like(x)
#                     alpha_hat_t_minus_1 = self.alpha_hat[t-1]
#                 else:
#                     noise = 0
#                     alpha_hat_t_minus_1 = 1.0
                
#                 # Compute predicted x_0
#                 x_0_predicted = (x - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
                
#                 # Compute parameters for the reverse step
#                 if t > 0:
#                     sigma_t = torch.sqrt(beta_t * (1 - alpha_hat_t_minus_1) / (1 - alpha_hat_t))
#                     mean = (
#                         torch.sqrt(alpha_hat_t_minus_1) * beta_t / (1 - alpha_hat_t) * x_0_predicted + 
#                         torch.sqrt(alpha_t) * (1 - alpha_hat_t_minus_1) / (1 - alpha_hat_t) * x
#                     )
#                 else:
#                     sigma_t = 0
#                     mean = x_0_predicted
                
#                 # Update x with the reverse process formula
#                 x = mean + sigma_t * noise
        
#         # Return final denoised sample and original noise
#         return x, original_noise


# # Function to load and split LeNet-5 weights
# def load_lenet5_weights(weights_dir, train_ratio=0.95):
#     """Load LeNet-5 weights from individual .pt files"""
#     weight_files = sorted(glob.glob(os.path.join(weights_dir, "*.pt")))
#     if not weight_files:
#         raise ValueError(f"No weight files found in {weights_dir}")
    
#     print(f"Found {len(weight_files)} weight files")
    
#     # Create a model to get the expected parameter count
#     model = LeNet5()
#     expected_param_count = model.param_dim
    
#     # Validate and load files
#     valid_weights = []
#     for file_path in tqdm(weight_files, desc="Loading weights"):
#         try:
#             weights = torch.load(file_path, map_location='cpu')
#             if weights.numel() == expected_param_count:
#                 valid_weights.append(weights)
#         except Exception as e:
#             print(f"Warning: Could not load {file_path}: {str(e)}")
    
#     print(f"Loaded {len(valid_weights)} valid weight files")
    
#     # Split into train and test
#     train_size = int(len(valid_weights) * train_ratio)
#     train_weights = torch.stack(valid_weights[:train_size])
#     test_weights = torch.stack(valid_weights[train_size:])
    
#     print(f"Using {len(train_weights)} files for training and {len(test_weights)} for testing")
    
#     return train_weights, test_weights


# # Function to normalize and denormalize weights
# def normalize_weights(train_weights, test_weights=None):
#     """Normalize weights for better diffusion modeling"""
#     weight_mean = train_weights.mean(dim=0, keepdim=True)
#     weight_std = train_weights.std(dim=0, keepdim=True)
    
#     # Avoid division by zero
#     weight_std = torch.clamp(weight_std, min=1e-6)
    
#     normalized_train = (train_weights - weight_mean) / weight_std
    
#     normalization_params = {
#         'mean': weight_mean,
#         'std': weight_std
#     }
    
#     if test_weights is not None:
#         normalized_test = (test_weights - weight_mean) / weight_std
#         return normalized_train, normalized_test, normalization_params
#     else:
#         return normalized_train, normalization_params


# def denormalize_weights(normalized_weights, normalization_params):
#     """Denormalize weights to original scale"""
#     # Ensure mean and std are on the same device as normalized_weights
#     mean = normalization_params['mean'].to(normalized_weights.device)
#     std = normalization_params['std'].to(normalized_weights.device)
#     return normalized_weights * std + mean



# # Function to train the diffusion model
# def train_diffusion_model(weight_data, test_weight_data=None, epochs=50, batch_size=32, 
#                          timesteps=500, beta_schedule='cosine', lr=1e-4):
#     """Train a diffusion model on model weights"""
#     # Create dataset and dataloader
#     dataset = TensorDataset(weight_data)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     # Create validation dataloader if test data is provided
#     val_dataloader = None
#     if test_weight_data is not None:
#         val_dataset = TensorDataset(test_weight_data)
#         val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
#     # Get input dimension
#     input_dim = weight_data.shape[1]
#     print(f"Input dimension for diffusion model: {input_dim}")
    
#     # Create hidden dimensions based on input size
#     hidden_dim = min(max(input_dim // 2, 512), 2048)
#     hidden_dims = [hidden_dim, hidden_dim * 2, hidden_dim]
    
#     # Create diffusion model
#     diffusion_model = DDPM(
#         input_dim=input_dim, 
#         hidden_dims=hidden_dims,
#         timesteps=timesteps,
#         beta_schedule=beta_schedule
#     ).to(device)
    
#     # Use AdamW optimizer and learning rate scheduler
#     optimizer = optim.AdamW(diffusion_model.parameters(), lr=lr, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
#     # Training loop
#     train_losses = []
#     val_losses = []
    
#     for epoch in range(epochs):
#         # Training phase
#         diffusion_model.train()
#         epoch_loss = 0
        
#         for batch_idx, (weights,) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
#             weights = weights.to(device)
            
#             # Sample random timesteps
#             t = torch.randint(0, timesteps, (weights.shape[0],), device=device)
            
#             # Add noise to weights according to timestep
#             noisy_weights, target_noise = diffusion_model.add_noise(weights, t)
            
#             # Predict noise
#             predicted_noise = diffusion_model(noisy_weights, t)
            
#             # Calculate loss
#             loss = F.mse_loss(predicted_noise, target_noise)
            
#             # Update model
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             epoch_loss += loss.item()
        
#         # Calculate average training loss
#         avg_epoch_loss = epoch_loss / len(dataloader)
#         train_losses.append(avg_epoch_loss)
        
#         # Validation phase
#         if val_dataloader:
#             diffusion_model.eval()
#             val_loss = 0
#             with torch.no_grad():
#                 for (val_weights,) in val_dataloader:
#                     val_weights = val_weights.to(device)
#                     val_t = torch.randint(0, timesteps, (val_weights.shape[0],), device=device)
#                     val_noisy_weights, val_target_noise = diffusion_model.add_noise(val_weights, val_t)
#                     val_predicted_noise = diffusion_model(val_noisy_weights, val_t)
#                     val_loss += F.mse_loss(val_predicted_noise, val_target_noise).item()
            
#             val_loss /= len(val_dataloader)
#             val_losses.append(val_loss)
#             print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
#         else:
#             print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.6f}")
        
#         # Update learning rate
#         scheduler.step()
    
#     return diffusion_model, {"train_losses": train_losses, "val_losses": val_losses}


# # Function to evaluate model weights on MNIST
# def evaluate_weights(weights, num_models=5):
#     """Evaluate given weights on MNIST test set"""
#     # Prepare test dataset
#     transform = transforms.Compose([transforms.ToTensor(), 
#                                   transforms.Normalize((0.1307,), (0.3081,))])
#     testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#     testloader = DataLoader(testset, batch_size=1000, shuffle=False)
    
#     results = []
#     n_eval = min(len(weights), num_models)
    
#     for i in range(n_eval):
#         try:
#             model = LeNet5().to(device)
#             if torch.isnan(weights[i]).any():
#                 print(f"Warning: Weight sample {i} contains NaN values, skipping")
#                 continue
                
#             # Load weights
#             model.set_flat_weights(weights[i].to(device))
            
#             # Evaluate
#             model.eval()
#             correct = 0
#             total = 0
            
#             with torch.no_grad():
#                 for data, targets in testloader:
#                     data, targets = data.to(device), targets.to(device)
#                     outputs = model(data)
#                     _, predicted = torch.max(outputs.data, 1)
#                     total += targets.size(0)
#                     correct += (predicted == targets).sum().item()
            
#             accuracy = 100 * correct / total
#             results.append({
#                 'sample_idx': i,
#                 'accuracy': accuracy
#             })
#             print(f"Weight sample {i+1}: Accuracy = {accuracy:.2f}%")
#         except Exception as e:
#             print(f"Error evaluating weight sample {i}: {str(e)}")
    
#     return results


# # Function to visualize weights using PCA
# def visualize_weights_pca(original_weights, diffused_weights, original_noise=None):
#     """Visualize weight distributions using PCA"""
#     from sklearn.decomposition import PCA
    
#     # Apply PCA
#     combined = torch.cat([original_weights.cpu(), diffused_weights.cpu()], dim=0)
#     if original_noise is not None:
#         combined = torch.cat([combined, original_noise.cpu()], dim=0)
    
#     # Fit PCA on combined data
#     pca = PCA(n_components=2)
#     all_pca = pca.fit_transform(combined.numpy())
    
#     # Split components
#     n_orig = len(original_weights)
#     n_diff = len(diffused_weights)
    
#     orig_pca = all_pca[:n_orig]
#     diff_pca = all_pca[n_orig:n_orig+n_diff]
    
#     # Plot
#     plt.figure(figsize=(10, 8))
#     plt.scatter(orig_pca[:, 0], orig_pca[:, 1], alpha=0.5, label='Original Weights', color='blue')
#     plt.scatter(diff_pca[:, 0], diff_pca[:, 1], alpha=0.8, marker='x', s=100, 
#                label='Diffused Weights', color='red')
    
#     # Add noise samples if provided
#     if original_noise is not None:
#         noise_pca = all_pca[n_orig+n_diff:]
#         plt.scatter(noise_pca[:, 0], noise_pca[:, 1], alpha=0.5, marker='o', 
#                    label='Original Noise', color='green')
        
#         # Draw arrows from noise to diffused weights
#         for i in range(min(len(diff_pca), len(noise_pca))):
#             plt.arrow(noise_pca[i, 0], noise_pca[i, 1],
#                      diff_pca[i, 0] - noise_pca[i, 0],
#                      diff_pca[i, 1] - noise_pca[i, 1],
#                      head_width=0.3, head_length=0.3, fc='black', ec='black', alpha=0.5)
    
#     plt.title('PCA of Model Weights')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.savefig('weight_pca_visualization.png')
#     plt.show()
    
#     print("PCA visualization saved to 'weight_pca_visualization.png'")
    
#     return pca, all_pca


# # Main function to orchestrate the workflow
# def main(weights_dir="./lenet5_weights", 
#          train_ratio=0.95, 
#          diff_epochs=50, 
#          timesteps=500,
#          batch_size=32,
#          num_samples=5, 
#          guidance_scale=1.0):
#     """Main workflow to train DDPM on model weights and evaluate generated weights"""
    
#     # 1. Load and split weights
#     print("\n1. Loading weights...")
#     train_weights, test_weights = load_lenet5_weights(weights_dir, train_ratio)
    
#     # 2. Normalize weights
#     print("\n2. Normalizing weights...")
#     normalized_train, normalized_test, norm_params = normalize_weights(train_weights, test_weights)
    
#     # 3. Train diffusion model
#     print("\n3. Training diffusion model...")
#     diffusion_model, loss_history = train_diffusion_model(
#         normalized_train,
#         test_weight_data=normalized_test,
#         epochs=diff_epochs,
#         batch_size=batch_size,
#         timesteps=timesteps
#     )
    
#     # 4. Generate weights from the diffusion model
#     print("\n4. Generating weights from diffusion model...")
#     diffused_normalized_weights, original_noise = diffusion_model.sample(
#         num_samples, normalized_train.shape[1], guidance_scale
#     )
    
#     # 5. Denormalize the diffused weights
#     diffused_weights = denormalize_weights(diffused_normalized_weights, norm_params)
#     original_noise_denormalized = denormalize_weights(original_noise, norm_params)
    
#     # 6. Evaluate original test weights
#     print("\n5. Evaluating original test weights...")
#     orig_results = evaluate_weights(test_weights)
    
#     # 7. Evaluate diffused weights
#     print("\n6. Evaluating diffused weights...")
#     diff_results = evaluate_weights(diffused_weights)
    
#     # 8. Visualize with PCA
#     print("\n7. Visualizing weights with PCA...")
#     try:
#         pca_results = visualize_weights_pca(test_weights, diffused_weights, original_noise_denormalized)
#     except ImportError:
#         print("PCA visualization requires scikit-learn. Install with 'pip install scikit-learn'")
#         pca_results = None
    
#     # 9. Plot training loss
#     plt.figure(figsize=(10, 6))
#     plt.plot(loss_history['train_losses'], label='Training Loss')
#     if loss_history['val_losses']:
#         plt.plot(loss_history['val_losses'], label='Validation Loss')
#     plt.title('DDPM Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('MSE Loss')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.savefig('ddpm_training_loss.png')
#     plt.show()
    
#     # 10. Save models and results
#     print("\n8. Saving models and results...")
#     os.makedirs('results', exist_ok=True)
#     torch.save(diffusion_model.state_dict(), 'results/diffusion_model.pt')
#     torch.save(norm_params, 'results/normalization_params.pt')
#     torch.save(diffused_weights, 'results/diffused_weights.pt')
    
#     # 11. Print summary
#     print("\n=== Results Summary ===")
#     orig_acc = [r['accuracy'] for r in orig_results]
#     diff_acc = [r['accuracy'] for r in diff_results]
    
#     print(f"Original weights: avg acc = {np.mean(orig_acc):.2f}%, std = {np.std(orig_acc):.2f}%")
#     print(f"Diffused weights: avg acc = {np.mean(diff_acc):.2f}%, std = {np.std(diff_acc):.2f}%")
    
#     return {
#         'diffusion_model': diffusion_model,
#         'original_results': orig_results,
#         'diffused_results': diff_results,
#         'diffused_weights': diffused_weights,
#         'loss_history': loss_history,
#         'pca_results': pca_results
#     }


# if __name__ == "__main__":
#     # Set parameters
#     weights_dir = "./lenet5_weights"  # Directory containing weight files
#     train_ratio = 0.95                # Ratio for train/test split
#     diff_epochs = 50                  # Diffusion model training epochs
#     timesteps = 500                   # Number of diffusion timesteps
#     batch_size = 32                   # Batch size for training
#     num_samples = 5                   # Number of weights to generate
#     guidance_scale = 1.0              # Scale factor for noise prediction (>1 = stronger effect)
    
#     # Run main workflow
#     results = main(
#         weights_dir=weights_dir,
#         train_ratio=train_ratio,
#         diff_epochs=diff_epochs,
#         timesteps=timesteps,
#         batch_size=batch_size,
#         num_samples=num_samples,
#         guidance_scale=guidance_scale
#     )

# hf_ddpm_lenet_weights.py

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
from diffusers import DDPMScheduler
from sklearn.decomposition import PCA

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- LeNet-5 Definition ---
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 6, kernel_size=5, padding=2)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.AvgPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(6, 16, kernel_size=5)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(16 * 5 * 5, 120)),
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(120, 84)),
            ('relu4', nn.ReLU()),
            ('fc3', nn.Linear(84, 10))
        ]))
        self.param_dim = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_flat_weights(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_flat_weights(self, flat):
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat[idx:idx+numel].view_as(p))
            idx += numel

# --- Load and Normalize Weights ---
def load_weights(weights_dir, train_ratio=0.95):
    files = sorted(glob.glob(os.path.join(weights_dir, "*.pt")))
    model = LeNet5()
    D = model.param_dim
    valid = []
    for f in tqdm(files, desc="Loading weights"):
        w = torch.load(f, map_location='cpu')
        if w.numel() == D:
            valid.append(w)
    valid = torch.stack(valid)
    split = int(len(valid) * train_ratio)
    return valid[:split], valid[split:]


def normalize_weights(train, test=None):
    mean = train.mean(0, keepdim=True)
    std = torch.clamp(train.std(0, keepdim=True), min=1e-6)
    norm_train = (train - mean) / std
    norm_test = (test - mean) / std if test is not None else None
    return norm_train, norm_test, {'mean': mean, 'std': std}

def denormalize_weights(normed, norm):
    return normed * norm['std'].to(normed.device) + norm['mean'].to(normed.device)

# --- Diffusers Scheduler and Predictor ---
def create_scheduler(num_timesteps):
    return DDPMScheduler(
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
        num_train_timesteps=num_timesteps,
        clip_sample=False
    )

class NoisePredictor(nn.Module):
    def __init__(self, dim, scheduler):
        super().__init__()
        self.scheduler = scheduler
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, dim)
        )
    def forward(self, x, t):
        t_norm = t.float().unsqueeze(-1) / self.scheduler.num_train_timesteps
        inp = torch.cat([x, t_norm.to(x.device)], dim=1)
        return self.net(inp)

# --- Training Function ---
def train_diffusers(train_w, test_w, timesteps=500, epochs=50, batch_size=32, lr=1e-4):
    D = train_w.shape[1]
    scheduler = create_scheduler(timesteps)
    model = NoisePredictor(D, scheduler).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(train_w), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_w), batch_size=batch_size) if test_w is not None else None
    train_losses, val_losses = [], []
    for ep in range(epochs):
        model.train(); tl = 0
        for (x,) in train_loader:
            x = x.to(device)
            t = torch.randint(0, timesteps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            noisy = scheduler.add_noise(x, noise, t)
            pred = model(noisy, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()
        train_losses.append(tl/len(train_loader))
        if test_loader:
            model.eval(); vl = 0
            with torch.no_grad():
                for (x,) in test_loader:
                    x = x.to(device)
                    t = torch.randint(0, timesteps, (x.size(0),), device=device)
                    noise = torch.randn_like(x)
                    noisy = scheduler.add_noise(x, noise, t)
                    pred = model(noisy, t)
                    vl += F.mse_loss(pred, noise).item()
            val_losses.append(vl/len(test_loader))
            print(f"Epoch {ep+1}/{epochs}: Train={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}")
        else:
            print(f"Epoch {ep+1}/{epochs}: Train={train_losses[-1]:.4f}")
    return model, scheduler, {'train_losses': train_losses, 'val_losses': val_losses}

# --- Sampling and PCA Progression ---
def sample_with_progress(model, scheduler, num_samples, D, steps=10):
    model.eval()
    scheduler.set_timesteps(scheduler.num_train_timesteps)
    x = torch.randn(num_samples, D).to(device)
    snapshots = []
    indices = set(np.linspace(0, len(scheduler.timesteps)-1, steps, dtype=int))
    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            pred = model(x, torch.full((num_samples,), t, device=device, dtype=torch.long))
        x = scheduler.step(pred, t, x).prev_sample
        if i in indices:
            snapshots.append(x.cpu().clone())
    return snapshots, x.cpu()

# --- PCA Plot ---
def plot_pca_progress(orig, snaps, final, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    data = torch.cat([orig.cpu()] + snaps + [final.cpu()], dim=0)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(data.numpy())
    n_orig = orig.size(0)
    idx = n_orig
    plt.figure(figsize=(10,8))
    plt.scatter(proj[:n_orig,0], proj[:n_orig,1], label='Original', alpha=0.5)
    for j, snap in enumerate(snaps):
        m = snap.size(0)
        pts = proj[idx:idx+m]
        plt.scatter(pts[:,0], pts[:,1], label=f'Step {j+1}', alpha=0.6)
        idx += m
    pts = proj[idx:idx+final.size(0)]
    plt.scatter(pts[:,0], pts[:,1], label='Final', marker='x')
    plt.legend(); plt.grid(True)
    plt.title('PCA of Diffusion Progression')
    plt.savefig(os.path.join(save_dir,'pca_progression.png'))
    plt.close()

# --- Evaluation ---
def evaluate_weights(weights, num_models=5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    loader = DataLoader(test_ds, batch_size=1000)
    results = []
    for i in range(min(num_models, len(weights))):
        model = LeNet5().to(device)
        model.set_flat_weights(weights[i].to(device))
        model.eval()
        correct, total = 0,0
        with torch.no_grad():
            for x,y in loader:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        results.append({'idx':i, 'accuracy':100*correct/total})
    return results

# --- Main ---
def main():
    out = 'huggingface_ddpm'
    os.makedirs(out, exist_ok=True)
    train_w, test_w = load_weights('./lenet5_weights')
    norm_t, norm_s, normp = normalize_weights(train_w, test_w)
    model, sched, loss_h = train_diffusers(norm_t, norm_s, timesteps=500, epochs=50, batch_size=32)
    snaps, fin = sample_with_progress(model, sched, num_samples=5, D=norm_t.shape[1], steps=10)
    gen_snaps = [denormalize_weights(s, normp) for s in snaps]
    gen_fin = denormalize_weights(fin, normp)
    orig_res = evaluate_weights(test_w)
    gen_res = evaluate_weights(gen_fin)
    plot_pca_progress(test_w, gen_snaps, gen_fin, out)
    plt.figure(); plt.plot(loss_h['train_losses'], label='train'); plt.plot(loss_h['val_losses'], label='val'); plt.legend(); plt.savefig(os.path.join(out,'loss.png')); plt.close()
    torch.save(model.state_dict(), os.path.join(out,'model.pt'))
    torch.save(gen_fin, os.path.join(out,'generated.pt'))
    torch.save(normp, os.path.join(out,'norm_params.pt'))
    print("Orig avg:", np.mean([r['accuracy'] for r in orig_res]))
    print("Gen avg:", np.mean([r['accuracy'] for r in gen_res]))

if __name__ == '__main__':
    main()
