# # latent_ddpm_hypersearch.py

# import os
# import glob
# import itertools
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision import datasets, transforms
# import numpy as np
# from tqdm import tqdm
# from collections import OrderedDict
# from diffusers import DDPMScheduler

# # Fix seeds for reproducibility
# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
# set_seed()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")



# # --- LeNet5 Definition ---
# class LeNet5(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.features = nn.Sequential(OrderedDict([
#             ('conv1', nn.Conv2d(1, 6, 5, padding=2)),
#             ('relu1', nn.ReLU()),
#             ('pool1', nn.AvgPool2d(2)),
#             ('conv2', nn.Conv2d(6, 16, 5)),
#             ('relu2', nn.ReLU()),
#             ('pool2', nn.AvgPool2d(2)),
#         ]))
#         self.classifier = nn.Sequential(OrderedDict([
#             ('fc1', nn.Linear(16*5*5, 120)),
#             ('relu3', nn.ReLU()),
#             ('fc2', nn.Linear(120, 84)),
#             ('relu4', nn.ReLU()),
#             ('fc3', nn.Linear(84, 10)),
#         ]))
#         self.param_dim = sum(p.numel() for p in self.parameters() if p.requires_grad)

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         return self.classifier(x)

#     def get_flat_weights(self):
#         return torch.cat([p.detach().flatten() for p in self.parameters()])

#     def set_flat_weights(self, flat):
#         idx = 0
#         for p in self.parameters():
#             num = p.numel()
#             p.data.copy_(flat[idx:idx+num].view_as(p))
#             idx += num

# # --- Load Weights ---
# def load_weights(weights_dir, ratio=0.95):
#     files = sorted(glob.glob(os.path.join(weights_dir, '*.pt')))
#     D = LeNet5().param_dim
#     weights = []
#     for f in files:
#         w = torch.load(f, map_location='cpu', weights_only=True)
#         if w.numel() == D:
#             weights.append(w)
#     W = torch.stack(weights)
#     split = int(len(W) * ratio)
#     return W[:split], W[split:]

# # --- Normalize and Denormalize ---
# def normalize(train, test=None):
#     mean = train.mean(0)
#     std = train.std(0)
#     std[std < 1e-6] = 1e-6
#     ntrain = (train - mean) / std
#     ntest = ((test - mean) / std) if test is not None else None
#     return ntrain, ntest, {'mean': mean, 'std': std}

# def denormalize(x, norm):
#     return x * norm['std'].to(x.device) + norm['mean'].to(x.device)

# # --- Autoencoder ---
# class AE(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super().__init__()
#         self.enc = nn.Linear(input_dim, latent_dim)
#         self.dec = nn.Linear(latent_dim, input_dim)

#     def forward(self, x):
#         return self.dec(self.enc(x))

#     def encode(self, x):
#         return self.enc(x)

# # --- Scheduler and Light DDPM ---
# def get_scheduler(T):
#     return DDPMScheduler(
#         beta_start=1e-4,
#         beta_end=2e-2,
#         beta_schedule='linear',
#         num_train_timesteps=T,
#         clip_sample=False
#     )

# class DDPMLight(nn.Module):
#     def __init__(self, dim, scheduler):
#         super().__init__()
#         self.sched = scheduler
#         self.net = nn.Sequential(
#             nn.Linear(dim + 1, 512), nn.ReLU(),
#             nn.Linear(512, 512), nn.ReLU(),
#             nn.Linear(512, dim)
#         )

#     def forward(self, x, t):
#         t_norm = t.float().unsqueeze(-1) / self.sched.config.num_train_timesteps
#         inp = torch.cat([x, t_norm.to(x.device)], dim=1)
#         return self.net(inp)

#     def add_noise(self, x, t):
#         noise = torch.randn_like(x)
#         noisy = self.sched.add_noise(x, noise, t)
#         return noisy, noise

# # --- Train Autoencoder ---
# def train_ae(data, dims=[64, 128, 256, 512], epochs=100, bs=128, lr=1e-3):
#     input_dim = data.shape[1]
#     best_model = None
#     best_loss = float('inf')
#     for ld in dims:
#         ae = AE(input_dim, ld).to(device)
#         opt = optim.Adam(ae.parameters(), lr=lr)
#         loader = DataLoader(TensorDataset(data), batch_size=bs, shuffle=True)
#         for ep in range(epochs):
#             total = 0.0
#             for (x,) in loader:
#                 x = x.to(device)
#                 recon = ae(x)
#                 loss = F.mse_loss(recon, x)
#                 opt.zero_grad()
#                 loss.backward()
#                 opt.step()
#                 total += loss.item()
#         avg = total / len(loader)
#         if avg < best_loss:
#             best_loss = avg
#             best_model = ae
#     return best_model

# # --- Train DDPM in Latent Space ---
# def train_ddpm(latent_train, latent_test, T, epochs, bs, lr):
#     scheduler = get_scheduler(T)
#     model = DDPMLight(latent_train.shape[1], scheduler).to(device)
#     opt = optim.AdamW(model.parameters(), lr=lr)
#     loader = DataLoader(TensorDataset(latent_train), batch_size=bs, shuffle=True)
#     vloader = DataLoader(TensorDataset(latent_test), batch_size=bs) if latent_test is not None else None

#     train_losses, val_losses = [], []
#     for ep in range(epochs):
#         model.train(); tr_loss = 0.0
#         for (x,) in loader:
#             x = x.to(device)
#             t = torch.randint(0, T, (x.size(0),), device=device)
#             noisy, noise = model.add_noise(x, t)
#             pred = model(noisy, t)
#             loss = F.mse_loss(pred, noise)
#             opt.zero_grad(); loss.backward(); opt.step()
#             tr_loss += loss.item()
#         train_losses.append(tr_loss / len(loader))

#         if vloader is not None:
#             model.eval(); vl = 0.0
#             with torch.no_grad():
#                 for (x,) in vloader:
#                     x = x.to(device)
#                     t = torch.randint(0, T, (x.size(0),), device=device)
#                     noisy, noise = model.add_noise(x, t)
#                     vl += F.mse_loss(model(noisy, t), noise).item()
#             val_losses.append(vl / len(vloader))
#         print(f"DDPM Ep {ep+1}/{epochs}: Train {train_losses[-1]:.4f}, Val {val_losses[-1] if val_losses else 'N/A'}")

#     return model, scheduler, train_losses, val_losses

# # --- Hyperparameter Search ---
# def hyper_search(train_w, test_w):
#     # Normalize weights
#     ntrain, ntest, normp = normalize(train_w, test_w)
#     # Autoencoder search
#     ae = train_ae(ntrain.cpu(), dims=[64,128,256,512], epochs=100, bs=128, lr=1e-3)
#     latent_train = ae.encode(ntrain.to(device)).detach().cpu()
#     latent_test = ae.encode(ntest.to(device)).detach().cpu() if ntest is not None else None

#     best = None
#     # Expanded grid
#     grid = {
#         'T': [500, 1000],
#         'epochs': [50, 100],
#         'bs': [32, 64, 128],
#         'lr': [1e-4, 5e-4, 1e-3],
#         'latent_dim': [128, 256, 512]  # ✅ Add this line
#     }

#     for T, ep, bs, lr, ld in itertools.product(grid['T'], grid['epochs'], grid['bs'], grid['lr'], grid['latent_dim']):
#         print(f"Grid: T={T}, epochs={ep}, bs={bs}, lr={lr}, latent_dim={ld}")
#         ae = train_ae(ntrain.cpu(), dims=[ld], epochs=100, bs=128, lr=1e-3)
#         latent_train = ae.encode(ntrain.to(device)).detach().cpu()
#         latent_test = ae.encode(ntest.to(device)).detach().cpu() if ntest is not None else None
#         mdl, sched, trl, vll = train_ddpm(latent_train, latent_test, T, ep, bs, lr)

#     return ae, normp, best

# # --- Main ---
# def main():
#     out = 'latent_hf_search'
#     os.makedirs(out, exist_ok=True)
#     train_w, test_w = load_weights('lenet5_weights')
#     ae, normp, best = hyper_search(train_w, test_w)
#     score, T, ep, bs, lr, mdl, sched, trl, vll = best
#     # Save best
#     torch.save({
#         'ae_state': ae.state_dict(),
#         'ddpm_state': mdl.state_dict(),
#         'norm_params': normp,
#         'config': {'T': T, 'epochs': ep, 'bs': bs, 'lr': lr},
#         'train_loss': trl,
#         'val_loss': vll
#     }, os.path.join(out, 'best_models.pt'))
#     # Plot losses
#     import matplotlib.pyplot as plt
#     plt.figure(); plt.plot(trl, label='train'); plt.plot(vll, label='val'); plt.legend(); plt.title('Best DDPM Loss'); plt.savefig(os.path.join(out, 'best_loss.png'))
#     print(f"Best config -> T: {T}, epochs: {ep}, bs: {bs}, lr: {lr}, val_loss: {score:.4f}")

# if __name__ == '__main__':
#     main()


# latent_ddpm_hf_search.py (updated)

import os
import glob
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 6, 5, padding=2)), ('relu1', nn.ReLU()), ('pool1', nn.AvgPool2d(2)),
            ('conv2', nn.Conv2d(6, 16, 5)), ('relu2', nn.ReLU()), ('pool2', nn.AvgPool2d(2)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(16 * 5 * 5, 120)), ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(120, 84)), ('relu4', nn.ReLU()), ('fc3', nn.Linear(84, 10)),
        ]))
        self.param_dim = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x): return self.classifier(torch.flatten(self.features(x), 1))
    def get_flat_weights(self): return torch.cat([p.detach().flatten() for p in self.parameters()])
    def set_flat_weights(self, flat):
        i = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[i:i+n].view_as(p))
            i += n

def load_weights(dir, ratio=0.95):
    files = sorted(glob.glob(os.path.join(dir, '*.pt')))
    D = LeNet5().param_dim
    weights = [torch.load(f, map_location='cpu', weights_only=True) for f in files if torch.load(f, map_location='cpu').numel() == D]
    W = torch.stack(weights)
    split = int(len(W) * ratio)
    return W[:split], W[split:]

def normalize(train, test=None):
    mean, std = train.mean(0), train.std(0).clamp(min=1e-6)
    return (train - mean) / std, (test - mean) / std if test is not None else None, {'mean': mean, 'std': std}

def denormalize(x, norm): return x * norm['std'].to(x.device) + norm['mean'].to(x.device)

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc = nn.Linear(input_dim, latent_dim)
        self.dec = nn.Linear(latent_dim, input_dim)
    def forward(self, x): return self.dec(self.enc(x))
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)

def get_scheduler(T):
    return DDPMScheduler(beta_start=1e-4, beta_end=2e-2, beta_schedule='linear', num_train_timesteps=T, clip_sample=False)

class DDPMLight(nn.Module):
    def __init__(self, dim, scheduler):
        super().__init__()
        self.sched = scheduler
        self.net = nn.Sequential(nn.Linear(dim + 1, 512), nn.ReLU(),
                                 nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, dim))

    def forward(self, x, t):
        t_norm = t.float().unsqueeze(-1) / self.sched.config.num_train_timesteps
        return self.net(torch.cat([x, t_norm.to(x.device)], dim=1))
    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        return self.sched.add_noise(x, noise, t), noise

def train_ae(data, dims=[64, 128, 256, 512], epochs=100, bs=128, lr=1e-3):
    input_dim = data.shape[1]
    best_model, best_loss = None, float('inf')
    for d in dims:
        ae = AE(input_dim, d).to(device)
        opt = optim.Adam(ae.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(data), batch_size=bs, shuffle=True)
        for _ in range(epochs):
            for (x,) in loader:
                x = x.to(device)
                loss = F.mse_loss(ae(x), x)
                opt.zero_grad(); loss.backward(); opt.step()
        final_loss = sum(F.mse_loss(ae(x.to(device)), x.to(device)).item() for (x,) in loader) / len(loader)
        if final_loss < best_loss:
            best_loss, best_model = final_loss, ae
    return best_model

def train_ddpm(latent_train, latent_test, T, epochs, bs, lr):
    sched = get_scheduler(T)
    model = DDPMLight(latent_train.shape[1], sched).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(latent_train), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(latent_test), batch_size=bs) if latent_test is not None else None
    trl, vll = [], []
    for ep in range(epochs):
        model.train(); tl = 0
        for (x,) in train_loader:
            x = x.to(device)
            t = torch.randint(0, T, (x.size(0),), device=device)
            noisy, noise = model.add_noise(x, t)
            loss = F.mse_loss(model(noisy, t), noise)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()
        trl.append(tl / len(train_loader))
        if val_loader:
            model.eval(); vl = 0
            with torch.no_grad():
                for (x,) in val_loader:
                    x = x.to(device)
                    t = torch.randint(0, T, (x.size(0),), device=device)
                    noisy, noise = model.add_noise(x, t)
                    vl += F.mse_loss(model(noisy, t), noise).item()
            vll.append(vl / len(val_loader))
        print(f"DDPM Ep {ep+1}/{epochs}: Train {trl[-1]:.4f}, Val {vll[-1] if val_loader else 'N/A'}")
    return model, sched, trl, vll

def evaluate(weights, N=5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=1000)
    res = []
    for i in range(min(N, len(weights))):
        model = LeNet5().to(device); model.set_flat_weights(weights[i].to(device)); model.eval()
        correct = sum((model(x.to(device)).argmax(1) == y.to(device)).sum().item() for x,y in loader)
        res.append({'idx': i, 'accuracy': 100 * correct / len(loader.dataset)})
    return res

def pca_progression_plot(z_list, save_path='pca_progression.png'):
    pca = PCA(n_components=2)
    all_z = torch.cat(z_list, dim=0).cpu().numpy()
    proj = pca.fit_transform(all_z)
    start, step = 0, len(z_list[0])
    plt.figure(figsize=(10, 7))
    for i, z in enumerate(z_list):
        sub = proj[start:start+step]
        plt.scatter(sub[:, 0], sub[:, 1], label=f"Step {i}" if i > 0 else "Init", alpha=0.6)
        start += step
    plt.legend(); plt.grid(True); plt.title("PCA Progression of Samples")
    plt.savefig(save_path); plt.close()

def sample_progression(model, scheduler, latent_dim, N=5, steps=10):
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    x = torch.randn(N, latent_dim).to(device)
    traj = [x.clone().cpu()]
    step_idxs = set(np.linspace(0, scheduler.config.num_train_timesteps-1, steps, dtype=int))
    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            noise = model(x, torch.full((N,), t, device=device, dtype=torch.long))
        x = scheduler.step(noise, t, x).prev_sample
        if i in step_idxs: traj.append(x.clone().cpu())
    return traj

def main():
    out = "latent_ddpm_hf_search_eval"
    os.makedirs(out, exist_ok=True)
    train_w, test_w = load_weights("lenet5_weights")
    ntrain, ntest, normp = normalize(train_w, test_w)

    ae = train_ae(ntrain)
    z_train = ae.encode(ntrain.to(device)).detach().cpu()
    z_test = ae.encode(ntest.to(device)).detach().cpu()
    ddpm, scheduler, _, _ = train_ddpm(z_train, z_test, T=1000, epochs=100, bs=64, lr=1e-4)

    # Sample and decode
    traj_z = sample_progression(ddpm, scheduler, z_train.shape[1], N=5, steps=10)
    decoded_weights = [denormalize(ae.decode(z.to(device)).cpu(), normp) for z in traj_z]
    final_weights = decoded_weights[-1]

    # Evaluate and plot
    results = evaluate(final_weights)
    pca_progression_plot(traj_z, save_path=os.path.join(out, 'pca_progression.png'))
    print("Generated weights average accuracy:", np.mean([r['accuracy'] for r in results]))

if __name__ == "__main__":
    main()
