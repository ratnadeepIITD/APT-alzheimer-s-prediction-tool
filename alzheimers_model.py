# alzheimers_model.py - DEPLOYMENT VERSION (no training code)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ MODEL ARCHITECTURE (SAME AS TRAINING) ============

class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = hidden_dim ** 0.5

    def forward(self, query, keys, values):
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, values).squeeze(1)
        return context, attn_weights.squeeze(1)

class Encoder(nn.Module):
    def __init__(self, input_dim=1, static_dim=7, hidden_dim=32, latent_dim=16, 
                 num_layers=1, use_attn=False):
        super().__init__()
        self.use_attn = use_attn
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, 
                          bidirectional=False, batch_first=True)
        if use_attn:
            self.attn = ScaledDotAttention(hidden_dim)
        self.fc_static = nn.Linear(static_dim, hidden_dim)
        self.fc_comb = nn.Linear(hidden_dim * 2, latent_dim * 2)

    def forward(self, temporal_data, static_data):
        gru_out, h_n = self.gru(temporal_data)
        last_h = h_n[-1]
        if self.use_attn:
            context, _ = self.attn(last_h, gru_out, gru_out)
        else:
            context = gru_out.mean(dim=1)
        static_feat = F.relu(self.fc_static(static_data))
        combined = torch.cat([context, static_feat], dim=1)
        mu, logvar = self.fc_comb(combined).chunk(2, dim=-1)
        return mu, logvar

class ODEFunc(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, t, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=32, output_dim=1, use_attn=False):
        super().__init__()
        self.use_attn = use_attn
        if use_attn:
            self.attn = ScaledDotAttention(latent_dim)
            fc_in = latent_dim * 2
        else:
            fc_in = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * 2)
        )

    def forward(self, latent_traj, *_):
        batch_size, pred_len, latent_dim = latent_traj.size()
        contexts = []
        for t in range(pred_len):
            query = latent_traj[:, t, :]
            if self.use_attn and t > 0:
                keys = latent_traj[:, :t, :]
                context, _ = self.attn(query, keys, keys)
            else:
                context = torch.zeros_like(query)
            contexts.append(context.unsqueeze(1))
        context_all = torch.cat(contexts, dim=1)
        dec_input = torch.cat([latent_traj, context_all], dim=-1) if self.use_attn else latent_traj
        out = self.fc(dec_input)
        return out.chunk(2, dim=-1)

class ADProgressionModel(nn.Module):
    def __init__(self, input_dim=1, static_dim=7, encoder_hidden_dim=32, 
                 latent_dim=16, decoder_hidden_dim=32, noise_std=0.1,
                 enc_attn=False, dec_attn=False):
        super().__init__()
        self.encoder = Encoder(input_dim, static_dim, encoder_hidden_dim, 
                              latent_dim, use_attn=enc_attn)
        self.ode_func = ODEFunc(latent_dim)
        self.decoder = Decoder(latent_dim, decoder_hidden_dim, 
                              output_dim=1, use_attn=dec_attn)
        self.noise_std = noise_std

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, temporal_data, static_data, t_obs, t_full):
        mu, logvar = self.encoder(temporal_data, static_data)
        z0 = self.reparameterize(mu, logvar)
        t_tensor = torch.tensor(t_full, device=z0.device, dtype=torch.float)
        z_all = odeint(self.ode_func, z0, t_tensor).permute(1, 0, 2)
        # Return full trajectory for bidirectional prediction
        return self.decoder(z_all), mu, logvar

# ============ DATA PREPROCESSING FOR INFERENCE ============

def process_form_data(form_data):
    """Convert web form data to model input format - MUST MATCH TRAINING EXACTLY"""
    
    # Extract years and sort them
    years = sorted([int(year) for year in form_data.keys()])
    first_year_data = form_data[str(years[0])]
    
    # Process static features - SAME ORDER AS TRAINING!
    # Based on your original training code static_cols order:
    # ['Age','BMI','HMHYPERT','PTGENDER_Male','PTGENDER_Female'] + DIAGNOSIS_* + APOE4_*
    
    static_features = []
    
    # 1. Age
    static_features.append(float(first_year_data['age']))
    
    # 2. BMI  
    static_features.append(float(first_year_data['bmi']))
    
    # 3. HMHYPERT (Hypertension)
    static_features.append(1.0 if first_year_data['hypertension'] == 'Yes' else 0.0)
    
    # 4. PTGENDER_Male
    static_features.append(1.0 if first_year_data['gender'] == 'Male' else 0.0)
    
    # 5. PTGENDER_Female  
    static_features.append(1.0 if first_year_data['gender'] == 'Female' else 0.0)
    
    # 6-N. DIAGNOSIS dummies (get_dummies creates these in alphabetical order)
    # Normal is baseline (all zeros), MCI and AD get dummy variables
    # Order depends on what diagnoses were in your training data
    # You need to check your training data for exact order!
    
    # For now, assuming standard order (adjust based on your training data):
    diagnosis = first_year_data['diagnosis']
    
    # DIAGNOSIS_AD (if AD was in training data)
    static_features.append(1.0 if diagnosis == 'AD' else 0.0)
    
    # DIAGNOSIS_MCI (if MCI was in training data) 
    static_features.append(1.0 if diagnosis == 'MCI' else 0.0)
    
    # DIAGNOSIS_Normal is usually baseline (not included as dummy)
    
    # N+1-M. APOE4 dummies (get_dummies creates these in alphabetical order)
    # Non is baseline (all zeros), Hetero and Homo get dummy variables
    apoe4 = first_year_data['apoe4']
    
    # APOE4_Hetero (if Hetero was in training data)
    static_features.append(1.0 if apoe4 == 'Hetero' else 0.0)
    
    # APOE4_Homo (if Homo was in training data)
    static_features.append(1.0 if apoe4 == 'Homo' else 0.0)
    
    # APOE4_Non is usually baseline (not included as dummy)
    
    # Temporal data (CDRSB values)
    temporal_data = []
    obs_times = []
    
    for year in years:
        temporal_data.append(float(form_data[str(year)]['cdr_sb']))
        obs_times.append(float(year))
    
    # Convert to tensors
    static_tensor = torch.tensor([static_features], dtype=torch.float32)
    temporal_tensor = torch.tensor([temporal_data], dtype=torch.float32).unsqueeze(-1)
    
    print(f"Static features: {len(static_features)} dimensions")
    print(f"Temporal data: {len(temporal_data)} time points")
    
    return static_tensor, temporal_tensor, obs_times

# ============ PREDICTION FUNCTION ============

def predict_progression(form_data, model_path='alzheimers_model.pth', n_samples=50):
    """
    Make predictions from web form data
    
    Args:
        form_data: Dictionary with years as keys, each containing patient data
        model_path: Path to trained model file
        n_samples: Number of Monte Carlo samples for uncertainty estimation
        
    Returns:
        Dictionary with timeline, predictions, and uncertainties
    """
    
    # Load the trained model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with saved parameters
    model = ADProgressionModel(**checkpoint['model_params']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Process input data
    static_features, temporal_data, obs_times = process_form_data(form_data)
    static_features = static_features.to(device)
    temporal_data = temporal_data.to(device)
    
    print(f"Processing {len(obs_times)} observation time points: {obs_times}")
    
    # Define prediction timeline (0 to 10 years, every 6 months)
    full_timeline = np.arange(0, 10.5, 0.5).tolist()
    
    # Monte Carlo sampling for uncertainty quantification
    predictions = []
    
    print(f"Generating {n_samples} Monte Carlo samples...")
    
    with torch.no_grad():
        for i in range(n_samples):
            # Forward pass through model
            (pred_mean, pred_logvar), _, _ = model(
                temporal_data, static_features, obs_times, full_timeline
            )
            
            # Sample from predictive distribution
            pred_std = torch.exp(0.5 * pred_logvar)
            sample = pred_mean + pred_std * torch.randn_like(pred_mean)
            
            predictions.append(sample.cpu().numpy().squeeze())
    
    # Convert to numpy array
    predictions = np.array(predictions)  # Shape: (n_samples, n_timepoints)
    
    # Calculate summary statistics
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)
    pred_lower = np.percentile(predictions, 2.5, axis=0)  # 95% CI lower bound
    pred_upper = np.percentile(predictions, 97.5, axis=0)  # 95% CI upper bound
    
    print(f"Prediction completed! Timeline: {len(full_timeline)} points")
    print(f"CDRSB at 10 years: {pred_mean[-1]:.2f} ± {pred_std[-1]:.2f}")
    
    # Return results
    return {
        'timeline': full_timeline,
        'mean': pred_mean.tolist(),
        'std': pred_std.tolist(),
        'lower_ci': pred_lower.tolist(),
        'upper_ci': pred_upper.tolist(),
        'observed_times': obs_times,
        'observed_indices': [i for i, t in enumerate(full_timeline) if t in obs_times]
    }

# ============ MODEL INFO ============

def get_model_info(model_path='alzheimers_model.pth'):
    """Get information about the trained model"""
    import os
    
    if not os.path.exists(model_path):
        return {"error": "Model file not found"}
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    info = {
        "model_type": "Neural ODE with Variational Autoencoder",
        "model_params": checkpoint.get('model_params', {}),
        "file_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
        "total_parameters": 0
    }
    
    if 'model_state_dict' in checkpoint:
        for param in checkpoint['model_state_dict'].values():
            info["total_parameters"] += param.numel()
    
    if 'data_info' in checkpoint:
        info.update(checkpoint['data_info'])
    
    if 'final_loss' in checkpoint:
        info["final_training_loss"] = checkpoint['final_loss']
    
    return info

if __name__ == "__main__":
    import os
    
    # Just check if model exists - no unnecessary testing
    if os.path.exists('alzheimers_model.pth'):
        print("✅ Model file found!")
        
        info = get_model_info()
        print(f"📁 Model size: {info['file_size_mb']} MB")
        print(f"🧠 Parameters: {info['total_parameters']:,}")
        
        if 'static_dim' in info:
            print(f"📊 Expected static features: {info['static_dim']}")
            print("⚠️  Make sure web form processing matches training data order!")
    else:
        print("❌ alzheimers_model.pth not found!")
        print("Please copy your trained model file here.")
