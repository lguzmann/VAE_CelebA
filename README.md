# VAE_CelebA
Variational AutoEncoder trained on the CelebA dataset.

**Model weights:** vae_celeba_model.pht

**Model architecture:** vae_model.py

**Distilled smile vector:** smile_vector.pt

```python
# Loading the Model
from vae_model import VAE
latent_dim = 128
vae = VAE(latent_dim=latent_dim).to(device)
vae.load_state_dict(torch.load("vae_celeba.pth", map_location=device))
vae.eval()
```
