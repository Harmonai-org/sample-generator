import torch
from torch import optim
from audio_diffusion.models import LatentAudioDiffusion
from audio_diffusion.utils import get_alphas_sigmas
from ema_pytorch import EMA
from audio_diffusion.utils import InverseLR
from torch.nn import functional as F
import pytorch_lightning as pl

class LatentAudioDiffusionTrainer(pl.LightningModule):
    def __init__(self, latent_diffusion_model: LatentAudioDiffusion):
        super().__init__()

        self.diffusion = latent_diffusion_model

        self.diffusion_ema = EMA(
            self.diffusion,
            beta = 0.9999,
            power=3/4,
            update_every = 1,
            update_after_step = 1000
        )

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def configure_optimizers(self):
        optimizer = optim.Adam([*self.diffusion.parameters()], lr=1e-4)

        scheduler = InverseLR(optimizer, inv_gamma=50000, power=1/2, warmup=0.9)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        reals = batch

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                latents = self.diffusion.encode(reals)

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(latents)
        noised_latents = latents * alphas + noise * sigmas
        targets = noise * alphas - latents * sigmas

        with torch.cuda.amp.autocast():
            v = self.diffusion(noised_latents, t)
            mse_loss = F.mse_loss(v, targets)
            loss = mse_loss

        log_dict = {
            'train/loss': loss.detach(),
            'train/lr': self.lr_schedulers().get_last_lr()[0]
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.diffusion_ema.update()