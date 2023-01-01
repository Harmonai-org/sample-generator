#!/usr/bin/env python3

from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
from copy import deepcopy
import math

import sys
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange
import numpy as np
import torchaudio
import wandb

from audio_diffusion.autoencoders import AudioAutoencoder
from audio_diffusion.models import LatentAudioDiffusion
from audio_diffusion.utils import get_alphas_sigmas
# from ema_pytorch import EMA
# from audio_diffusion.utils import InverseLR
from aeiou.viz import embeddings_table, pca_point_cloud, audio_spectrogram_image, tokens_spectrogram_image
from aeiou.datasets import AudioDataset
from trainers.trainers import LatentAudioDiffusionTrainer

@torch.no_grad()
def sample(model, x, steps, eta):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i]).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class DemoCallback(pl.Callback):
    def __init__(self, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.num_demos = global_args.num_demos
        self.sample_rate = global_args.sample_rate
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    #def on_train_epoch_end(self, trainer, module):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):   
        
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return
        
        self.last_demo_step = trainer.global_step
        
        print("Starting demo")
        try:
            latent_noise = torch.randn([self.num_demos, module.diffusion.latent_dim, self.demo_samples//module.diffusion.downsampling_ratio]).to(module.device)
            fake_latents = sample(module.diffusion_ema.ema_model, latent_noise, self.demo_steps, 0)
            print("Decoding fakes")
            fakes = module.diffusion.decode(fake_latents)

            # Put the demos together
            fakes = rearrange(fakes, 'b d n -> d (b n)')

            log_dict = {}
            
            print("Saving files")
            filename = f'demo_{trainer.global_step:08}.wav'
            fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fakes, self.sample_rate)


            log_dict[f'demo'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Reconstructed')
        

            log_dict[f'demo_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))

            log_dict[f'embeddings'] = embeddings_table(fake_latents)
            log_dict[f'embeddings_3dpca'] = pca_point_cloud(fake_latents)
            log_dict[f'embeddings_spec'] = wandb.Image(tokens_spectrogram_image(fake_latents))


            print("Done logging")
            trainer.logger.experiment.log(log_dict, step=trainer.global_step)

        except Exception as e:
            print(f'{type(e).__name__}: {e}')

def main():

    args = get_all_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(args.seed)

    
    wandb_logger = pl.loggers.WandbLogger(project=args.name, log_model='all' if args.save_wandb=='all' else None)
    
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(args)

    #Default parameters
    autoencoder_config = {"channels": 64, "c_mults": [2, 4, 8, 16, 32], "strides": [2, 2, 2, 2, 2], "latent_dim": 32}
    latent_diffusion_config = {"io_channels": 32, "n_attn_layers": 4, "channels": [512]*6 + [1024]*4, "depth": 10}

    #Create the autoencoder
    autoencoder = AudioAutoencoder(**autoencoder_config).requires_grad_(False)

    #Read LD config from file
    if args.model_path:
        model_info = torch.load(args.model_path)

        #Overwrite latent diffusion config if provided
        latent_diffusion_config = model_info["ld_config"]

    latent_diffusion_model = LatentAudioDiffusion(autoencoder, **latent_diffusion_config)

    if args.model_path:
        latent_diffusion_model.load_state_dict(model_info["ld_state_dict"])

    latent_diffusion_model.autoencoder.load_state_dict(torch.load(args.ae_ckpt_path)["state_dict"])

    latent_trainer = LatentAudioDiffusionTrainer(latent_diffusion_model)
    wandb_logger.watch(latent_diffusion_model)

    train_set = AudioDataset(
        [args.training_dir],
        sample_rate=args.sample_rate,
        sample_size=args.sample_size,
        random_crop=args.random_crop,
        augs='Stereo(), PhaseFlipper()'
    )

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    push_wandb_config(wandb_logger, args)

    diffusion_trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy='ddp_find_unused_parameters_false' if args.num_gpus > 1 else None,
        precision=16,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    diffusion_trainer.fit(latent_trainer, train_dl, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()

