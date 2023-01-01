import argparse 
import torch

from audio_diffusion.autoencoders import AudioAutoencoder
from audio_diffusion.models import LatentAudioDiffusion
from trainers.trainers import LatentAudioDiffusionTrainer
from torch.nn.parameter import Parameter

def prune_ckpt_weights(trainer_state_dict):
  new_state_dict = {}
  for name, param in trainer_state_dict.items():
      if name.startswith("diffusion_ema.ema_model."):
          new_name = name.replace("diffusion_ema.ema_model.", "diffusion.")
          if isinstance(param, Parameter):
              # backwards compatibility for serialized parameters
              param = param.data
          new_state_dict[new_name] = param
          
  return new_state_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt-path', help='Path to the checkpoint to be pruned')
    parser.add_argument('--sample-rate', help='Sample rate used for the model')
    parser.add_argument('--ld-sample-size', help='Number of samples used during training for the latent diffusion model')
    args = parser.parse_args()

    print("Creating the model...")

    ae_config = {"channels": 64, "c_mults": [2, 4, 8, 16, 32], "strides": [2, 2, 2, 2, 2], "latent_dim": 32}

    autoencoder = AudioAutoencoder( 
        **ae_config
    ).eval()

    latent_diffusion_config = {"io_channels": 32, "n_attn_layers": 4, "channels": [512]*6 + [1024]*4, "depth": 10}

    #Create the diffusion model itself
    latent_diffusion_model = LatentAudioDiffusion(autoencoder, **latent_diffusion_config)

    #Create the trainer
    ld_trainer = LatentAudioDiffusionTrainer(latent_diffusion_model)

    trainer_state_dict = torch.load(args.ckpt_path)["state_dict"]
    print(trainer_state_dict.keys())

    new_ckpt = {}

    new_ckpt["ld_state_dict"] = prune_ckpt_weights(trainer_state_dict)
    new_ckpt["ld_config"] = latent_diffusion_config
    new_ckpt["sample_rate"] = args.sample_rate
    new_ckpt["ld_sample_size"] = args.ld_sample_size

    latent_diffusion_model.load_state_dict(new_ckpt["ld_state_dict"])

    torch.save(new_ckpt, f'./pruned.ckpt')