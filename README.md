# sample-generator
Tools to train a generative model on arbitrary audio samples

Dance Diffusion notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harmonai-org/sample-generator/blob/main/Dance_Diffusion.ipynb)

Dance Diffusion fine-tune notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harmonai-org/sample-generator/blob/main/Finetune_Dance_Diffusion.ipynb)

## **🤗 Diffusers library**

Dance Diffusion is now easily accesible with the Hugging Face Hub and the 🧨 Diffusers library.

1. Install diffusers
```
pip install diffusers[torch] accelerate scipy
```

Now you can generate music files with just 10 lines of code:

```python
from diffusers import DiffusionPipeline
from scipy.io.wavfile import write

model_id = "harmonai/maestro-150k"
pipe = DiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")

audios = pipe(audio_length_in_s=4.0).audios

# To save locally
for i, audio in enumerate(audios):
    write(f"maestro_test_{i}.wav", pipe.unet.sample_rate, audio.transpose())
    
# To dislay in google colab
import IPython.display as ipd
for audio in audios:
    display(ipd.Audio(audio, rate=pipe.unet.sample_rate))
```

### Checkpoints with examples on the Hub

- [maestro-150k](https://huggingface.co/harmonai/maestro-150k)
- [jmann-small-190k](https://huggingface.co/harmonai/jmann-small-190k)
- [honk-140k](https://huggingface.co/harmonai/honk-140k)
- [unlocked-250k](https://huggingface.co/harmonai/unlocked-250k)
- [glitch-440k](https://huggingface.co/harmonai/glitch-440k)
- [jmann-large-580k](https://huggingface.co/harmonai/jmann-large-580k)


### Demo

🚀 You can try out a demo of dance diffusion under [harmonai/dance-diffusion](https://huggingface.co/spaces/harmonai/dance-diffusion)

## Prerequisites
Dance Diffusion requires Python 3.7+

You can install the required packages by running `pip install .` from the root of the repo

## Todo

- [x] Add inference notebook
- [x] Add interpolations to nobebook
- [x] Add fine-tune notebook
- [ ] Add guidance to notebook
