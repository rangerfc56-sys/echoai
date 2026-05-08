---
language:
- en
- hi
tags:
- audio
- automatic-speech-recognition
- whisper-event
- pytorch
- hinglish
inference: true
model-index:
- name: Whisper-Hindi2Hinglish-Prime
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: google/fleurs
      type: google/fleurs
      config: hi_in
      split: test
    metrics:
    - type: wer
      value: 28.6806
      name: WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: mozilla-foundation/common_voice_20_0
      type: mozilla-foundation/common_voice_20_0
      config: hi
      split: test
    metrics:
    - type: wer
      value: 32.4314
      name: WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Indic-Voices
      type: Indic-Voices
      config: hi
      split: test
    metrics:
    - type: wer
      value: 60.8224
      name: WER
widget:
- src: audios/c0637211-7384-4abc-af69-5aacf7549824_1_2629072_2656224.wav
  output:
    text: Mehnat to poora karte hain.
- src: audios/c0faba11-27ba-4837-a2eb-ccd67be07f40_1_3185088_3227568.wav
  output:
    text: Haan vahi ek aapko bataaya na.
- src: audios/663eb653-d6b5-4fda-b5f2-9ef98adc0a61_0_1098400_1118688.wav
  output:
    text: Aap pandrah log hain.
- src: audios/f5e0178c-354c-40c9-b3a7-687c86240a77_1_2613728_2630112.wav
  output:
    text: Kitne saal ki?
- src: audios/f5e0178c-354c-40c9-b3a7-687c86240a77_1_1152496_1175488.wav
  output:
    text: Lander cycle chaahie.
- src: audios/c0637211-7384-4abc-af69-5aacf7549824_1_2417088_2444224.wav
  output:
    text: Haan haan, dekhe hain.
- src: audios/common_voice_hi_23796065.mp3
  example_title: Speech Example 1
- src: audios/common_voice_hi_41666099.mp3
  example_title: Speech Example 2
- src: audios/common_voice_hi_41429198.mp3
  example_title: Speech Example 3
- src: audios/common_voice_hi_41429259.mp3
  example_title: Speech Example 4
- src: audios/common_voice_hi_40904697.mp3
  example_title: Speech Example 5
pipeline_tag: automatic-speech-recognition
license: apache-2.0
metrics:
- wer
base_model:
- openai/whisper-large-v3
library_name: transformers
---
A better version of this model is available: [Oriserve/Whisper-Hindi2Hinglish-Apex](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Apex)


## Whisper-Hindi2Hinglish-Prime:
- GITHUB LINK: [github link](https://github.com/OriserveAI/Whisper-Hindi2Hinglish)
- SPEECH-TO-TEXT ARENA: [Speech-To-Text Arena](https://huggingface.co/spaces/Oriserve/ASR_arena)
### Table of Contents:
- [Key Features](#key-features)
- [Training](#training)
    - [Data](#data)
    - [Finetuning](#finetuning)
- [Usage](#usage)
- [Performance Overview](#performance-overview)
  - [Qualitative Performance Overview](#qualitative-performance-overview)
  - [Quantitative Performance Overview](#quantitative-performance-overview)
- [Miscellaneous](#miscellaneous) 

### Key Features:
1. **Hinglish as a language**: Added ability to transcribe audio into spoken Hinglish language reducing chances of grammatical errors
2. **Whisper Architecture**: Based on the whisper architecture making it easy to use with the transformers package
3. **Better Noise handling**: The model is resistant to noise and thus does not return transcriptions for audios with just noise
4. **Hallucination Mitigation**: Minimizes transcription hallucinations to enhance accuracy.
5. **Performance Increase**: ~39% average performance increase versus pretrained model across benchmarking datasets

### Training:
#### Data:
- **Duration**: A total of ~550 Hrs of noisy Indian-accented Hindi data was used to finetune the model.
- **Collection**: Due to a lack of ASR-ready hinglish datasets available, a specially curated proprietary dataset was used.
- **Labelling**: This data was then labeled using a SOTA model and the transcriptions were improved by human intervention.
- **Quality**: Emphasis was placed on collecting noisy data for the task as the intended use case of the model is in Indian environments where background noise is abundant.
- **Processing**: It was ensured that the audios are all chunked into chunks of length <30s, and there are at max 2 speakers in a clip. No further processing steps were done so as to not change the quality of the source data.

#### Finetuning:
- **Novel Trainer Architecture**: A custom trainer was written to ensure efficient supervised finetuning, with custom callbacks to enable higher observability during the training process.
- **Custom Dynamic Layer Freezing**: Most active layers were identified in the model by running inference on a subset of the training data using the pre-trained models. These layers were then kept unfrozen during the training process while all the other layers were kept frozen. This enabled faster convergence and efficient finetuning
- **Deepspeed Integration**: Deepspeed was also utilized to speed up, and optimize the training process.

### Performance Overview

#### Qualitative Performance Overview
| Audio | Whisper Large V3 | Whisper-Hindi2Hinglish-Prime |
|-------|------------------|------------------------------|
| <audio controls><source src="https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Prime/resolve/main/audios/c0637211-7384-4abc-af69-5aacf7549824_1_2629072_2656224.wav" type="audio/wav"></audio> | maynata pura, canta maynata | Mehnat to poora karte hain. |
| <audio controls><source src="https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Prime/resolve/main/audios/c0faba11-27ba-4837-a2eb-ccd67be07f40_1_3185088_3227568.wav" type="audio/wav"></audio> | Where did they come from? | Haan vahi ek aapko bataaya na. |
| <audio controls><source src="https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Prime/resolve/main/audios/663eb653-d6b5-4fda-b5f2-9ef98adc0a61_0_1098400_1118688.wav" type="audio/wav"></audio> | A Pantral Logan. | Aap pandrah log hain. |
| <audio controls><source src="https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Prime/resolve/main/audios/f5e0178c-354c-40c9-b3a7-687c86240a77_1_2613728_2630112.wav" type="audio/wav"></audio> | Thank you, Sanchez. | Kitne saal ki? |
| <audio controls><source src="https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Prime/resolve/main/audios/f5e0178c-354c-40c9-b3a7-687c86240a77_1_1152496_1175488.wav" type="audio/wav"></audio> | Rangers, I can tell you. | Lander cycle chaahie. |
| <audio controls><source src="https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Prime/resolve/main/audios/c0637211-7384-4abc-af69-5aacf7549824_1_2417088_2444224.wav" type="audio/wav"></audio> | Uh-huh. They can't. | Haan haan, dekhe hain. |


#### Quantitative Performance Overview

***Note***: 
- *The below WER scores are for Hinglish text generated by our model and the original whisper model*
- *To check our model's real-world performance against other SOTA models please head to our [Speech-To-Text Arena](https://huggingface.co/spaces/Oriserve/ASR_arena) arena space.*

| Dataset | Whisper Large V3 | Whisper-Hindi2Hinglish-Prime |
|-------|------------------------|-------------------------|
| [Common-Voice](https://commonvoice.mozilla.org/en) | 61.9432| 32.4314 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | 50.8425 | 28.6806 |
| [Indic-Voices](https://ai4bharat.iitm.ac.in/datasets/indicvoices)| 82.5621 | 60.8224 |

### Usage:
#### Using Transformers
- To run the model, first install the Transformers library

```pip install -U transformers```

- The model can be used with the [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)
class to transcribe audios of arbitrary length:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

# Set device (GPU if available, otherwise CPU) and precision
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Specify the pre-trained model ID
model_id = "Oriserve/Whisper-Hindi2Hinglish-Prime"

# Load the speech-to-text model with specified configurations
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype,        # Use appropriate precision (float16 for GPU, float32 for CPU)
    low_cpu_mem_usage=True,         # Optimize memory usage during loading
    use_safetensors=True            # Use safetensors format for better security
)
model.to(device)                    # Move model to specified device

# Load the processor for audio preprocessing and tokenization
processor = AutoProcessor.from_pretrained(model_id)

# Create speech recognition pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={
        "task": "transcribe",       # Set task to transcription
        "language": "en"            # Specify English language
    }
)

# Process audio file and print transcription
sample = "sample.wav"               # Input audio file path
result = pipe(sample)               # Run inference
print(result["text"])               # Print transcribed text
```

#### Using Flash Attention 2

Flash-Attention 2 can be used to make the transcription fast. If your GPU supports Flash-Attention you can use it by, first installing Flash Attention:

```pip install flash-attn --no-build-isolation```

- Once installed you can then load the model using the below code:

```python
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
```

#### Using the OpenAI Whisper module

- First, install the openai-whisper library

```pip install -U openai-whisper tqdm```

- Convert the huggingface checkpoint to a pytorch model

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq
import re
from tqdm import tqdm
from collections import OrderedDict
import json

# Load parameter name mapping from HF to OpenAI format
with open('convert_hf2openai.json', 'r') as f:
    reverse_translation = json.load(f)

reverse_translation = OrderedDict(reverse_translation)

def save_model(model, save_path):
    def reverse_translate(current_param):
        # Convert parameter names using regex patterns
        for pattern, repl in reverse_translation.items():
            if re.match(pattern, current_param):
                return re.sub(pattern, repl, current_param)

    # Extract model dimensions from config
    config = model.config
    model_dims = {
        "n_mels": config.num_mel_bins,           # Number of mel spectrogram bins
        "n_vocab": config.vocab_size,            # Vocabulary size
        "n_audio_ctx": config.max_source_positions,    # Max audio context length
        "n_audio_state": config.d_model,         # Audio encoder state dimension
        "n_audio_head": config.encoder_attention_heads,  # Audio encoder attention heads
        "n_audio_layer": config.encoder_layers,   # Number of audio encoder layers
        "n_text_ctx": config.max_target_positions,     # Max text context length
        "n_text_state": config.d_model,          # Text decoder state dimension
        "n_text_head": config.decoder_attention_heads,  # Text decoder attention heads
        "n_text_layer": config.decoder_layers,    # Number of text decoder layers
    }

    # Convert model state dict to Whisper format
    original_model_state_dict = model.state_dict()
    new_state_dict = {}

    for key, value in tqdm(original_model_state_dict.items()):
        key = key.replace("model.", "")          # Remove 'model.' prefix
        new_key = reverse_translate(key)         # Convert parameter names
        if new_key is not None:
            new_state_dict[new_key] = value

    # Create final model dictionary
    pytorch_model = {"dims": model_dims, "model_state_dict": new_state_dict}

    # Save converted model
    torch.save(pytorch_model, save_path)

# Load Hugging Face model
model_id = "Oriserve/Whisper-Hindi2Hinglish-Prime"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    low_cpu_mem_usage=True,        # Optimize memory usage
    use_safetensors=True           # Use safetensors format
)

# Convert and save model
model_save_path = "Whisper-Hindi2Hinglish-Prime.pt"
save_model(model,model_save_path)
```

- Transcribe

```python
import whisper
# Load converted model with Whisper and transcribe
model = whisper.load_model("Whisper-Hindi2Hinglish-Prime.pt")
result = model.transcribe("sample.wav")
print(result["text"])
```


### Miscellaneous
This model is from a family of transformers-based ASR models trained by Oriserve. To compare this model against other models from the same family or other SOTA models please head to our [Speech-To-Text Arena](https://huggingface.co/spaces/Oriserve/ASR_arena). To learn more about our other models, and other queries regarding AI voice agents you can reach out to us at our email [ai-team@oriserve.com](ai-team@oriserve.com)