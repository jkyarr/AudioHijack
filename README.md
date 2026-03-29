# AudioHijack
This is the artifact of the IEEE S&P 2026 paper "Hijacking Large Audio-Language Models via Context-agnostic Auditory Prompt Injection".

## Hardware Requirements
- CPU >= 32GB
- GPU >= 48GB
- Disk >= 1T

## Software Requirements
- GPU Driver: CUDA >= 12.1
- Package Manager: uv
- LLM API Keys: GPT, Gemini, Qwen

## Enviroment Setup
1. Download the project to `/path/to/AudioHijack`
```
git clone git@github.com:zju-muslab/AudioHijack.git
```

Then set the root dir of this project in `config/run_attack.yaml`:
```
root_dir: /path/to/AudioHijack
```

2. Configure api_key variables in `.env`:
```
OPENAI_API_KEY= 
GOOGLE_API_KEY=
DASHSCOPE_API_KEY=
```

3. Create a venv environment with necessary packages:
```
cd /path/to/AudioHijack
pip install uv
uv sync
source .venv/bin/activate
```

## LALM Download
Download LALM weights to `weight/lalm`, with corresponding encoder weights to `weight/encoder` and backbone weights to `weight/backbone`:

| LALM | Encoder | Backbone |
|----------|-----------|-----------|
| [SpeechGPT](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt) | [mHuBERT Base](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/textless_s2st_real_data.md)| |
| [GLM-4-Voice](https://huggingface.co/zai-org/glm-4-voice-9b) | [GLM-4-Voice-tokenizer](https://huggingface.co/zai-org/glm-4-voice-tokenizer) | |
| [VITA-Audio](https://huggingface.co/VITA-MLLM/VITA-Audio-Boost) | [GLM-4-Voice-tokenizer](https://huggingface.co/zai-org/glm-4-voice-tokenizer) | |
| [Llama-Omni](https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni) | [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | |
| [Llama-Omni2](https://huggingface.co/ICTNLP/LLaMA-Omni2-7B-Bilingual) | [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | |
| [SALMONN](https://huggingface.co/tsinghua-ee/SALMONN-7B) | [Whisper-large-v2](https://huggingface.co/openai/whisper-large-v2), [BEATs](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBcWVCeWhHVXRJTnJnY3BqOHVqWEgxWVV0eG9vRWc%5FZT1FOU5jZWE&cid=6B83B49411CA81A7&id=6B83B49411CA81A7%2125955&parId=6B83B49411CA81A7%2125952&o=OneUp) | [Vicuna-7B-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| [Qwen-Audio](https://huggingface.co/Qwen/Qwen-Audio-Chat) | | |
| [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) |  | |
| [Gemma-3n](https://huggingface.co/google/gemma-3n-E2B-it) |  | |
| [Ultravox-v5](https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_1-8b) | [Whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| [Phi-4-Multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) |  | |
| [Voxtral-mini](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) |  | |
| [Kimi-Audio](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct) | [GLM-4-Voice-tokenizer](https://huggingface.co/zai-org/glm-4-voice-tokenizer), [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | |

## Dataset Download
Download audio-text data samples from our [Zenodo repo](https://doi.org/10.5281/zenodo.19309781) to the `data` directory.

## Attack Evaluation
1. Run the following script to train and test adversarial audio:
```
python run_attack.py lalm=voxtral_mini attack=caa
```
The `lalm` and `attack` options can be modified to test different lalms and attack variants in the `config` directory. 

2. Perform behavior match evaluation using OpenAI's batch inference API, following the jupyter notebook `run_judge.ipynb`.

3. Calculate the perception metrics, following the jupyter notebook `run_percept.ipynb`

**Note**:
- It will take 3~10 hours for training and testing the attack on an LALM with all 15 target behaviors.
- The PISR and BMSR across different misbehavior categories are summarized in the output, and detailed result of all attack trials are recorded in `exp/attack/${lalm}-${attack}`.