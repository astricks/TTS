import os
import torch
import time

from TTS.tts.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.tts.utils.speakers import load_speaker_mapping, load_language_mapping
from TTS.tts.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.synthesis import synthesis


TTS_MODEL = "../output/multilingual-v0.1.3-btcsessions-August-09-2021_09+08AM-cc7e7171/checkpoint_300000.pth.tar"
TTS_CONFIG = "../output/multilingual-v0.1.3-btcsessions-August-09-2021_09+08AM-cc7e7171/config.json"
TTS_LANGUAGES = "../output/multilingual-v0.1.3-btcsessions-August-09-2021_09+08AM-cc7e7171/languages.json"
TTS_SPEAKERS = "../output/multilingual-v0.1.3-btcsessions-August-09-2021_09+08AM-cc7e7171/speakers.json"
VOCODER_MODEL = "/home/arvind/.local/share/tts/vocoder_models--en--ek1--wavegrad/model_file.pth.tar"
VOCODER_CONFIG = "/home/arvind/.local/share/tts/vocoder_models--en--ek1--wavegrad/config.json"


def tts(model, text, CONFIG, use_cuda, ap, speaker_id=None, language_id=None, language_mapping=None, figures=True, speaker_embedding=None):
    t_1 = time.time()
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(model, text, CONFIG, use_cuda, ap, speaker_id=speaker_id, language_id=language_id, language_mapping=language_mapping, style_wav=None, truncated=False, enable_eos_bos_chars=CONFIG.enable_eos_bos_chars, use_griffin_lim=True, speaker_embedding=speaker_embedding)
    return alignment, mel_postnet_spec, stop_tokens, waveform


use_cuda = True
TTS_CONFIG = load_config(TTS_CONFIG)
TTS_CONFIG.audio['stats_path'] = None
ap = AudioProcessor(**TTS_CONFIG.audio)

# LOAD TTS MODEL

# Load speakers and languages
speaker_mapping = load_speaker_mapping(TTS_SPEAKERS)
language_mapping = load_language_mapping(TTS_LANGUAGES)

# load the model
num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
model = setup_model(num_chars, len(speaker_mapping), len(language_mapping), TTS_CONFIG, speaker_embedding_dim=512)

# load model state
cp =  torch.load(TTS_MODEL, map_location=torch.device('cpu'))

# load the model
model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()

# set model stepsize
if 'r' in cp:
    model.decoder.set_r(cp['r'])

sp_emb = speaker_mapping["21_42.wav"]["embedding"]
print(sp_emb)

sentence =  "Y ahora el final está aquí y así me enfrento al telón final, amigo mío lo diré claro, expondré mi caso del que estoy seguro."
align, spec, stop_tokens, wav = tts(model, sentence, TTS_CONFIG, use_cuda, ap, language_id=1, language_mapping=language_mapping, figures=True, speaker_embedding=sp_emb)
ap.save_wav(wav, '/tmp/btc1.wav')

print("done")
