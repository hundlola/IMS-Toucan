import sys
import warnings

import torch

from InferenceInterfaces.Meta_FastSpeech2 import Meta_FastSpeech2
from InferenceInterfaces.MultiEnglish_FastSpeech2 import MultiEnglish_FastSpeech2
from InferenceInterfaces.MultiGerman_FastSpeech2 import MultiGerman_FastSpeech2

tts_dict = {
    "fast_meta"   : Meta_FastSpeech2,
    "fast_german" : MultiGerman_FastSpeech2,
    "fast_english": MultiEnglish_FastSpeech2
    }

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    model_id = input("Which model do you want? \nCurrently supported are: {}\n".format("".join("\n\t- {}".format(key) for key in tts_dict.keys())))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = tts_dict[model_id](device=device)
    while True:
        text = input("\nWhat should I say? (or 'exit')\n")
        if text == "exit":
            sys.exit()
        tts.read_aloud(text, view=True, blocking=False)
