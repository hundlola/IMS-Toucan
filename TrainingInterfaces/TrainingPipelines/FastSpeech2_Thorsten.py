import os
import random

import torch

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeechDataset import FastSpeechDataset
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop import train_loop
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_thorsten as build_path_to_transcript_dict


def run(gpu_id, resume_checkpoint, finetune, model_dir):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(13)
    random.seed(13)

    print("Preparing")
    cache_dir = os.path.join("Corpora", "Thorsten")
    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_Thorsten")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict()

    acoustic_model = Tacotron2()
    acoustic_model.load_state_dict(torch.load(os.path.join("Models", "Tacotron2_Thorsten", "best.pt"), map_location='cpu')["model"])

    train_set = FastSpeechDataset(path_to_transcript_dict,
                                  cache_dir=cache_dir,
                                  acoustic_model=acoustic_model,
                                  lang="de",
                                  device=device)

    model = FastSpeech2(spk_embed_dim=None)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               device=device,
               save_directory=save_dir,
               steps=300000,
               batch_size=24,
               epochs_per_save=10,
               use_speaker_embedding=False,
               lang="de",
               lr=0.001,
               warmup_steps=14000,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune)
