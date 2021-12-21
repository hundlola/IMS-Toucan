import random

import librosa.display as lbd
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing
import torch.multiprocessing
import torch.multiprocessing
import torch.multiprocessing
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeechDataset import FastSpeechDataset
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop import train_loop
from Utility.path_to_transcript_dicts import *
from Utility.utils import cumsum_durations
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    datasets = list()

    base_dir = os.path.join("Models", "FastSpeech2_MetaCheckpoint")

    print("Preparing")
    cache_dir_english_nancy = os.path.join("Corpora", "Nancy")
    os.makedirs(cache_dir_english_nancy, exist_ok=True)
    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_nancy(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=cache_dir_english_nancy,
                                      device=torch.device("cuda"),
                                      lang="en"))

    cache_dir_german_karlsson = os.path.join("Corpora", "Karlsson")
    os.makedirs(cache_dir_german_karlsson, exist_ok=True)
    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_karlsson(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=cache_dir_german_karlsson,
                                      device=torch.device("cuda"),
                                      lang="de"))

    cache_dir_greek = os.path.join("Corpora", "meta_Greek")
    os.makedirs(cache_dir_greek, exist_ok=True)
    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10el(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=cache_dir_greek,
                                      device=torch.device("cuda"),
                                      lang="el"))

    cache_dir_spanish = os.path.join("Corpora", "meta_Spanish")
    os.makedirs(cache_dir_spanish, exist_ok=True)
    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10es(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=cache_dir_spanish,
                                      device=torch.device("cuda"),
                                      lang="es"))

    cache_dir_finnish = os.path.join("Corpora", "meta_Finnish")
    os.makedirs(cache_dir_finnish, exist_ok=True)
    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10fi(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=cache_dir_finnish,
                                      device=torch.device("cuda"),
                                      lang="fi"))

    cache_dir_russian = os.path.join("Corpora", "meta_Russian")
    os.makedirs(cache_dir_russian, exist_ok=True)
    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10ru(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=cache_dir_russian,
                                      device=torch.device("cuda"),
                                      lang="ru"))

    cache_dir_hungarian = os.path.join("Corpora", "meta_Hungarian")
    os.makedirs(cache_dir_hungarian, exist_ok=True)
    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10hu(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=cache_dir_hungarian,
                                      device=torch.device("cuda"),
                                      lang="hu"))

    cache_dir_dutch = os.path.join("Corpora", "meta_Dutch")
    os.makedirs(cache_dir_dutch, exist_ok=True)
    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10nl(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=cache_dir_dutch,
                                      device=torch.device("cuda"),
                                      lang="nl"))

    cache_dir_french = os.path.join("Corpora", "meta_French")
    os.makedirs(cache_dir_french, exist_ok=True)
    datasets.append(FastSpeechDataset(build_path_to_transcript_dict_css10fr(),
                                      acoustic_checkpoint_path="Models/Aligner/aligner.pt",
                                      cache_dir=cache_dir_french,
                                      device=torch.device("cuda"),
                                      lang="fr"))

    if model_dir is not None:
        meta_save_dir = model_dir
    else:
        meta_save_dir = base_dir
    os.makedirs(meta_save_dir, exist_ok=True)

    train_loop(net=FastSpeech2(),
               device=torch.device("cuda"),
               datasets=datasets,
               batch_size=5,
               save_directory=meta_save_dir,
               steps=100000,
               steps_per_checkpoint=1000,
               lr=0.001,
               path_to_checkpoint=resume_checkpoint,
               resume=resume)


def train_loop(net,
               datasets,
               device,
               save_directory,
               batch_size,
               steps,
               steps_per_checkpoint,
               lr,
               path_to_checkpoint,
               resume=False):
    # ============
    # Preparations
    # ============
    net = net.to(device)
    train_loaders = list()
    train_iters = list()
    for dataset in datasets:
        train_loaders.append(DataLoader(batch_size=batch_size,
                                        dataset=dataset,
                                        drop_last=True,
                                        num_workers=2,
                                        pin_memory=False,
                                        shuffle=True,
                                        prefetch_factor=16,
                                        collate_fn=collate_and_pad,
                                        persistent_workers=True))
        train_iters.append(iter(train_loaders[-1]))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1.0e-06, weight_decay=0.0)
    grad_scaler = GradScaler()
    if resume:
        previous_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
        if previous_checkpoint is not None:
            path_to_checkpoint = previous_checkpoint
        else:
            raise RuntimeError(f"No checkpoint found that can be resumed from in {save_directory}")
    step_counter = 0
    train_losses_total = list()
    if path_to_checkpoint is not None:
        check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
        net.load_state_dict(check_dict["model"])
        if resume:
            optimizer.load_state_dict(check_dict["optimizer"])
            step_counter = check_dict["step_counter"]
            grad_scaler.load_state_dict(check_dict["scaler"])
            if step_counter > steps:
                print("Desired steps already reached in loaded checkpoint.")
                return

    # =============================
    # Actual train loop starts here
    # =============================
    for step in tqdm(range(step_counter, steps)):
        net.train()
        batches = []
        for index in range(len(datasets)):
            # we get one batch for each task (i.e. language in this case)
            try:
                batch = next(train_iters[index])
                batches.append(batch)
            except StopIteration:
                train_iters[index] = iter(train_loaders[index])
                batch = next(train_iters[index])
                batches.append(batch)
        train_losses = list()
        for batch in batches:
            with autocast():
                # we sum the loss for each task, as we would do for the
                # second order regular MAML, but we do it only over one
                # step (i.e. iterations of inner loop = 1)
                train_losses.append(net(text_tensors=batch[0].to(device),
                                        text_lengths=batch[1].to(device),
                                        gold_speech=batch[2].to(device),
                                        speech_lengths=batch[3].to(device),
                                        gold_durations=batch[4].to(device),
                                        gold_pitch=batch[6].to(device),  # mind the switched order
                                        gold_energy=batch[5].to(device),
                                        return_mels=False))
        # then we directly update our meta-parameters without
        # the need for any task specific parameters
        train_loss = sum(train_losses)
        train_losses_total.append(train_loss.item())
        optimizer.zero_grad()
        grad_scaler.scale(train_loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=False)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        if step % steps_per_checkpoint == 0:
            # ==============================
            # Enough steps for some insights
            # ==============================
            net.eval()
            print(f"Total Loss: {round(sum(train_losses_total) / len(train_losses_total), 3)}")
            train_losses_total = list()
            torch.save({
                "model"       : net.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "scaler"      : grad_scaler.state_dict(),
                "step_counter": step
                },
                os.path.join(save_directory, "checkpoint_{}.pt".format(step)))
            delete_old_checkpoints(save_directory, keep=5)
            for lang in ["en", "de", "el", "es", "fi", "ru", "hu", "nl", "fr"]:
                plot_progress_spec(net=net,
                                   device=device,
                                   lang=lang,
                                   save_dir=save_directory,
                                   step=step)


@torch.no_grad()
def plot_progress_spec(net, device, save_dir, step, lang):
    tf = ArticulatoryCombinedTextFrontend(language=lang)
    sentence = ""
    if lang == "en":
        sentence = "This is a complex sentence, it even has a pause!"
    elif lang == "de":
        sentence = "Dies ist ein komplexer Satz, er hat sogar eine Pause!"
    elif lang == "el":
        sentence = "Αυτή είναι μια σύνθετη πρόταση, έχει ακόμη και παύση!"
    elif lang == "es":
        sentence = "Esta es una oración compleja, ¡incluso tiene una pausa!"
    elif lang == "fi":
        sentence = "Tämä on monimutkainen lause, sillä on jopa tauko!"
    elif lang == "ru":
        sentence = "Это сложное предложение, в нем даже есть пауза!"
    elif lang == "hu":
        sentence = "Ez egy összetett mondat, még szünet is van benne!"
    elif lang == "nl":
        sentence = "Dit is een complexe zin, er zit zelfs een pauze in!"
    elif lang == "fr":
        sentence = "C'est une phrase complexe, elle a même une pause !"
    phoneme_vector = tf.string_to_tensor(sentence).squeeze(0).to(device)
    spec, durations, *_ = net.inference(text=phoneme_vector, return_duration_pitch_energy=True)
    spec = spec.transpose(0, 1).to("cpu").numpy()
    duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
    if not os.path.exists(os.path.join(save_dir, "spec")):
        os.makedirs(os.path.join(save_dir, "spec"))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    lbd.specshow(spec,
                 ax=ax,
                 sr=16000,
                 cmap='GnBu',
                 y_axis='mel',
                 x_axis=None,
                 hop_length=256)
    ax.yaxis.set_visible(False)
    ax.set_xticks(duration_splits, minor=True)
    ax.xaxis.grid(True, which='minor')
    ax.set_xticks(label_positions, minor=False)
    ax.set_xticklabels(tf.get_phone_string(sentence))
    ax.set_title(sentence)
    plt.savefig(os.path.join(os.path.join(save_dir, "spec"), f"{step}_{lang}.png"))
    plt.clf()
    plt.close()


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, energy, pitch
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[2] for datapoint in batch], batch_first=True),
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5] for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6] for datapoint in batch], batch_first=True))
