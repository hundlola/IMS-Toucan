import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Dio
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.AlignmentLoss import mas_width1
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.TacotronDataset import TacotronDataset
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2 import Tacotron2

class FastSpeechDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 acoustic_checkpoint_path,
                 cache_dir,
                 lang,
                 loading_processes=6,
                 min_len_in_seconds=1,
                 max_len_in_seconds=20,
                 cut_silence=False,
                 reduction_factor=1,
                 device=torch.device("cpu"),
                 rebuild_cache=False):

        if not os.path.exists(os.path.join(cache_dir, "fast_train_cache.pt")) or rebuild_cache:
            if not os.path.exists(os.path.join(cache_dir, "taco_train_cache.pt")) or rebuild_cache:
                TacotronDataset(path_to_transcript_dict=path_to_transcript_dict,
                                cache_dir=cache_dir,
                                lang=lang,
                                loading_processes=loading_processes,
                                min_len_in_seconds=min_len_in_seconds,
                                max_len_in_seconds=max_len_in_seconds,
                                cut_silences=cut_silence,
                                rebuild_cache=rebuild_cache)
            datapoints = torch.load(os.path.join(cache_dir, "taco_train_cache.pt"), map_location='cpu')
            # we use the tacotron dataset as basis and augment it to contain the additional information we need for fastspeech.
            if not isinstance(datapoints, tuple):  # check for backwards compatibility
                TacotronDataset(path_to_transcript_dict=path_to_transcript_dict,
                                cache_dir=cache_dir,
                                lang=lang,
                                loading_processes=loading_processes,
                                min_len_in_seconds=min_len_in_seconds,
                                max_len_in_seconds=max_len_in_seconds,
                                cut_silences=cut_silence,
                                rebuild_cache=True)
                datapoints = torch.load(os.path.join(cache_dir, "taco_train_cache.pt"), map_location='cpu')
            dataset = datapoints[0]
            norm_waves = datapoints[1]

            # build cache
            print("... building dataset cache ...")
            self.datapoints = list()
            self.pop_ids = list()

            try:
                acoustic_model=Tacotron2()
                acoustic_model.load_state_dict(torch.load(acoustic_checkpoint_path, map_location='cpu')["model"])
            except RuntimeError:
                acoustic_model = Tacotron2(elayers=0, econv_layers=0, adim=256, embed_dim=256, prenet_layers=0, postnet_layers=0)
                acoustic_model.load_state_dict(torch.load(acoustic_checkpoint_path, map_location='cpu')["model"])
            
            
            # ==========================================
            # actual creation of datapoints starts here
            # ==========================================

            vis_dir = os.path.join(cache_dir, "duration_vis")
            os.makedirs(vis_dir, exist_ok=True)
            acoustic_model = acoustic_model.to(device)
            dc = DurationCalculator(reduction_factor=reduction_factor)
            dio = Dio(reduction_factor=reduction_factor, fs=16000)
            energy_calc = EnergyCalculator(reduction_factor=reduction_factor, fs=16000)

            for index in tqdm(range(len(dataset))):

                norm_wave = norm_waves[index]
                norm_wave_length = torch.LongTensor([len(norm_wave)])

                text = dataset[index][0]
                melspec = dataset[index][2]
                melspec_length = dataset[index][3]

                attention_map = acoustic_model.inference(text_tensor=text.to(device),
                                                        speech_tensor=melspec.to(device),
                                                        use_teacher_forcing=True)[2]
                cached_duration = dc(attention_map, vis=None).cpu()

                if np.count_nonzero(cached_duration.numpy() == 0) > 4:
                    # here we figure out whether the attention map makes any sense or whether it failed.
                    self.pop_ids.append(index)
                    continue
                # if it didn't fail, we can use viterbi to refine the path and then calculate the durations again.
                # not the most efficient method, but it is the safest I can think of and I like safety over speed here.

                attention_map_viterbi_path = torch.from_numpy(mas_width1(attention_map.detach().cpu().numpy()))

                cached_duration = dc(attention_map_viterbi_path, vis=os.path.join(vis_dir, f"{index}.png")).cpu()

                cached_energy = energy_calc(input_waves=norm_wave.unsqueeze(0),
                                            input_waves_lengths=norm_wave_length,
                                            feats_lengths=melspec_length,
                                            durations=cached_duration.unsqueeze(0),
                                            durations_lengths=torch.LongTensor([len(cached_duration)]))[0].squeeze(0).cpu().numpy()
                cached_pitch = dio(input_waves=norm_wave.unsqueeze(0),
                                input_waves_lengths=norm_wave_length,
                                feats_lengths=melspec_length,
                                durations=cached_duration.unsqueeze(0),
                                durations_lengths=torch.LongTensor([len(cached_duration)]))[0].squeeze(0).cpu().numpy()
                self.datapoints.append([dataset[index][0],
                                                    dataset[index][1],
                                                    dataset[index][2],
                                                    dataset[index][3],
                                                    cached_duration.cpu().numpy(),
                                                    cached_energy,
                                                    cached_pitch])

            # =============================
            # done with datapoint creation
            # =============================

            print(f"Removing the following IDs to get a cleaner Tacotron Dataset: {self.pop_ids}")
            while len(self.pop_ids) > 0:
                pop_id = self.pop_ids.pop()
                dataset.pop(pop_id)
                norm_waves.pop(pop_id)
            os.rename(os.path.join(cache_dir, "taco_train_cache.pt"), os.path.join(cache_dir, "taco_train_cache_unclean.pt"))
            torch.save((dataset, norm_waves), os.path.join(cache_dir, "taco_train_cache.pt"))

            tensored_datapoints = list()
            # we had to turn all of the tensors to numpy arrays to avoid shared memory
            # issues. Now that the multi-processing is over, we can convert them back
            # to tensors to save on conversions in the future.
            print("Converting into convenient format...")
            for datapoint in tqdm(self.datapoints):
                tensored_datapoints.append([datapoint[0],
                                            datapoint[1],
                                            datapoint[2],
                                            datapoint[3],
                                            torch.LongTensor(datapoint[4]),  # durations
                                            torch.Tensor(datapoint[5]),  # energy
                                            torch.Tensor(datapoint[6])])  # pitch
            self.datapoints = tensored_datapoints
            # save to cache
            if len(self.datapoints) > 0:
                torch.save(self.datapoints, os.path.join(cache_dir, "fast_train_cache.pt"))
            else:
                import sys
                print("No datapoints were prepared! Exiting...")
                sys.exit()
        else:
            # just load the datapoints from cache
            self.datapoints = torch.load(os.path.join(cache_dir, "fast_train_cache.pt"), map_location='cpu')
        print("Prepared {} datapoints.".format(len(self.datapoints)))



    def __getitem__(self, index):
        return self.datapoints[index][0], \
               self.datapoints[index][1], \
               self.datapoints[index][2], \
               self.datapoints[index][3], \
               self.datapoints[index][4], \
               self.datapoints[index][5], \
               self.datapoints[index][6]

    def __len__(self):
        return len(self.datapoints)
