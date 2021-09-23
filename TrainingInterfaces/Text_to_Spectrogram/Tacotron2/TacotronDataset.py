import os
import random
import time

import soundfile as sf
import torch
from speechbrain.pretrained import EncoderClassifier
from torch.multiprocessing import Manager
from torch.multiprocessing import Process
from torch.utils.data import Dataset
from tqdm import tqdm

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.AudioPreprocessor import AudioPreprocessor


class TacotronDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 cache_dir,
                 lang,
                 speaker_embedding=False,
                 loading_processes=10,
                 min_len_in_seconds=1,
                 max_len_in_seconds=20,
                 cut_silences=False,
                 rebuild_cache=False,
                 return_language_id=False,
                 device="cpu"):
        self.return_language_id = return_language_id
        self.language_id = ArticulatoryCombinedTextFrontend(language=lang).language_id
        self.speaker_embedding = speaker_embedding
        if not os.path.exists(os.path.join(cache_dir, "taco_train_cache.pt")) or rebuild_cache:
            resource_manager = Manager()
            self.path_to_transcript_dict = resource_manager.dict(path_to_transcript_dict)
            key_list = list(self.path_to_transcript_dict.keys())
            random.shuffle(key_list)
            # build cache
            print("... building dataset cache ...")
            self.datapoints = resource_manager.list()
            # make processes
            key_splits = list()
            process_list = list()
            for i in range(loading_processes):
                key_splits.append(key_list[i * len(key_list) // loading_processes:(i + 1) * len(key_list) // loading_processes])
            for key_split in key_splits:
                process_list.append(
                    Process(target=self.cache_builder_process,
                            args=(key_split,
                                  speaker_embedding,
                                  lang,
                                  min_len_in_seconds,
                                  max_len_in_seconds,
                                  cut_silences),
                            daemon=True))
                process_list[-1].start()
                time.sleep(5)
            for process in process_list:
                process.join()
            self.datapoints = list(self.datapoints)
            tensored_datapoints = list()
            # we had to turn all of the tensors to numpy arrays to avoid shared memory
            # issues. Now that the multi-processing is over, we can convert them back
            # to tensors to save on conversions in the future.
            print("Converting into convenient format...")
            norm_waves = list()
            if self.speaker_embedding:
                if speaker_embedding:
                    speaker_embedding_function_ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                                      run_opts={"device": str(device)},
                                                                                      savedir="Models/speechbrain_speaker_embedding_ecapa")
                    speaker_embedding_function_xvector = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                                                        run_opts={"device": str(device)},
                                                                                        savedir="Models/speechbrain_speaker_embedding_xvector")
                    wav2mel = torch.jit.load("Models/SpeakerEmbedding/wav2mel.pt")
                    dvector = torch.jit.load("Models/SpeakerEmbedding/dvector-step250000.pt").to(device).eval()
                    # everything assumes 16kHz audio as input here
                    for datapoint in tqdm(self.datapoints):
                        ecapa_spemb = speaker_embedding_function_ecapa.encode_batch(torch.Tensor(datapoint[4]).to(device)).flatten().detach().cpu()
                        xvector_spemb = speaker_embedding_function_xvector.encode_batch(torch.Tensor(datapoint[4]).to(device)).flatten().detach().cpu()
                        dvector_spemb = dvector.embed_utterance(wav2mel(torch.Tensor(datapoint[4]), 16000).to(device)).flatten().detach().cpu()
                        combined_spemb = torch.cat([ecapa_spemb, xvector_spemb, dvector_spemb], dim=0)

                        tensored_datapoints.append([torch.Tensor(datapoint[0]),
                                                    torch.LongTensor(datapoint[1]),
                                                    torch.Tensor(datapoint[2]),
                                                    torch.LongTensor(datapoint[3]),
                                                    combined_spemb])
                        norm_waves.append(torch.Tensor(datapoint[-1]))
                    del speaker_embedding_function_ecapa
                    del speaker_embedding_function_xvector
                    del wav2mel
                    del dvector
                # loading the speaker embedding function messes up something in torchaudios resample layer.
                # So you possibly need to restart the program after every dataset creation if you are using
                # multiple datasets together.
            else:
                for datapoint in tqdm(self.datapoints):
                    tensored_datapoints.append([torch.Tensor(datapoint[0]),
                                                torch.LongTensor(datapoint[1]),
                                                torch.Tensor(datapoint[2]),
                                                torch.LongTensor(datapoint[3])])
                    norm_waves.append(torch.Tensor(datapoint[-1]))

            self.datapoints = tensored_datapoints
            # save to cache
            torch.save((self.datapoints, norm_waves), os.path.join(cache_dir, "taco_train_cache.pt"))
        else:
            # just load the datapoints from cache
            self.datapoints = torch.load(os.path.join(cache_dir, "taco_train_cache.pt"), map_location='cpu')

            if isinstance(self.datapoints, tuple):  # check for backwards compatibility
                self.datapoints = self.datapoints[0]
        print("Prepared {} datapoints.".format(len(self.datapoints)))

    def cache_builder_process(self, path_list, speaker_embedding, lang, min_len, max_len, cut_silences):
        process_internal_dataset_chunk = list()
        tf = ArticulatoryCombinedTextFrontend(language=lang)
        _, sr = sf.read(path_list[0])

        ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=cut_silences)
        for path in tqdm(path_list):
            transcript = self.path_to_transcript_dict[path]
            wave, sr = sf.read(path)
            if sr != ap.sr:
                print(f"Inconsistent sampling rate in the Data! Excluding {path}")
                continue
            try:
                norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
            except ValueError:
                continue
            dur_in_seconds = len(norm_wave) / 16000
            if not (min_len <= dur_in_seconds <= max_len):
                print(f"Excluding {path} because of its duration of {round(dur_in_seconds, 2)} seconds.")
            else:
                cached_text = tf.string_to_tensor(transcript).squeeze(0).cpu().numpy()
                cached_text_len = torch.LongTensor([len(cached_text)]).numpy()
                cached_speech = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False).transpose(0, 1).cpu().numpy()
                cached_speech_len = torch.LongTensor([len(cached_speech)]).numpy()
                if speaker_embedding:
                    process_internal_dataset_chunk.append([cached_text,
                                                           cached_text_len,
                                                           cached_speech,
                                                           cached_speech_len,
                                                           norm_wave.numpy(),
                                                           norm_wave.cpu().detach().numpy()])
                else:
                    process_internal_dataset_chunk.append([cached_text,
                                                           cached_text_len,
                                                           cached_speech,
                                                           cached_speech_len,
                                                           norm_wave.cpu().detach().numpy()])
        self.datapoints += process_internal_dataset_chunk

    def __getitem__(self, index):
        if self.return_language_id:
            if not self.speaker_embedding:
                return self.datapoints[index][0], \
                       self.datapoints[index][1], \
                       self.datapoints[index][2], \
                       self.datapoints[index][3], \
                       self.language_id
            else:
                return self.datapoints[index][0], \
                       self.datapoints[index][1], \
                       self.datapoints[index][2], \
                       self.datapoints[index][3], \
                       self.datapoints[index][4], \
                       self.language_id
        else:
            if not self.speaker_embedding:
                return self.datapoints[index][0], \
                       self.datapoints[index][1], \
                       self.datapoints[index][2], \
                       self.datapoints[index][3]
            else:
                return self.datapoints[index][0], \
                       self.datapoints[index][1], \
                       self.datapoints[index][2], \
                       self.datapoints[index][3], \
                       self.datapoints[index][4]

    def __len__(self):
        return len(self.datapoints)
