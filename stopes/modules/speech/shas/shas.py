# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This source code is adapted with minimal changes
# from https://github.com/mt-upc/SHAS


from multiprocessing import cpu_count
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from stopes.modules.speech.vad_segment_audio import FromWavAudioDataset

from .data import FixedSegmentationDatasetNoTarget, Segment, segm_collate_fn

HIDDEN_SIZE = 1024  # output dimensionality of wav2vec 2.0 models with 300m params


def infer(
    wav2vec_model,
    sfc_model,
    dataloader,
    main_device,
) -> Tuple[np.array, np.array]:
    """Does inference with the Segmentation Frame Classifier for a single wav file

    Args:
        wav2vec_model: an instance of a wav2vec 2.0 model
        sfc_model: an instance of a segmentation frame classifier
        dataloader: a dataloader with the FixedSegmentationDataset of a wav
        main_device: the main torch.device

    Returns:
        Tuple[np.array, np.array]: the segmentation frame probabilities for the wav
            and (optionally) their ground truth values
    """

    duration_outframes = dataloader.dataset.duration_outframes

    talk_probs = np.empty(duration_outframes)
    talk_probs[:] = np.nan
    talk_targets = np.zeros(duration_outframes)

    for audio, targets, in_mask, out_mask, included, starts, ends in iter(dataloader):

        audio = audio.to(main_device)
        in_mask = in_mask.to(main_device)
        out_mask = out_mask.to(main_device)

        with torch.no_grad():
            wav2vec_hidden = wav2vec_model(
                audio, attention_mask=in_mask
            ).last_hidden_state

            # some times the output of wav2vec is 1 frame larger/smaller
            # correct for these cases
            size1 = wav2vec_hidden.shape[1]
            size2 = out_mask.shape[1]
            if size1 != size2:
                if size1 < size2:
                    out_mask = out_mask[:, :-1]
                    ends = [e - 1 for e in ends]
                else:
                    wav2vec_hidden = wav2vec_hidden[:, :-1, :]

            logits = sfc_model(wav2vec_hidden, out_mask)
            probs = torch.sigmoid(logits)
            probs[~out_mask] = 0

        probs = probs.detach().cpu().numpy()

        # fill-in the probabilities and targets for the talk
        for i in range(len(probs)):
            start, end = starts[i], ends[i]
            if included[i] and end > start:
                duration = end - start
                talk_probs[start:end] = probs[i, :duration]
                if targets is not None:
                    talk_targets[start:end] = targets[i, :duration].numpy()
            elif not included[i]:
                talk_probs[start:end] = 0

    # account for the rare incident that a frame didnt have a prediction
    # fill-in those frames with the average of the surrounding frames
    nan_idx = np.where(np.isnan(talk_probs))[0]
    for j in nan_idx:
        talk_probs[j] = np.nanmean(
            talk_probs[max(0, j - 2) : min(duration_outframes, j + 3)]
        )

    return talk_probs, talk_targets


def trim(sgm: Segment, threshold: float) -> Segment:
    """reduces the segment to between the first and last points that are above the threshold

    Args:
        sgm (Segment): a segment
        threshold (float): probability threshold

    Returns:
        Segment: new reduced segment
    """
    included_indices = np.where(sgm.probs >= threshold)[0]

    # return empty segment
    if not len(included_indices):
        return Segment(sgm.start, sgm.start, np.empty([0]))

    i = included_indices[0]
    j = included_indices[-1] + 1

    sgm = Segment(sgm.start + i, sgm.start + j, sgm.probs[i:j])

    return sgm


def split_and_trim(
    sgm: Segment, split_idx: int, threshold: float
) -> Tuple[Segment, Segment]:
    """splits the input segment at the split_idx and then trims and returns the two resulting segments

    Args:
        sgm (Segment): input segment
        split_idx (int): index to split the input segment
        threshold (float): probability threshold

    Returns:
        Tuple[Segment, Segment]: the two resulting segments
    """

    probs_a = sgm.probs[:split_idx]
    sgm_a = Segment(sgm.start, sgm.start + len(probs_a), probs_a)

    probs_b = sgm.probs[split_idx + 1 :]
    sgm_b = Segment(sgm_a.end + 1, sgm.end, probs_b)

    sgm_a = trim(sgm_a, threshold)
    sgm_b = trim(sgm_b, threshold)

    return sgm_a, sgm_b


def pdac(
    probs: np.array,
    max_segment_length: float,
    min_segment_length: float,
    threshold: float,
) -> List[Segment]:
    """applies the probabilistic Divide-and-Conquer algorithm to split an audio
    into segments satisfying the max-segment-length and min-segment-length conditions

    Args:
        probs (np.array): the binary frame-level probabilities
            output by the segmentation-frame-classifier
        max_segment_length (float): the maximum length of a segment
        min_segment_length (float): the minimum length of a segment
        threshold (float): probability threshold

    Returns:
        list[Segment]: resulting segmentation
    """

    segments = []
    sgm = Segment(0, len(probs), probs)
    sgm = trim(sgm, threshold)

    def recusrive_split(sgm):
        if sgm.duration < max_segment_length:
            segments.append(sgm)
        else:
            j = 0
            sorted_indices = np.argsort(sgm.probs)
            while j < len(sorted_indices):
                split_idx = sorted_indices[j]
                split_prob = sgm.probs[split_idx]
                sgm_a, sgm_b = split_and_trim(sgm, split_idx, threshold)
                if (
                    sgm_a.duration > min_segment_length
                    and sgm_b.duration > min_segment_length
                ):
                    recusrive_split(sgm_a)
                    recusrive_split(sgm_b)
                    break
                j += 1
            else:
                if sgm_a.duration > min_segment_length:
                    recusrive_split(sgm_a)
                if sgm_b.duration > min_segment_length:
                    recusrive_split(sgm_b)

    recusrive_split(sgm)
    return segments


class SHAS:
    def __init__(
        self,
        path_to_checkpoint,
        inference_batch_size: int = 12,
        inference_times: int = 2,
        max_segment_length: int = 18,
        min_segment_length: int = 2,
        dac_threshold=0.5,
        use_gpu=True,
    ):
        """_summary_
        Args:
            path_to_checkpoint (_type_): absolute path to the shas pretrained
                                        multilingual model checkpoint found on
                                        https://github.com/mt-upc/SHAS
            inference_batch_size (int, optional): batch size (in examples) of inference with
                                                  the audio-frame-classifier. Defaults to 12.
            inference_times (int, optional): how many times to apply inference on different
                                             fixed-length segmentations. Defaults to 2.
            max_segment_length (int, optional): the segmentation algorithm splits until all
                                                segments are below this value. Defaults to 18.
            min_segment_length (int, optional): a split by the algorithm is carried
                                                out only if the resulting two segments
                                                larger than this value. Defaults to 2.
            dac_threshold (float, optional): after each split by the algorithm, the resulting
                                            segments are trimmed to the first and last points
                                            that corresponds to a probability above this value.
                                            Defaults to 0.5.
            use_gpu (bool, optional): running the SHAS model on GPU
                                            (cuda:0 is automatically selected).
                                                Defaults to True.
        """

        self.inference_batch_size = inference_batch_size
        self.inference_times = inference_times
        self.max_segment_length = max_segment_length
        self.min_segment_length = min_segment_length
        self.dac_threshold = dac_threshold
        self.use_gpu = use_gpu

        if use_gpu is True:
            if torch.cuda.device_count() > 0:
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
                logger.warning(
                    "You requested use_gpu=True, but there is no GPU available. Falling back to CPU."
                )
        else:
            self.device = torch.device("cpu")

        self.wav2vec_model, self.sfc_model = self.load_shas_models(path_to_checkpoint)

    def load_shas_models(self, path_to_checkpoint):
        from .models import SegmentationFrameClassifer, prepare_wav2vec

        # shas model consists of a (frozen) wav2vec encoder
        # and a Trainable Segmentation Frame classifier (sfc) on top.
        # more info (Tsiamas et. al 2022)
        # https://www.isca-speech.org/archive/pdfs/interspeech_2022/tsiamas22_interspeech.pdf

        checkpoint = torch.load(path_to_checkpoint, map_location=self.device)

        # init wav2vec 2.0
        wav2vec_model = prepare_wav2vec(
            checkpoint["args"].model_name,
            checkpoint["args"].wav2vec_keep_layers,
            self.device,
        )
        # initalize segmentation frame classifier (sfc)
        sfc_model = SegmentationFrameClassifer(
            d_model=HIDDEN_SIZE,
            n_transformer_layers=checkpoint["args"].classifier_n_transformer_layers,
        ).to(self.device)
        sfc_model.load_state_dict(checkpoint["state_dict"])
        sfc_model.eval()

        return wav2vec_model, sfc_model

    def segment(self, audio_path):
        # initialize a dataset for the fixed segmentation
        dataset = FixedSegmentationDatasetNoTarget(
            audio_path, self.max_segment_length, self.inference_times
        )
        sgm_frame_probs = None

        for inference_iteration in range(self.inference_times):

            # create a dataloader for this fixed-length segmentation of the wav file
            dataset.fixed_length_segmentation(inference_iteration)
            dataloader = DataLoader(
                dataset,
                batch_size=self.inference_batch_size,
                num_workers=min(cpu_count() // 2, 4),
                shuffle=False,
                drop_last=False,
                collate_fn=segm_collate_fn,
            )

            # get frame segmentation frame probabilities in the output space
            probs, _ = infer(
                self.wav2vec_model,
                self.sfc_model,
                dataloader,
                self.device,
            )
            if sgm_frame_probs is None:
                sgm_frame_probs = probs.copy()
            else:
                sgm_frame_probs += probs

        sgm_frame_probs /= self.inference_times

        # Run Probabilistic DAC algorithm
        # segments: List[Segment]
        segments = pdac(
            sgm_frame_probs,
            self.max_segment_length,
            self.min_segment_length,
            self.dac_threshold,
        )
        # shas predictions are usually in a downsamples space called "output_space"
        # we need to revert these to the input frame rate (usually 16khz)
        # and load it in an interface is used in the rest of the stopes pipeline
        # segments # wav[start:end]
        # indices # (start_frame,end_frame)
        # sizes # (end_frame -start_frame)
        # time stamps (start_time, end_time) for each
        # converting data.Segment into this interface
        segments_wavs = []
        indices = []
        sizes = []

        for segment in segments:
            # changing segments from SHAS output space (typically 50hz) to input space (16khz)
            start_frame = dataset.secs_to_inframes(segment.offset)
            end_frame = dataset.secs_to_inframes(segment.offset_plus_duration)
            # loading the start fram and end frame in the 16khz space to the rest of the stopes pipeline
            segments_wavs.append(dataset.wavform[start_frame:end_frame])
            indices.append((start_frame, end_frame))
            sizes.append(end_frame - start_frame)
        return segments_wavs, indices, sizes

    # STOPES INTERFACE
    def build_segments_dataset(self, audio_path, max_tokens=1280000):
        segments, segment_indices, sizes = self.segment(audio_path)
        dataset = FromWavAudioDataset(segments, audio_path, segment_indices, sizes)
        batch_indices = dataset.ordered_indices()
        batch_sampler = dataset.batch_by_size(
            batch_indices,
            max_tokens=max_tokens,
            max_sentences=None,
            required_batch_size_multiple=1,
        )
        return batch_sampler, dataset
