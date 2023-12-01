# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import functools
import io
import logging
import typing as tp
from pathlib import Path

import submitit
import torchaudio

from stopes.core import utils
from stopes.core.stopes_module import Requirements, StopesModule

from . import shas_segment_audio, vad_segment_audio

log = logging.getLogger("stopes.segment_dataset")


@dataclasses.dataclass
class SegmentDATASETConfig:
    dataset_dir: Path
    folders_per_manifest: int
    output_dir: Path
    segment: tp.Any
    job_limit: int = -1


@dataclasses.dataclass
class Progress:
    dataset_folders: tp.List[Path]
    """List of folders to process"""

    dataset_mp3s: tp.Optional[tp.List[Path]]
    """Mp3 in the current folder to process. None means we need to list the folder."""

    total_folders: int

    def __repr__(self) -> str:
        left, total = len(self.dataset_folders), self.total_folders
        if left:
            return f"Progress({total - left} / {total}). Current folder {self.dataset_folders[-1]}."
        else:
            return f"Progress({total - left} / {total})."


class SegmentDATASETModule(StopesModule):
    """
    Regroup several DATASET folders and segment them into one file.

    python -m stopes.modules \
        +speech_preproc=segment_dataset \
        speech_preproc.dataset_dir=??? \
        speech_preproc.segment.model=??? \
        speech_preproc.output_dir=~/dataset/vad/ \
        speech_preproc.job_limit=2 \
        launcher.cluster=debug

    python -m stopes.modules \
        +speech_preproc=segment_dataset \
        speech_preproc.dataset_dir=??? \
        speech_preproc.segment.model=??? \
        speech_preproc@speech_preproc.segment=shas \
        speech_preproc.output_dir=~/dataset/shas/ \
        laucher.cluster=debug
    """

    config: SegmentDATASETConfig

    def __init__(self, config: tp.Any):
        super().__init__(config, SegmentDATASETConfig)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._segment: tp.Any = None
        self._current_progress: tp.Optional[Progress] = None
        segment_cls = self.config.segment._target_
        if "vad" in segment_cls.lower():
            self.segment_cls = "vad"
        elif "shas" in segment_cls.lower():
            self.segment_cls = "shas"
        else:
            raise ValueError(f"Unknown segment_cls: {segment_cls}")

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=1,
            timeout_min=3 * 24 * 60,
        )

    def array(self) -> tp.List[Progress]:
        """
        DATASET has N (~20k) folders numbered from 0 to N.
        We submit one job per 20 folders.
        That way each job runs in ~1 day, and we don't create too many jobs for SLURM.
        """
        folders = [
            d
            for d in self.config.dataset_dir.iterdir()
            if d.is_dir() and d.name.isdecimal()
        ]
        folders.sort(key=lambda d: int(d.name))
        dataset_folders = utils.batch(folders, self.config.folders_per_manifest)

        jobs = [Progress(group[::-1], None, len(group)) for group in dataset_folders]
        if self.config.job_limit > 0:
            return jobs[: self.config.job_limit]

        return jobs

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: tp.Optional[int] = None,
    ) -> tp.Any:
        # allow to load .mp3
        torchaudio.set_audio_backend("sox_io")

        progress = iteration_value
        assert isinstance(progress, Progress)
        start_from_scratch = self._current_progress is None

        self._current_progress = progress
        self._segment = load_segmenter(self.config)

        n = self.config.folders_per_manifest
        i = int(iteration_index)  # type: ignore
        output_file = self.output_dir / f"dataset.{i * n}-{(i+1) * n - 1}.{i:05d}.gz"
        if start_from_scratch:
            log.info(
                f"Will segment DATASET to {output_file} with {self.segment_cls}. {progress}"
            )
        else:
            log.info(
                f"Will resume DATASET segmentation from {output_file} with {self.segment_cls}. {progress}"
            )

        mode = "w" if start_from_scratch else "a"
        corrupted = output_file.with_suffix(".corrupted.gz")
        n_mp3s, n_corrupted = 0, 0
        with utils.open(output_file, mode) as o, utils.open(corrupted, mode) as cor:
            while progress.dataset_folders:
                if progress.dataset_mp3s is None:
                    # Since we pop from the end of list, we reverse sort the list
                    # so that mp3 files appear in alphabetical order in the manifest.
                    progress.dataset_mp3s = sorted(
                        progress.dataset_folders[-1].glob("*.mp3"), reverse=True
                    )

                while progress.dataset_mp3s:
                    input_file = progress.dataset_mp3s[-1]
                    n_mp3s += 1
                    try:
                        # Try to make the writing as atomic as possible by calling print once per mp3
                        mp3_o = io.StringIO()
                        for start, end, batch_no in self._segment(input_file):
                            # TODO: why aren't we writing a regular manifest file ?
                            mp3_o.write(f"{input_file} {start} {end} {batch_no}\n")
                        print(mp3_o.getvalue(), end="", flush=True, file=o)
                    except Exception as e:
                        n_corrupted += 1
                        err_msg = f"Failed to process {input_file}: {e} ({n_corrupted}/{n_mp3s} corrupted mp3)"
                        log.error(err_msg)
                        print(err_msg, file=cor)
                    finally:
                        # We have flushed all segments for this mp3, we can move to the next one
                        done_file = progress.dataset_mp3s.pop()
                        assert done_file is input_file

                # We have handled all mp3s from this folder, move to next folder
                progress.dataset_mp3s = None
                folder = progress.dataset_folders.pop()
                log.info(
                    f"Done processing {folder}. {progress}. {n_corrupted}/{n_mp3s} corrupted mp3"
                )

        if n_corrupted == 0:
            # Remove empty file we created
            corrupted.unlink()

        return output_file

    def checkpoint(
        self, iteration_value: Progress, iteration_index: int, **kwargs: tp.Any
    ) -> submitit.helpers.DelayedSubmission:
        progress = self._current_progress

        self._segment = None

        # resubmit the module with updated progress
        return submitit.helpers.DelayedSubmission(
            self, progress, iteration_index, **kwargs
        )

    def name(self) -> str:
        return f"segment_dataset_{self.segment_cls}"


# TODO(@hadyelsahar): move this to a new Segmenter class
def load_segmenter(
    config: SegmentDATASETConfig,
) -> tp.Callable[[Path], tp.Iterator[tp.Tuple[int, int, int]]]:
    segment_cls = config.segment._target_
    if "vad" in segment_cls.lower():
        model = vad_segment_audio.load_model(config.segment)
        return functools.partial(vad_segment_audio.segment_file, model)

    if "shas" in segment_cls.lower():
        model = shas_segment_audio.load_model(config.segment)
        return functools.partial(shas_segment_audio.segment_file, model)

    raise ValueError(f"unknown segmenter {segment_cls}")
