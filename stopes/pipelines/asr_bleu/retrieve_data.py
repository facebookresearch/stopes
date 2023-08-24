import logging
import os
import typing as tp
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import pandas as pd
from omegaconf.omegaconf import MISSING
from stopes.core import utils
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.pipelines.asr_bleu.configs import CorporaConfig


@dataclass
class RetrieveDataJob:
    audio_path: str = MISSING
    reference_path: str = MISSING
    audio_format: str = MISSING
    reference_format: str = MISSING
    reference_tsv_column: tp.Optional[str] = None
    config_key: str = MISSING
    lang: str = MISSING
    asr_version: str = MISSING


@dataclass
class RetrieveDataConfig:
    retrieve_data_jobs: tp.List[RetrieveDataJob] = MISSING


class RetrieveData(StopesModule):
    def __init__(self, config: RetrieveDataConfig):
        super().__init__(config=config, config_class=RetrieveDataConfig)

    def array(self):
        return self.config.retrieve_data_jobs

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=1,
            timeout_min=24 * 60,
        )

    def _extract_audio_for_eval(
        self,
        audio_dirpath: str,
        audio_format: str
    ) -> tp.List[str]:
        """Extract audio file paths for subsequent transcription"""
        if audio_format == "n_pred.wav":
            """
            The assumption here is that 0_pred.wav corresponds to the reference
            at line position 0 from the reference manifest
            """
            audio_list = []
            audio_fp_list = glob((Path(audio_dirpath)
                                  / "*_pred.wav").as_posix())
            audio_fp_list = sorted(
                audio_fp_list,
                key=lambda x: int(os.path.basename(x).split("_")[0])
            )
            for i in range(len(audio_fp_list)):
                try:
                    audio_fp = (Path(audio_dirpath)
                                / f"{i}_pred.wav").as_posix()
                    assert (
                        audio_fp in audio_fp_list
                    ), f"{Path(audio_fp).name} \
                       does not exist in {audio_dirpath}"
                except AssertionError:
                    # check the audio with random speaker
                    audio_fp = Path(audio_dirpath) / f"{i}_spk*_pred.wav"
                    audio_fp = glob(
                        audio_fp.as_posix()
                    )  # resolve audio filepath with random speaker
                    assert len(audio_fp) == 1
                    audio_fp = audio_fp[0]

                audio_list.append(audio_fp)
        else:
            raise NotImplementedError

        return audio_list

    def _extract_text_for_eval(
        self,
        references_filepath: str,
        reference_format: str,
        reference_tsv_column: str = None
    ) -> tp.List[str]:
        """Extract sentences for reference"""
        if reference_format == "txt":
            reference_sentence_file = utils.open(references_filepath, "r")
            reference_sentences = [
                line.strip() for line in reference_sentence_file.readlines()
            ]
        elif reference_format == "tsv":
            tsv_df = pd.read_csv(references_filepath, sep="\t", quoting=3)
            reference_sentences = tsv_df[reference_tsv_column].to_list()
            reference_sentences = [
                line.strip() for line in reference_sentences
            ]
        else:
            raise NotImplementedError

        return reference_sentences

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Tuple[tp.Dict[str, tp.List], str, str]:
        """Retrieves data for each RetrieveDataJob"""
        assert iteration_value is not None, "iteration value is null"
        self.logger = logging.getLogger("stopes.asr_bleu.prepare_data")

        self.logger.info(
            f"Retrieving audio data from {iteration_value.audio_path}"
        )
        audio_list = self._extract_audio_for_eval(
            iteration_value.audio_path,
            iteration_value.audio_format
        )

        self.logger.info(
            f"Retrieving text data from {iteration_value.reference_path}"
        )
        reference_sentences = self._extract_text_for_eval(
            iteration_value.reference_path,
            iteration_value.reference_format,
            iteration_value.reference_tsv_column
        )

        eval_manifest = {
            "prediction": audio_list,
            "reference": reference_sentences,
        }

        return (
            eval_manifest,
            iteration_value.config_key,
            iteration_value.lang,
            iteration_value.asr_version
        )


async def retrieve_data(
    corpora_conf: CorporaConfig,
    launcher: Launcher,
) -> tp.List[tp.Tuple[tp.Dict[str, tp.List], str, str]]:
    """
    Retrieve data for transcription
    Returns a list of 4 tuples: (eval_manifest, config_key, lang, asr_version)
    """
    datasets = corpora_conf.datasets
    retrieve_data_jobs = [
        RetrieveDataJob(
            audio_path=datasets[corpus].audio_dirpath,
            reference_path=datasets[corpus].reference_path,
            audio_format=datasets[corpus].audio_format,
            reference_format=datasets[corpus].reference_format,
            reference_tsv_column=datasets[corpus].reference_tsv_column,
            config_key=datasets[corpus].config_key,
            lang=datasets[corpus].lang,
            asr_version=datasets[corpus].asr_version,
        ) for corpus in datasets
    ]
    retrieve_data_module = RetrieveData(
        RetrieveDataConfig(
            retrieve_data_jobs=retrieve_data_jobs,
        )
    )
    retrieved_datasets = await launcher.schedule(retrieve_data_module)
    return retrieved_datasets
