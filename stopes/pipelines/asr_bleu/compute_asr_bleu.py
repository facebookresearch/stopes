import logging
import torch
import typing as tp
from dataclasses import dataclass

from m4t_scripts.evaluate.asr_bleu import ASRBleu
from omegaconf.omegaconf import MISSING

from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.pipelines.asr_bleu.configs import Dataset

@dataclass
class ComputeASRBleuJob:
    lang_dir: str = MISSING
    split: str = MISSING
    num_data_pairs: int = MISSING
    model_name: str = MISSING
    eval_first_pass: bool = MISSING
    dataset: str = MISSING
    audio_format: str = MISSING


@dataclass
class ComputeASRBleuConfig:
    compute_asrbleu_jobs: tp.List[ComputeASRBleuJob] = MISSING
    output_dir: str = MISSING


class ComputeASRBleu(StopesModule):
    def __init__(self, config: ComputeASRBleuConfig):
        super().__init__(config=config, config_class=ComputeASRBleuConfig)
        self.asrbleu = ASRBleu(config.output_dir)
        self.logger = logging.getLogger("stopes.asr_bleu")

    def array(self):
        return self.config.compute_asrbleu_jobs

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=1,
            timeout_min=24 * 60,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        """Runs compute_asr_bleu for each ComputeASRBleuJob"""
        assert iteration_value is not None, "iteration value is null"
        self.logger.info(f"Running compute_asr_bleu on {iteration_value.lang_dir}")
        self.asrbleu.compute_asr_bleu(
            iteration_value.lang_dir,
            iteration_value.split,
            iteration_value.num_data_pairs,
            iteration_value.model_name,
            iteration_value.eval_first_pass,
            iteration_value.dataset,
            iteration_value.audio_format,
        )


async def compute_asr_bleu(
    output_dir: str,
    split: str,
    model_name: str,
    eval_first_pass: bool,
    dataset_name: str,
    audio_format: str,
    datasets: tp.Dict[str, Dataset],
    launcher: Launcher,
) -> tp.List[tp.Tuple[tp.Dict[str, tp.List], str, str]]:
    """
    Compute ASRBleu on specified datasets
    """
    compute_asrbleu_jobs = [
        ComputeASRBleuJob(
            lang_dir=datasets[dataset].lang_dir,
            split=split,
            num_data_pairs=datasets[dataset].num_data_pairs,
            model_name=model_name,
            eval_first_pass=eval_first_pass,
            dataset=dataset_name,
            audio_format=audio_format,
        )
        for dataset in datasets
    ]
    compute_asrbleu_module = ComputeASRBleu(
        ComputeASRBleuConfig(
            compute_asrbleu_jobs=compute_asrbleu_jobs, output_dir=output_dir
        )
    )
    await launcher.schedule(compute_asrbleu_module)
