import logging
import sacrebleu
import typing as tp
from dataclasses import dataclass
from omegaconf.omegaconf import MISSING

from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements, StopesModule


@dataclass
class ComputeBleuScoreJob:
    reference: tp.List[str] = MISSING
    prediction_transcripts: tp.List[str] = MISSING


@dataclass
class ComputeBleuScoresConfig:
    compute_bleu_score_jobs: tp.List[ComputeBleuScoreJob] = MISSING


class ComputeBleuScores(StopesModule):
    def __init__(self, config: ComputeBleuScoresConfig):
        super().__init__(config=config, config_class=ComputeBleuScoresConfig)

    def array(self):
        return self.config.compute_bleu_score_jobs

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=1,
            timeout_min=24 * 60,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> str:
        """Computes BLEU score for each BleuScoreJob"""
        assert iteration_value is not None, "iteration value is null"
        self.logger = logging.getLogger("stopes.asr_bleu.compute_bleu_scores")

        bleu_score = sacrebleu.corpus_bleu(
            iteration_value.prediction_transcripts,
            [iteration_value.reference]
        )

        return bleu_score


async def compute_bleu_scores(
    retrieved_data: tp.List[tp.Tuple[tp.Dict[str, tp.List], str, str]],
    transcribed_audio: tp.List[tp.List[str]],
    launcher: Launcher,
) -> tp.List:
    """
    Compute BLEU scores for the transcribed audio files
    Returns a list of computed BLEU scores
    """
    compute_bleu_score_jobs = [
        ComputeBleuScoreJob(
            reference=retrieved_data[i][0]["reference"],
            prediction_transcripts=prediction_transcripts,
        ) for i, prediction_transcripts in enumerate(transcribed_audio)
    ]

    ComputeBleuScoresModule = ComputeBleuScores(
        ComputeBleuScoresConfig(
            compute_bleu_score_jobs=compute_bleu_score_jobs,
        )
    )
    bleu_scores = await launcher.schedule(ComputeBleuScoresModule)
    return bleu_scores
