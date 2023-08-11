import asyncio
import logging

from stopes.pipelines.asr_bleu.configs import AsrBleuConfig
from stopes.pipelines.asr_bleu.utils import retrieve_asr_config
from stopes.pipelines.asr_bleu.transcribe_audio import transcribe_audio
from stopes.pipelines.asr_bleu.retrieve_data import retrieve_data
from stopes.core import utils

from pathlib import Path
import hydra
from omegaconf import OmegaConf
import sacrebleu

logger = logging.getLogger("asr_bleu")


class AsrBleu:
    def __init__(self, config: AsrBleuConfig):
        self.config = config
        self.ensure_all_dirs()
        self.config.launcher.cache.caching_dir = Path(self.output_dir) \
            / "cache"
        self.launcher = hydra.utils.instantiate(self.config.launcher)
        OmegaConf.save(
            config=config,
            f=str(self.output_dir / "asr_bleu.yaml"),
        )

        OmegaConf.set_readonly(self.config, True)

    async def run(self):
        # 1. Retrieve ASR configuration
        logger.info("Setting up ASR model...")
        asr_config = retrieve_asr_config(
            self.config.corpora.lang,
            self.config.corpora.asr_version,
            json_path="../../../conf/asr_model/asr_model_cfgs.json"
        )

        # 2. Compose evaluation data.
        logger.info("Composing evaluation data...")
        eval_manifests = await retrieve_data([
                (self.config.corpora.audio_dirpath,
                 self.config.corpora.reference_path,
                 self.config.corpora.audio_format,
                 self.config.corpora.reference_format,
                 self.config.corpora.reference_tsv_column,)
            ],
            self.launcher,
        )

        # 3. Transcribe audio predictions and compute BLEU score.
        logger.info("Transcribing audio predictions...")
        transcribed_audio = await transcribe_audio(
            eval_manifests,
            self.launcher,
            asr_config,
        )

        # 4. Compute BLEU score
        logger.info("Computing BLEU scores...")
        bleu_scores = []
        for i, prediction_transcripts in enumerate(transcribed_audio):
            references = eval_manifests[i]["reference"]
            bleu_score = sacrebleu.corpus_bleu(
                prediction_transcripts,
                [references]
            )
            bleu_scores.append(bleu_score)
            logger.info(bleu_score)

        # 5. Save the BLEU score
        bleu_scores_file = self.output_dir / "bleu_scores"

        with open(bleu_scores_file, "w") as f:
            for score in bleu_scores:
                f.write(str(score) + "\n")

        return transcribed_audio, bleu_scores

    def ensure_all_dirs(self) -> None:
        self.output_dir = Path(self.config.output_dir).resolve()
        utils.ensure_dir(self.output_dir)


@hydra.main(config_path="conf", config_name="asr_bleu")
def main(config: AsrBleuConfig) -> None:
    pipeline = AsrBleu(config)
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
