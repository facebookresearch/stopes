import asyncio
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from stopes.core import utils
from stopes.pipelines.asr_bleu.compute_bleu_scores import compute_bleu_scores
from stopes.pipelines.asr_bleu.configs import AsrBleuConfig
from stopes.pipelines.asr_bleu.retrieve_data import retrieve_data
from stopes.pipelines.asr_bleu.transcribe_audio import transcribe_audio

logger = logging.getLogger("asr_bleu")


class AsrBleu:
    def __init__(self, config: AsrBleuConfig):
        self.config = config
        self.ensure_all_dirs()
        self.launcher = hydra.utils.instantiate(self.config.launcher)
        self.config.launcher.cache.caching_dir = Path(self.output_dir) / \
            "cache"
        OmegaConf.save(
            config=config,
            f=str(self.output_dir / "asr_bleu.yaml"),
        )

        OmegaConf.set_readonly(self.config, True)

    async def run(self):
        # 1. Compose evaluation data.
        logger.info("Composing evaluation data...")
        retrieved_data = await retrieve_data(
            self.config.corpora,
            self.launcher,
        )

        # 2. Transcribe audio predictions and compute BLEU score.
        logger.info("Transcribing audio predictions...")
        transcribed_audio = await transcribe_audio(
            retrieved_data,
            self.launcher,
        )

        # 3. Compute BLEU score
        logger.info("Computing BLEU scores...")
        bleu_scores = await compute_bleu_scores(
            retrieved_data,
            transcribed_audio,
            self.launcher,
        )

        # 4. Save the BLEU score
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
