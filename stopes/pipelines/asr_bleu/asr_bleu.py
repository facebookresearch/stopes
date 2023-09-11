import asyncio
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from stopes.core import utils
from stopes.pipelines.asr_bleu.compute_asr_bleu import compute_asr_bleu
from stopes.pipelines.asr_bleu.configs import AsrBleuConfig

logger = logging.getLogger("asr_bleu")


class AsrBleu:
    def __init__(self, config: AsrBleuConfig):
        self.config = config
        self.ensure_all_dirs()
        self.launcher = hydra.utils.instantiate(self.config.launcher)
        self.config.launcher.cache.caching_dir = Path(self.output_dir) / "cache"
        OmegaConf.save(
            config=config,
            f=str(self.output_dir / "asr_bleu.yaml"),
        )

        OmegaConf.set_readonly(self.config, True)

    async def run(self):
        logger.info("Computing ASRBleu on selected datasets...")
        await compute_asr_bleu(
            self.config.output_dir,
            self.config.split,
            self.config.model_name,
            self.config.eval_first_pass,
            self.config.dataset_name,
            self.config.datasets,
            self.launcher,
        )

    def ensure_all_dirs(self) -> None:
        self.output_dir = Path(self.config.output_dir).resolve()
        utils.ensure_dir(self.output_dir)


@hydra.main(config_path="conf", config_name="asr_bleu")
def main(config: AsrBleuConfig) -> None:
    pipeline = AsrBleu(config)
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
