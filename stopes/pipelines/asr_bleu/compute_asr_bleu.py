import asyncio
import logging

from stopes.core import utils
from stopes.pipelines.asr_bleu.configs import AsrBleuConfig
from stopes.pipelines.asr_bleu.utils import retrieve_asr_config
from stopes.pipelines.asr_bleu.transcribe_audio import transcribe_audio
from stopes.pipelines.asr_bleu.retrieve_data import retrieve_data

import hydra
import sacrebleu

logger = logging.getLogger("stopes.asr_bleu")

class AsrBleu:
    def __init__(self, config: AsrBleuConfig):
        self.config = config
        self.launcher = hydra.utils.instantiate(self.config.launcher)
        utils.ensure_dir(self.config.output_dir)

    async def run(self):
        # 1. Retrieve ASR configuration 
        logger.info("Setting up ASR model...")
        asr_config = retrieve_asr_config(self.config.corpora.lang, self.config.asr_version, json_path="../../../conf/asr_model/asr_model_cfgs.json")

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
            bleu_score = sacrebleu.corpus_bleu(prediction_transcripts, [references])
            bleu_scores.append(bleu_score)
            logger.info(bleu_score)

        return transcribed_audio, bleu_scores

@hydra.main(config_path="conf", config_name="asr_bleu")
def main(config: AsrBleuConfig) -> None:
    pipeline = AsrBleu(config)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()        
