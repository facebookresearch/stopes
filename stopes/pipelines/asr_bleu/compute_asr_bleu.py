import asyncio
import logging

from stopes.pipelines.asr_bleu.configs import AsrBleuConfig
from stopes.pipelines.asr_bleu.utils import retrieve_asr_config
from stopes.pipelines.asr_bleu.transcribe_audio import transcribe_audio
from stopes.pipelines.asr_bleu.retrieve_data import retrieve_data

import hydra
import sacrebleu

class AsrBleu:
    def __init__(self, config: AsrBleuConfig):
        self.config = config
        self.launcher = hydra.utils.instantiate(self.config.launcher)
        self.logger = logging.getLogger("asr_bleu")

    async def run(self):
        # 1. Retrieve ASR configuration 
        logging.info("Setting up ASR model...")
        asr_config = retrieve_asr_config(self.config.corpora.lang, self.config.asr_version, json_path="/home/calderj/Documents/Coding/MLH/stopes/stopes/pipelines/asr_bleu/conf/asr_model/asr_model_cfgs.json")

        # 2. Compose evaluation data.
        logging.info("Composing evaluation data...")
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
        logging.info("Transcribing audio predictions...")
        transcribed_audio = await transcribe_audio(
            eval_manifests,
            self.launcher,
            asr_config,
        )

        # 4. Compute BLEU score
        logging.info("Computing BLEU scores...")
        bleu_scores = []
        for i, prediction_transcripts in enumerate(transcribed_audio):
            references = eval_manifests[i]["reference"]
            bleu_score = sacrebleu.corpus_bleu(prediction_transcripts, [references])
            bleu_scores.append(bleu_score)
            print(bleu_score)

        return transcribed_audio, bleu_scores

def merge_tailo_init_final(text):
    """
    Hokkien ASR hypothesis post-processing.
    """
    sps = text.strip().split()
    results = []
    last_syllable = ""
    for sp in sps:
        if sp == "NULLINIT" or sp == "nullinit":
            continue
        last_syllable += sp
        if sp[-1].isnumeric():
            results.append(last_syllable)
            last_syllable = ""
    if last_syllable != "":
        results.append(last_syllable)
    return " ".join(results)

def remove_tone(text):
    """
    Used for tone-less evaluation of Hokkien
    """
    return " ".join([t[:-1] for t in text.split()])

@hydra.main(config_path="conf", config_name="asr_bleu")
def main(config: AsrBleuConfig) -> None:
    pipeline = AsrBleu(config)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()        
