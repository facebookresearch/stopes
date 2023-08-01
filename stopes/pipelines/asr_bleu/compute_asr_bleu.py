import asyncio
import logging
import os
import typing as tp
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from glob import glob

from stopes.pipelines.asr_bleu.configs import AsrBleuConfig
from stopes.pipelines.asr_bleu.utils import retrieve_asr_config, ASRGenerator
from stopes.pipelines.asr_bleu.transcribe_audio import transcribe_audiofile
from stopes.pipelines.asr_bleu.retrieve_data import retrieve_data

import hydra
import sacrebleu
from omegaconf import OmegaConf

logger = logging.getLogger("asr_bleu")

class AsrBleu:
    def __init__(self, config: AsrBleuConfig):
        self.config = config
        self.launcher = hydra.utils.instantiate(self.config.launcher)

    async def run(self):
        # 1. Retrieve ASR configuration 

        asr_config = retrieve_asr_config(self.config.corpora.lang, self.config.asr_version, json_path="/home/calderj/Documents/Coding/MLH/stopes/stopes/pipelines/asr_bleu/conf/asr_models/asr_model_cfgs.json")
        asr_model = ASRGenerator(asr_config)

        # 2. Compose evaluation data.

        #UNCOMMENT TO TEST (currently non functional) retrieve_data MODULE
        #eval_manifest = await retrieve_data(
        #    [(self.config.corpora.audio_dirpath, self.config.corpora.reference_path)], 
        #    self.launcher,
        #    self.config.corpora.audio_format,
        #    self.config.corpora.reference_format,
        #    self.config.corpora.reference_tsv_column
        #)

        eval_manifest = compose_eval_data(
            self.config.corpora.audio_dirpath,
            self.config.corpora.audio_format,
            self.config.corpora.reference_path,
            self.config.corpora.reference_format,
            self.config.corpora.reference_tsv_column
        )

        # 3. Transcribe audio predictions and compute BLEU score.
        prediction_transcripts = []
        for _, eval_pair in tqdm(
            eval_manifest.iterrows(),
            desc="Transcribing predictions",
            total=len(eval_manifest),
        ):
            transcription = await transcribe_audiofile(asr_model, eval_pair.prediction)
            prediction_transcripts.append(transcription.lower())

        if self.config.corpora.lang == "hok":
            prediction_transcripts = [
                merge_tailo_init_final(text) for text in prediction_transcripts
            ]

        references = eval_manifest["reference"].tolist()
        bleu_score = sacrebleu.corpus_bleu(prediction_transcripts, [references])

        print(bleu_score)

        
        return prediction_transcripts, bleu_score   

 

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


def extract_audio_for_eval(audio_dirpath: str, audio_format: str):
    if audio_format == "n_pred.wav":
        """
        The assumption here is that 0_pred.wav corresponds to the reference at line position 0 from the reference manifest
        """
        audio_list = []
        audio_fp_list = glob((Path(audio_dirpath) / "*_pred.wav").as_posix())
        audio_fp_list = sorted(
            audio_fp_list, key=lambda x: int(os.path.basename(x).split("_")[0])
        )
        for i in range(len(audio_fp_list)):
            try:
                audio_fp = (Path(audio_dirpath) / f"{i}_pred.wav").as_posix()
                assert (
                    audio_fp in audio_fp_list
                ), f"{Path(audio_fp).name} does not exist in {audio_dirpath}"
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


def extract_text_for_eval(
    references_filepath: str, reference_format: str, reference_tsv_column: str = None
):
    if reference_format == "txt":
        reference_sentences = open(references_filepath, "r").readlines()
        reference_sentences = [l.strip() for l in reference_sentences]
    elif reference_format == "tsv":
        tsv_df = pd.read_csv(references_filepath, sep="\t", quoting=3)
        reference_sentences = tsv_df[reference_tsv_column].to_list()
        reference_sentences = [l.strip() for l in reference_sentences]
    else:
        raise NotImplementedError

    return reference_sentences


def compose_eval_data(
    audio_dirpath: str,
    audio_format: str,
    references_filepath: str,
    reference_format: str,
    reference_tsv_column: str = None,
    save_manifest_filepath=None,
):
    """
    Speech matrix decoding pipeline produces audio with the following mask "N_pred.wav" where N is the order of the corresponding input sample
    Returns:
    pandas.DataFrame: the evaluation dataframe with columns `prediction` and `reference`
    """
    audio_list = extract_audio_for_eval(audio_dirpath, audio_format)
    reference_sentences = extract_text_for_eval(
        references_filepath, reference_format, reference_tsv_column
    )

    eval_df = pd.DataFrame(
        {
            "prediction": audio_list,
            "reference": reference_sentences,
        }
    )

    if save_manifest_filepath is not None:
        eval_df.to_csv(save_manifest_filepath, index=False)

    return eval_df


@hydra.main(config_path="conf", config_name="asr_bleu")
def main(config: AsrBleuConfig) -> None:
    pipeline = AsrBleu(config)
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()        
