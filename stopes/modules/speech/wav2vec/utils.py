# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Different functions that interface with fairseq.models.wav2vec

import logging
from pathlib import Path

import fairseq  # type: ignore[import]
import numpy as np
import torch
from omegaconf import OmegaConf

logger = logging.getLogger("stopes.preprocess.wav2vec_compat")

# compat config from base xlsr model
# fmt: off
compat_args = {'_name': None, 'bmuf': {'_name': None, 'average_sync': False, 'block_lr': 1.0, 'block_momentum': 0.875, 'distributed_world_size': 200, 'global_sync_iter': 50, 'use_nbm': False, 'warmup_iterations': 500}, 'bpe': None, 'checkpoint': {'_name': None, 'best_checkpoint_metric': 'loss', 'checkpoint_shard_count': 1, 'checkpoint_suffix': '', 'finetune_from_model': None, 'keep_best_checkpoints': -1, 'keep_interval_updates': 20, 'keep_interval_updates_pattern': -1, 'keep_last_epochs': -1, 'load_checkpoint_on_all_dp_ranks': False, 'maximize_best_checkpoint_metric': False, 'model_parallel_size': 1, 'no_epoch_checkpoints': True, 'no_last_checkpoints': False, 'no_save': False, 'no_save_optimizer_state': False, 'optimizer_overrides': '{}', 'patience': -1, 'reset_dataloader': False, 'reset_lr_scheduler': False, 'reset_meters': False, 'reset_optimizer': False, 'restore_file': 'checkpoint_last.pt', 'save_dir': 'checkpoints', 'save_interval': 1, 'save_interval_updates': 25001, 'write_checkpoints_asynchronously': False}, 'common': {'_name': None, 'all_gather_list_size': 16384, 'azureml_logging': False, 'bf16': False, 'cpu': False, 'empty_cache_freq': 0, 'fp16': True, 'fp16_init_scale': 16384, 'fp16_no_flatten_grads': False, 'fp16_scale_tolerance': 0.0, 'fp16_scale_window': None, 'log_file': None, 'log_format': 'json', 'log_interval': 200, 'memory_efficient_bf16': False, 'memory_efficient_fp16': False, 'min_loss_scale': 0.0001, 'model_parallel_size': 1, 'no_progress_bar': False, 'plasma_path': '/dev/null', 'profile': False, 'quantization_config_path': None, 'reset_logging': False, 'seed': 1, 'suppress_crashes': False, 'tensorboard_logdir': 'tb', 'threshold_loss_scale': None, 'tpu': False, 'use_plasma_view': False, 'user_dir': None, 'wandb_project': None}, 'common_eval': {'_name': None, 'model_overrides': '{}', 'path': None, 'post_process': None, 'quiet': False, 'results_path': None}, 'criterion': {'_name': 'wav2vec', 'infonce': True, 'log_keys': ['prob_perplexity', 'code_perplexity', 'temp'], 'loss_weights': [0.1, 0.0]}, 'dataset': {'_name': None, 'batch_size': None, 'batch_size_valid': None, 'curriculum': 0, 'data_buffer_size': 10, 'dataset_impl': None, 'disable_validation': False, 'fixed_validation_seed': None, 'gen_subset': 'test', 'max_tokens': 1024000, 'max_tokens_valid': 1024000, 'max_valid_steps': None, 'num_shards': 1, 'num_workers': 1, 'required_batch_size_multiple': 8, 'required_seq_len_multiple': 1, 'shard_id': 0, 'skip_invalid_size_inputs_valid_test': True, 'train_subset': 'train', 'valid_subset': 'valid_total_downsampled', 'validate_after_updates': 0, 'validate_interval': 50, 'validate_interval_updates': 0}, 'distributed_training': {'_name': None, 'broadcast_buffers': False, 'bucket_cap_mb': 25, 'cpu_offload': False, 'ddp_backend': 'fully_sharded', 'device_id': 0, 'distributed_backend': 'nccl', 'distributed_init_method': '', 'distributed_no_spawn': True, 'distributed_port': 43150, 'distributed_rank': 0, 'distributed_world_size': 200, 'fast_stat_sync': False, 'find_unused_parameters': False, 'fix_batches_to_gpus': False, 'fp16': True, 'fp32_reduce_scatter': False, 'heartbeat_timeout': -1, 'localsgd_frequency': 3, 'memory_efficient_fp16': False, 'no_reshard_after_forward': False, 'nprocs_per_node': 1, 'pipeline_balance': None, 'pipeline_checkpoint': 'never', 'pipeline_chunks': 0, 'pipeline_decoder_balance': None, 'pipeline_decoder_devices': None, 'pipeline_devices': None, 'pipeline_encoder_balance': None, 'pipeline_encoder_devices': None, 'pipeline_model_parallel': False, 'slowmo_algorithm': 'LocalSGD', 'slowmo_momentum': None, 'tpu': False, 'zero_sharding': 'none'}, 'eval_lm': {'_name': None, 'context_window': 0, 'output_word_probs': False, 'output_word_stats': False, 'softmax_batch': 9223372036854775807}, 'generation': {'_name': None, 'beam': 5, 'constraints': None, 'decoding_format': None, 'diverse_beam_groups': -1, 'diverse_beam_strength': 0.5, 'diversity_rate': -1.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_force_max_iter': False, 'iter_decode_max_iter': 10, 'iter_decode_with_beam': 1, 'iter_decode_with_external_reranker': False, 'lenpen': 1.0, 'lm_path': None, 'lm_weight': 0.0, 'match_source_len': False, 'max_len_a': 0.0, 'max_len_b': 200, 'min_len': 1, 'nbest': 1, 'no_beamable_mm': False, 'no_early_stop': False, 'no_repeat_ngram_size': 0, 'no_seed_provided': False, 'prefix_size': 0, 'print_alignment': None, 'print_step': False, 'replace_unk': None, 'retain_dropout': False, 'retain_dropout_modules': None, 'retain_iter_history': False, 'sacrebleu': False, 'sampling': False, 'sampling_topk': -1, 'sampling_topp': -1.0, 'score_reference': False, 'temperature': 1.0, 'unkpen': 0.0, 'unnormalized': False}, 'interactive': {'_name': None, 'buffer_size': 0, 'input': '-'}, 'job_logging_cfg': {'disable_existing_loggers': False, 'formatters': {'simple': {'format': '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'}}, 'handlers': {'console': {'class': 'logging.StreamHandler', 'formatter': 'simple', 'stream': 'ext://sys.stdout'}, 'file': {'class': 'logging.FileHandler', 'filename': 'hydra_train.log', 'formatter': 'simple'}}, 'root': {'handlers': ['console', 'file'], 'level': 'INFO'}, 'version': 1}, 'lr_scheduler': {'_name': 'polynomial_decay', 'end_learning_rate': 0.0, 'force_anneal': None, 'lr': [8e-05], 'power': 1.0, 'total_num_update': 1000000, 'warmup_updates': 32000}, 'model': {'_name': 'wav2vec2', 'activation_dropout': 0.0, 'activation_fn': 'gelu', 'attention_dropout': 0.0, 'checkpoint_activations': True, 'codebook_negatives': 0, 'conv_bias': True, 'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]', 'conv_pos': 128, 'conv_pos_groups': 16, 'cross_sample_negatives': 0, 'dropout': 0.0, 'dropout_features': 0.1, 'dropout_input': 0.1, 'encoder_attention_heads': 16, 'encoder_embed_dim': 1920, 'encoder_ffn_embed_dim': 7680, 'encoder_layerdrop': 0.0, 'encoder_layers': 48, 'extractor_mode': 'layer_norm', 'feature_grad_mult': 1.0, 'final_dim': 1024, 'latent_dim': 0, 'latent_groups': 2, 'latent_temp': [2.0, 0.1, 0.999995], 'latent_vars': 320, 'layer_norm_first': True, 'logit_temp': 0.1, 'mask_channel_length': 10, 'mask_channel_min_space': 1, 'mask_channel_other': 0.0, 'mask_channel_prob': 0.0, 'mask_channel_selection': 'static', 'mask_length': 10, 'mask_min_space': 1, 'mask_other': 0.0, 'mask_prob': 0.65, 'mask_selection': 'static', 'negatives_from_everywhere': False, 'no_mask_channel_overlap': False, 'no_mask_overlap': False, 'num_negatives': 100, 'offload_activations': False, 'quantize_input': False, 'quantize_targets': True, 'same_quantizer': False, 'target_glu': False}, 'optimization': {'_name': None, 'clip_norm': 0.0, 'lr': [8e-05], 'max_epoch': 0, 'max_update': 1000000, 'sentence_avg': False, 'stop_min_lr': -1.0, 'stop_time_hours': 0.0, 'update_freq': [1], 'use_bmuf': False}, 'optimizer': {'_name': 'adam', 'adam_betas': '(0.9,0.98)', 'adam_eps': 1e-06, 'lr': [8e-05], 'tpu': False, 'use_old_adam': False, 'weight_decay': 0.01}, 'scoring': None, 'task': {'_name': 'audio_pretraining', 'data': '/dev/null', 'enable_padding': False, 'labels': None, 'max_sample_size': 320000, 'min_sample_size': 32000, 'normalize': True, 'num_batch_buckets': 0, 'precompute_mask_indices': False, 'sample_rate': 16000, 'tpu': False}, 'tokenizer': None}
# fmt: on

SAMPLING_RATE = 16000


def load_speech_encoder(checkpoint_dir: Path, checkpoint_file: str):
    """
    Load a speech encoder by trying several methods:
    1. Using generic "load_model_ensemble_and_task" from fairseq;
    2. Using the specific Wav2VecLaser class, if it is implemented in fairseq;
    3. Using lower-leavel loading with patched config.
    Return the model, its config, and its Fairseq task.
    """
    model_path = str(checkpoint_dir / checkpoint_file)
    try:
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [model_path]
        )
        encoder = model[0]
        return encoder, cfg, task
    except AssertionError:  # could not infer the task type because of a wrong branch
        # we will do some other attempts, so just passing here
        pass

    try:
        # This code depends on fairseq branch `ust`
        from fairseq.models.wav2vec import Wav2VecLaser  # type: ignore[import]

        laser = Wav2VecLaser.from_pretrained(checkpoint_dir, checkpoint_file)

        return laser.models[0], laser.cfg, laser.task
    except ImportError:
        logger.error(
            "Wav2VecLaser is not defined in fairseq.models.wav2vec, check that your fairseq branch has this model or use `main` or `ust`"
        )
    except AssertionError:
        logger.warning(
            f"{checkpoint_dir/checkpoint_file} refers to an unknown task, converting to a compatible task."
        )

        # we don't have the right task from faiseq, convert the model config
        state = torch.load(str(checkpoint_dir / checkpoint_file))
        cfg = OmegaConf.create(state["cfg"])

        # change task from the model
        to_del = ["langs", "alpha", "max_source_positions", "laser"]
        for el in to_del:
            del cfg["task"][el]
        del cfg["model"]["load_laser_model"]

        cfg["task"].update({"_name": "audio_finetuning"})
        cfg["model"].update({"w2v_path": "", "w2v_args": compat_args})

        task = fairseq.tasks.setup_task(cfg["task"])
        model = task.build_model(cfg["model"])
        model.load_state_dict(state["model"], strict=True, model_cfg=cfg["model"])
        return model, cfg, task


class WavesDataset(fairseq.data.audio.raw_audio_dataset.RawAudioDataset):
    """
    The Audio Dataset that wraps the audio waveforms.

    NOTE: We want to keep this module neutral to mining logic, and this class
    is used only within the mining context. It is kept here for 2 reasons:
    1) This module is the only place `AudioDataset` is referenced.
    2) This module imports the laser embedding-related part of fairseq-py.
      Keeping AudioDataset here encapsulates the fairseq dependence.

    Consider refactor this module to decouple further the laser-wav2vec and
    the mining logic
    """

    def __init__(self, wav_list, sizes, fbank_features=0):
        super().__init__(
            sample_rate=SAMPLING_RATE,
            shuffle=False,
            pad=True,
            normalize=True,
            compute_mask_indices=False,
            fbank_features=fbank_features,
        )
        self.wav_list = wav_list
        self.sizes = sizes

    def __getitem__(self, index):
        if self.fbank_features > 0:
            normalization = False
            always_2d = True

            # handling empty or very small segments
            # adding a small 400 frames of silence
            # this is enough to generate dummy embeddings
            # and keep the segment order the same
            if self.sizes[index] < 400:
                waveform = np.array([[0] * 400], dtype=np.float32)
            else:
                # we are doing copy because we are doing destructive operations a few lines below: waveform *= 255
                waveform = self.wav_list[index].reshape(1, -1).copy()
            # replicating fairseq code from `FileAudioDataset.__getitem__`
            # with `self.fbank_features > 0`
            waveform, _ = fairseq.data.audio.audio_utils.convert_waveform(
                waveform,
                sample_rate=SAMPLING_RATE,
                normalize_volume=False,
                to_mono=True,
                to_sample_rate=None,
            )
            if not normalization:
                waveform *= 2**15  # denormalized to 16-bit signed integers
            if not always_2d:
                waveform = waveform.squeeze(axis=0)
            feats = fairseq.data.audio.audio_utils._get_torchaudio_fbank(
                waveform, SAMPLING_RATE, n_bins=self.fbank_features
            )
            feats = self.postprocess_fbank(feats)
            feats = torch.tensor(feats)
        else:
            waveform = torch.tensor(self.wav_list[index])  # type: ignore[assignment]
            feats = self.postprocess(waveform, SAMPLING_RATE)
        return {"id": index, "source": feats}
