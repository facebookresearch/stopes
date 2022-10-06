

import os
import sys
source=str(sys.argv[1])
target=str(sys.argv[2])
data_name_or_path=str(sys.argv[3])
filen=sys.argv[4]
ofilen=sys.argv[5]

#'/private/home/costajussa/interpretability/nmt/data/'


outputFile = open(ofilen, 'w')


#print (data_name_or_path)

import torch
#torch.cuda.empty_cache()
#torch.cuda.set_device(1)
#torch.cuda.current_device()
#torch.cuda.memory_summary(device=None, abbreviated=False)
import warnings
from pathlib import Path

from wrappers.multilingual_transformer_wrapper import FairseqMultilingualTransformerHub

from fairseq.data.multilingual.multilingual_utils import (
    EncoderLangtok,
    LangTokSpec,
    LangTokStyle,
    augment_dictionary,
    get_lang_tok,
)

import alignment.align as align

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
import fairseq
import string
import logging
logger = logging.getLogger()
logger.setLevel('WARNING')
#warnings.simplefilter('ignore')

from dotenv import load_dotenv
load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

#print (device)

model_size = 'small' # small (412M) /big (1.2B)
data_sample = 'generate' # generate/interactive
teacher_forcing = False # teacher forcing/free decoding
only_alti = True


# Paths
# Checkpoint path
#
ckpt_dir = '/checkpoint/vedanuj/nmt/flores200/dense.mfp16.mu300000.uf1.lss.tmp1.lr0.002.drop0.1.maxtok4096.seed2.max_pos512.shem.NBF.adam16bit.fully_sharded.entsrc.det.transformer.ELS24.DLS24.ef8192.df8192.E2048.H16.AD0.1.RD0.0.ts_train_train_mining_train_mmt_bt_train_smt_bt.ngpu256/'
#
checkpoint_file = 'checkpoint_18_300000-shard0.pt'


#ckpt_dir ='/large_experiments/nllb/opensource/nllb_200_dense_3b/'
#ckpt_dir='/large_experiments/nllb/opensource/nllb_200_dense_distill_1b/'
#checkpoint_file = 'checkpoint.pt'




# Path to binarized data

hub = FairseqMultilingualTransformerHub.from_pretrained(
    ckpt_dir,
    checkpoint_file=checkpoint_file,
    data_name_or_path=data_name_or_path,
    dict_path='/large_experiments/nllb/mmt/multilingual_bin/validation.en_xx_en.v4.4/data_bin/shard000//dict.eng_Latn.txt',
    source_lang= source,
    target_lang= target,
    lang_pairs =source+'-'+target)
NUM_LAYERS = 24

def contrib_tok2words_partial(contributions, tokens, axis, reduction):
    from string import punctuation

    reduction_fs = {
        'avg': np.mean,
        'sum': np.sum
    }

    words = []
    w_contributions = []
    for counter, (tok, contrib) in enumerate(zip(tokens, contributions.T)):
        if tok.startswith('▁') or tok.startswith('__') or tok.startswith('<') or counter==0:# or tok in punctuation:
            if tok.startswith('▁'):
                tok = tok[1:]
            words.append(tok)
            w_contributions.append([contrib])
        else:
            words[-1] += tok
            w_contributions[-1].append(contrib)

    reduction_f = reduction_fs[reduction]
    word_contrib = np.stack([reduction_f(np.stack(contrib, axis=axis), axis=axis) for contrib in w_contributions], axis=axis)

    return word_contrib, words


def contrib_tok2words(contributions, tokens_in, tokens_out):
    word_contrib, words_in = contrib_tok2words_partial(contributions, tokens_in, axis=0, reduction='sum')
    word_contrib, words_out = contrib_tok2words_partial(word_contrib, tokens_out, axis=1, reduction='avg')
    return word_contrib.T, words_in, words_out


if data_sample=='generate':
    with open(data_name_or_path+str(filen)+"."+source, 'r') as fp:
        #first_line = fp.readline()    #
        for i, line in enumerate(fp):
            src_sent, src_tok, src_tensor, tgt_sent, tgt_tok, tgt_tensor = hub.get_sample(filen, i)
            pred_tok = tgt_tok
            # get_sample returns tensor [lang_tok, ..., </s>]
            # we need [</s>, lang_tok, ...] to feed into the decoder
            tgt_tensor = torch.cat([torch.tensor([tgt_tensor[-1]]), tgt_tensor[:-1]])
            # same for subwords list
            tgt_tok = [tgt_tok[-1]] + tgt_tok[:-1]


            if only_alti==False:

                if teacher_forcing:
                  #  print (tgt_tensor)
                    
                    model_output, log_probs, encoder_out, layer_inputs, layer_outputs = hub.trace_forward(src_tensor, tgt_tensor)
                    
                    #print("\n\nGREEDY DECODING\n")
                    pred_log_probs, pred_tensor = torch.max(log_probs, dim=-1)
                    pred_tok = hub.decode(pred_tensor, hub.task.target_dictionary)
                    pred_sent = hub.decode(pred_tensor, hub.task.target_dictionary, as_string=True)
                    #print(f"Predicted sentence: \t {pred_sent}")
                    
                    
                if not teacher_forcing: #we do not need a reference here
                    tgt_tensor_free = []
               
                    #print("\n\nBEAM SEARCH\n")
                    src_tensor = src_tensor[1:]
                    for pred in hub.generate(src_tensor,4,verbose=True): #added 0
                        tgt_tensor_free.append(pred['tokens'])
                        pred_sent = hub.decode(pred['tokens'], hub.task.target_dictionary, as_string=True)
                        score = pred['score'].item()


                    hypo = 0# first hypothesis we do teacher forcing with the best hypothesis
                    tgt_tensor = tgt_tensor_free[hypo]

                    # We add eos token at the beginning of sentence and delete it from the end
                    tgt_tensor = torch.cat([torch.tensor([hub.task.target_dictionary.eos_index]).to(tgt_tensor.device),tgt_tensor[:-1]]).to(tgt_tensor.device)
                    tgt_tok = hub.decode(tgt_tensor, hub.task.target_dictionary, as_string=False)
                    target_sentence = tgt_tok
                    # We assume the predicted sentence is the same as the top hypothesis
                    pred_tok = tgt_tok[1:] + ['</s>']

            source_sentence = src_tok
            target_sentence = tgt_tok
            predicted_sentence = pred_tok

            # Output of ALTI+
            total_rollout = hub.get_contribution_rollout(src_tensor, tgt_tensor,
                                            'l1', norm_mode='min_sum',
                                            pre_layer_norm=True)['total']
            contributions_rollout_layer = total_rollout[-1]
            contributions_rollout_layer_np = contributions_rollout_layer.detach().cpu().numpy()

            # Source contributions average (over all sentence)
            src_contribution = contributions_rollout_layer_np[:,:len(src_tok)].sum(-1) #src contribution for target token, 
            src_contribution_mean= np.mean(src_contribution)


            # Word-word alignment with ALTI+
            contributions_rollout_layer_np, words_in, words_out = contrib_tok2words(
                contributions_rollout_layer_np,
                tokens_in=(source_sentence + target_sentence),
                tokens_out=predicted_sentence
            )
            # Now contributions_rollout_layer_np is a word-to-word matrix

            # We obtain only source words (until </s>)
            source_sentence_ = words_in[:words_in.index('</s>')+1]
            # We obtain only target words (from </s>)
            target_sentence_ = words_in[words_in.index('</s>')+1:]
            # We obtain predicted words (should match target_sentence_)
            predicted_sentence_ = words_out




            # get source contribution

            src_cont=""
            for idx, word in enumerate(predicted_sentence_):
                src_contrib = contributions_rollout_layer_np[idx,:len(source_sentence_)].sum()
                src_cont+=" "+str(src_contrib)
            src_cont.strip()
            print (src_cont, file = outputFile)
            
            ############

            alignments_src_tgt = contributions_rollout_layer_np[:,:len(source_sentence_)]
            # Eliminate language tags
            alignments_src_tgt = alignments_src_tgt[1:,1:]
            # We don't consider alignment of EOS (target)
            alignments_src_tgt = alignments_src_tgt[:-1]

            ## Hard alignment (via argmax per row)
            a_argmax = np.argmax(alignments_src_tgt, -1)
            contributions_word_word_hard = np.zeros(alignments_src_tgt.shape)

            sent=""
            for i, j in enumerate(a_argmax):
                contributions_word_word_hard[i][j] = 1
                sent+=str(i)+"-"+str(j)+" "

            print (sent)
            # contributions_word_word_hard is a matrix with alignments

            # Plot heatmap with aligments
            # plt.figure(figsize=(20,8))
            # df = pd.DataFrame(contributions_word_word_hard, columns = source_sentence_[1:], index = predicted_sentence_[1:-1])
            # sns.set(font_scale=1.6)
            # sns.heatmap(df,cmap="Blues",square=True,cbar=True);


outputFile.close()
