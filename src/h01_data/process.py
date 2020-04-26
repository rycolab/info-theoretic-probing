# Process a UD file to embeddings
import os
import sys
import bz2
import pickle
import argparse
import logging
import numpy as np
import torch
from conllu import parse_incr
from scipy.sparse import lil_matrix
from transformers import BertTokenizer, BertModel
import fasttext
import fasttext.util

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import constants
from util import util
from util.ud_list import UD_LIST


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--batch-size",
                        help="The size of the mini batches",
                        default=8,
                        required=False,
                        type=int)
    parser.add_argument("--language",
                        help="The language to use",
                        required=True,
                        type=str)
    parser.add_argument("--ud-path",
                        help="The path to raw ud data",
                        default='data/ud/ud-treebanks-v2.5/',
                        required=False,
                        type=str)
    parser.add_argument("--output-path",
                        help="The path to save processed data",
                        default='data/processed/',
                        required=False,
                        type=str)
    args = parser.parse_args()
    logging.info(args)

    return args


def get_ud_file_base(ud_path, language):
    return os.path.join(ud_path, UD_LIST[language])


def get_data_file_base(output_path, language):
    output_path = os.path.join(output_path, language)
    util.mkdir(output_path)
    return os.path.join(output_path, '%s--%s.pickle.bz2')


def load_bert(bert_name):
    bert_tokenizer = BertTokenizer.from_pretrained(bert_name)
    bert_model = BertModel.from_pretrained(bert_name).to(device=constants.device)
    bert_model.eval()

    return bert_tokenizer, bert_model


def tokenize_ud(file_name, bert_tokenizer):
    all_ud_tokens = []
    all_bert_tokens = []
    all_bert2target_map = []
    all_tree_matrices = []
    all_ud_data = []

    # Initialise all the trees and embeddings
    with open(file_name, "r", encoding="utf-8") as file:
        for token_list in parse_incr(file):

            # Extract text tokens for each sentence
            ud_tokens = []
            ud_data = []

            for t in token_list:
                ud_tokens.append(t['form'])
                ud_data.append({
                    'word': t['form'],
                    'pos': t['upostag'],
                    'head': t['head'],
                    'rel': t['deprel'],
                })

            # Tokenize the sentence
            ud2bert_mapping = []
            bert_tokens = []
            for token in ud_tokens:
                bert_decomposition = bert_tokenizer.tokenize(token)
                if len(bert_decomposition) == 0:
                    bert_decomposition = ['[UNK]']

                bert_tokens += bert_decomposition
                ud2bert_mapping.append(len(bert_decomposition))

            # If there are more than 512 tokens skip the sentence (since that's over BERT's max size)
            if not len(bert_tokens) > 510:
                all_ud_tokens.append(ud_tokens)
                all_bert2target_map.append(ud2bert_mapping)
                all_bert_tokens.append(bert_tokens)
                all_ud_data.append(ud_data)

    return all_ud_tokens, all_bert_tokens, all_bert2target_map, all_ud_data


def embed_bert(all_bert_tokens, batch_size, model, bert_tokenizer):
    all_bert_embeddings = []

    batch_num = 0
    for batch_start in range(0, len(all_bert_tokens), batch_size):

        batch_num += 1
        if batch_num % 10 == 0:
            logging.info("Processing batch {} to embeddings".format(batch_num))

        # Get the batch
        batch_end = batch_start + batch_size
        batch = all_bert_tokens[batch_start:batch_end]

        lengths = [(len(sentence) + 2) for sentence in batch]  # +2 for CLS/SEP
        longest_sent = max(lengths)

        padded_line = ["[PAD]"] * longest_sent

        attention_off = [0] * longest_sent
        attention_on = [1] * longest_sent

        padded_batch = []
        attention_mask = []
        # Mask is 1 for tokens that are NOT MASKED, 0 for MASKED tokens.

        for sentence in range(len(batch)):
            sentence_len = lengths[sentence]
            padded_batch.append(bert_tokenizer.convert_tokens_to_ids(
                ["[CLS]"] + batch[sentence] + ["[SEP]"] + padded_line[sentence_len:]))
            attention_mask.append(attention_on[:sentence_len] + attention_off[sentence_len:])

        # Pad it & build up attention mask
        input_ids = torch.tensor(padded_batch).to(device=constants.device)
        attention_mask_tensor = torch.tensor(attention_mask).to(device=constants.device)

        outputs, _ = model(input_ids, attention_mask=attention_mask_tensor)

        # Remove paddings
        last_hidden_states = [x[1:lengths[i] - 1] for i, x in enumerate(outputs.cpu().numpy())]  # Discards CLS and SEP

        all_bert_embeddings += last_hidden_states

    return all_bert_embeddings


def process_bert_token(token):
    if token.startswith("##"):
        # Chop off the '##'
        return token[2:]

    return token


def check_bert_word(word, bert_tokens, target_tokens):
    word_bert = "".join([
        process_bert_token(token)
        for token in bert_tokens
    ])

    if word_bert == word:
        return True

    # BERT has had something it doesn't recognise, so skip the sentence
    logging.warning("Failed to embed \'{}\' from BERT tokens {} in sentence {}"
                    .format(word,
                            '+'.join(bert_tokens),
                            '+'.join(target_tokens)))
    saved = []
    return False  # Skip the whole sentence


def combine_bert(all_target_token, all_bert2target_map, all_bert_tokens, all_bert_embeddings):
    output_embeddings = []
    output_words = []

    sentence_num = 0

    for sentence in range(len(all_target_token)):

        sentence_num += 1
        if sentence_num % 10000 == 0:
            logging.info("Re-merging and saving sentence {}".format(sentence_num))

        target_tokens = all_target_token[sentence]
        target_tokens_mapping = all_bert2target_map[sentence]
        bert_tokens = all_bert_tokens[sentence]
        last_hidden_states = all_bert_embeddings[sentence]

        bert_index = 0

        saved = []
        words = []

        bert_index_start, bert_index_end = 0, 0

        for target_index in range(0, len(target_tokens)):
            assert len(target_tokens_mapping) == len(target_tokens)

            num_of_bert_toks = target_tokens_mapping[target_index]
            word = target_tokens[target_index]
            bert_index_end += num_of_bert_toks

            # if not check_bert_word(word, bert_tokens[bert_index_start: bert_index_end], target_tokens):
            #     saved = []
            #     break

            embedding = last_hidden_states[bert_index_start: bert_index_end].mean(axis=0)

            saved.append(embedding)
            words.append(word)

            bert_index_start = bert_index_end

        # Save the embeddings - if they were erronous the list is empty
        output_embeddings.append(saved)
        output_words.append(words)

    return output_embeddings, output_words


def load_fasttext(language):
    lang = constants.LANGUAGE_CODES[language]
    ft_path = 'data/fasttext'
    ft_fname = os.path.join(ft_path, 'cc.%s.300.bin' % lang)
    if not os.path.exists(ft_fname):
        logging.info("Downloading fasttext model")
        temp_fname = fasttext.util.download_model(lang, if_exists='ignore')
        util.mkdir(ft_path)
        os.rename(temp_fname, ft_fname)
        os.rename(temp_fname + '.gz', ft_fname + '.gz')

    logging.info("Loading fasttext model")
    return fasttext.load_model(ft_fname)


def get_fasttext(fasttext_model, words):
    embeddings = [[fasttext_model[word] for word in sentence]for sentence in words]
    return embeddings


def process_file(bert_model, bert_tokenizer, fasttext_model, batch_size, language, ud_file, output_file):
    logging.info("Processing file {}".format(ud_file))

    logging.info("PHASE ONE: reading file and tokenizing")
    all_target_tokens, all_bert_tokens, all_bert2target_map, all_ud = tokenize_ud(ud_file, bert_tokenizer)

    logging.info("PHASE TWO: padding, batching, and embedding for bert")
    all_bert_embeddings = embed_bert(all_bert_tokens, batch_size, bert_model, bert_tokenizer)

    logging.info("PHASE THREE: re-merging BERT tokens")
    bert_embeddings, words = combine_bert(all_target_tokens, all_bert2target_map, all_bert_tokens,
                                          all_bert_embeddings)
    del all_target_tokens, all_bert2target_map, all_bert_tokens, all_bert_embeddings

    logging.info("PHASE FOUR: getting fasttext embeddings")
    fast_embeddings = get_fasttext(fasttext_model, words)

    logging.info("PHASE FIVE: saving")
    output_data_raw = list(zip(bert_embeddings, fast_embeddings, all_ud, words))
    del bert_embeddings, fast_embeddings, all_ud, words

    # Prune the failed attempts:
    output_data = [(bert_embs, fast_embs, ud, words) for (bert_embs, fast_embs, ud, words) in output_data_raw if bert_embs != []]
    del output_data_raw
    output_ud = [(ud, words) for (_, _, ud, words) in output_data]
    output_bert = [(bert_embs, words) for (bert_embs, _, _, words) in output_data]
    output_fast = [(fast_embs, words) for (_, fast_embs, _, words) in output_data]
    del output_data

    # Pickle, compress, and save
    util.write_data(output_file % 'ud', output_ud)
    del output_ud
    util.write_data(output_file % 'fast', output_fast)
    del output_fast
    util.write_data(output_file % 'bert', output_bert)
    del output_bert

    logging.info("Completed {}".format(ud_file))


def process(language, ud_path, batch_size, bert_name, output_path):
    logging.info("Loading pre-trained BERT network")
    bert_tokenizer, bert_model = load_bert(bert_name)
    fasttext_model = load_fasttext(language)

    logging.info("Precessing language %s" % language)
    ud_file_base = get_ud_file_base(ud_path, language)
    output_file_base = get_data_file_base(output_path, language)
    for mode in ['train', 'dev', 'test']:
        ud_file = ud_file_base % mode
        output_file = output_file_base % (mode, '%s')
        process_file(bert_model, bert_tokenizer, fasttext_model, batch_size, language, ud_file, output_file)

    logging.info("Process finished")


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(processName)s : %(message)s', level=logging.INFO)
    args = get_args()

    batch_size = args.batch_size
    language = args.language
    ud_path = args.ud_path
    output_path = args.output_path
    bert_name = 'bert-base-multilingual-cased'

    with torch.no_grad():
        process(language, ud_path, batch_size, bert_name, output_path)


if __name__ == "__main__":
    main()
