from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from rank_bm25 import BM25Okapi
# from gensim.models.fasttext import FastText
import nmslib
import re
import numpy as np
import pickle
import time
import json
from os.path import join, dirname, abspath
from os import walk
# from farasa.segmenter import FarasaSegmenter
from DATA.Arabert.preprocess_arabert import never_split_tokens, preprocess
import torch
import torch.nn as nn

project_root = dirname(dirname(__file__)).replace('RTA_Legislations_Chatbot', '')
data_Path = join(join(project_root, 'Data'), 'Legislations_Chatbot')

raBERT_models_Path = join(join(join(project_root, 'Data'), 'Arabert'), 'AraBERT_models')
arabert_tokenizer = AutoTokenizer.from_pretrained(
    join(raBERT_models_Path, 'bert-base-arabert'),
    do_lower_case=False,
    do_basic_tokenize=False,
    never_split=never_split_tokens)
# you can replace the path here with the folder containing the the pytorch model
arabert_model = AutoModel.from_pretrained(join(raBERT_models_Path, 'bert-base-arabert'))


# farasa_segmenter = FarasaSegmenter(interactive=False)


def getembeddings(text=''):
    global raBERT_models_Path
    global arabert_tokenizer
    global arabert_model
    # global farasa_segmenter
    text_preprocessed = preprocess(text, do_farasa_tokenization=False, use_farasapy=False)
    arabert_input = arabert_tokenizer.encode(text_preprocessed, add_special_tokens=False)
    embeddings = torch.sum(arabert_model(torch.tensor(arabert_input).unsqueeze(0)).last_hidden_state,
                           dim=1).detach().numpy() / len(arabert_input)  # [0][0][1:-1]

    return embeddings


def absoluteFilePaths(directory):
    for dirpath, _, filenames in walk(directory):
        for f in filenames:
            if f.lower().endswith('.json'):
                yield abspath(join(dirpath, f))


def clean_str(text):
    # remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)

    a = ['ض', 'ص', 'ث', 'ق', 'ف', 'غ', 'ع', 'ه', 'خ', 'ح', 'ج', 'د', 'ش', 'س', 'ي', 'ب', 'ل', 'ا', 'ت', 'ن', 'م', 'ك',
         'ط', 'ئ', 'ء', 'ؤ', 'ر', 'لا', 'ى', 'ة', 'و', 'ز', 'ظ', 'لإ', 'إ', 'لأ', 'أ', '~', 'لآ', 'آ', 'ذ', '1', '2',
         '3', '4', '5', '6', '7', '8', '9', '0', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩', '٠']
    r = re.compile('[^%s]+' % ''.join(a))
    text = r.sub(' ', text)
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t',
              '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ',
               ' ! ']

    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)

    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # trim
    text = text.strip()

    return text


def flattenjson(b):
    val = {}
    for i in b.keys():
        if isinstance(b[i], dict):
            get = flattenjson(b[i])
            for j in get.keys():
                val[i + '_' + j] = get[j]

        elif isinstance(b[i], list):
            for idx, items in enumerate(b[i]):
                if isinstance(items, dict):
                    get = flattenjson(items)
                    for k in get.keys():
                        val[i + '_' + str(idx) + '_' + k] = get[k]
                else:
                    val[i] = items
        else:
            val[i] = b[i]
    return val


def loadData():
    tok_text = []
    tok_key = []
    paths = absoluteFilePaths(join(data_Path, 'AR'))
    for path in paths:
        try:
            with open(path, encoding='utf-8') as f:
                dd = flattenjson(json.load(f))
                k = str(dd['Type']) + '_' + str(dd['No']) + '_' + str(dd['Year']) + '_' + str(dd['Concerning']) + '_'
                for key, val in dd.items():
                    s = clean_str(str(val)).split() + clean_str(' '.join(str(key).split('_'))).split()
                    if len(s) > 2:
                        tok_text.append(s)
                        tok_key.append(k + key)
        except:
            pass
    # ft_model = FastText.load(join(data_Path, '_fasttext.model'))  # load
    return tok_key, tok_text  # , ft_model


"""
def createFastTextModel(tok_text):
    ft_model = FastText(
        sg=1,  # use skip-gram: usually gives better results
        size=100,  # embedding dimension (default)
        window=10,  # window size: 10 tokens before and 10 tokens after to get wider context
        min_count=5,  # only consider tokens with at least n occurrences in the corpus
        negative=15,  # negative subsampling: bigger than default to sample negative examples more
        min_n=2,  # min character n-gram
        max_n=5  # max character n-gram
    )

    # tok_text is our tokenized input text - a list of lists relating to docs and tokens respectivley
    ft_model.build_vocab(tok_text)
    ft_model.train(
        tok_text,
        epochs=6,
        total_examples=ft_model.corpus_count,
        total_words=ft_model.corpus_total_words)

    ft_model.save(join(data_Path, '_fasttext.model'))  # save
"""


def createmodel(tok_text=[]):
    # ft_model = FastText.load(join(data_Path, '_fasttext.model'))
    weighted_doc_vects = []
    bm25 = BM25Okapi(tok_text)
    for i, dd in tqdm(enumerate(tok_text)):
        txt = ' '.join(dd)

        doc_vector = []
        for word in dd:
            # vector = ft_model[word]
            vector = getembeddings(word)
            weight = (bm25.idf[word] * ((bm25.k1 + 1.0) * bm25.doc_freqs[i][word])) / (
                    bm25.k1 * (1.0 - bm25.b + bm25.b * (bm25.doc_len[i] / bm25.avgdl)) + bm25.doc_freqs[i][word])
            weighted_vector = vector * weight
            doc_vector.append(weighted_vector)
        doc_vector_mean = np.mean(doc_vector, axis=0)
        weighted_doc_vects.append(doc_vector_mean)
        pickle.dump(weighted_doc_vects, open(join(data_Path, "weighted_doc_vects.p"), "wb"))  # save the results to disc

    # create a matrix from our document vectors
    try:
        data = np.vstack(weighted_doc_vects)
    except Exception as ex:
        pass

    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(data)
    index.createIndex({'post': 2}, print_progress=True)
    index.saveIndex(join(data_Path, '_NMSLIB.index'), save_data=True)


def search(input, tok_key, tok_text):
    input = clean_str(input).split()
    query = [getembeddings(word) for word in input]
    query = np.mean(query, axis=0)
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.loadIndex(join(data_Path, '_NMSLIB.index'), load_data=True)
    ids, distances = index.knnQuery(query, k=10)
    results = []
    for i, j in zip(ids, distances):
        results.append((str(tok_key[i]), str(tok_text[i])))
    return results


def main():
    tok_key, tok_text = loadData()
    createmodel(tok_text)
    input = clean_str('ما هو قانون المخالفات لقيادة العبارة بدون تصريح ').split()
    query = [getembeddings(word) for word in input]
    query = np.mean(query, axis=0)
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.loadIndex(join(data_Path, '_NMSLIB.index'), load_data=True)
    ids, distances = index.knnQuery(query, k=10)

    for i, j in zip(ids, distances):
        print('Confidence is ' + str(round(1 - j, 3)) + '\nkey is ' + str(tok_key[i]) + '\nText is ' + str(tok_text[i]))
        print('----------------------------------------------------------------------------')


if __name__ == "__main__":
    main()
