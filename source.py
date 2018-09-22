#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Brno University of Technology FIT
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import os
import sys
import ctypes
import shutil
import logging
import argparse
import multiprocessing
import pickle

import numpy as np

from my_vbdiar import process_file
from my_vbdiar import get_num_segments
from my_vbdiar import get_segments
from my_vbdiar import get_vad
from my_vbdiar import EmbeddingSet
from utils import Utils
from utils import write_txt_matrix
from utils import read_txt_matrix
from extractors import KaldiXVectorExtraction
from extractors import KaldiMFCCFeatureExtraction

#from vbdiar.kaldi.mfcc_features_extraction import KaldiMFCCFeatureExtraction


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CDIR = os.path.dirname(os.path.realpath(__file__))


def generateMFCCs(MFCC_dir, config_name, regen=False, wav_dir="wav", vad_dir="vad",max_size=3000, tolerance=0):
    if not os.path.isdir(MFCC_dir):
        os.mkdir(MFCC_dir)

    config = Utils.read_config(config_name)
    config_mfcc = config['MFCC']
    config_path = os.path.abspath(config_mfcc['config_path'])
    if not os.path.isfile(config_path):
        raise ValueError('Path to MFCC configuration `{}` not found.'.format(config_path))
    features_extractor = KaldiMFCCFeatureExtraction(
        config_path=config_path, apply_cmvn_sliding=config_mfcc['apply_cmvn_sliding'],
        norm_vars=config_mfcc['norm_vars'], center=config_mfcc['center'], cmn_window=config_mfcc['cmn_window'])

    if regen:
        for the_file in os.listdir(MFCC_dir):
            file_path = os.path.join(MFCC_dir, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    files = [line.rstrip('\n') for line in open("lists/list.scp")]
    for file_name in files:
        logger.info('Processing file {}.'.format(file_name))

        wav_dir, vad_dir = os.path.abspath(wav_dir), os.path.abspath(vad_dir)

        # extract features
        tempfile, features = features_extractor.audio2features(os.path.join(wav_dir, '{}{}'.format(file_name, ".wav")))
        #shutil.copy(tempfile,os.path.join(os.path.abspath(folder_path),"{}{}".format(file_name.split('/')[-1],".mfcc")))
        os.remove(tempfile)

        # load voice activity detection from file
        vad, _, _ = get_vad('{}{}'.format(os.path.join(vad_dir, file_name), ".lab.gz"), features.shape[0])

        # parse segments and split features
        features_dict = {}
        for seg in get_segments(vad, max_size, tolerance):
            start, end = get_num_segments(seg[0]), get_num_segments(seg[1])
            if seg[0] > features.shape[0] - 1 or seg[1] > features.shape[0] - 1:
                raise ValueError('Unexpected features dimensionality - check VAD input or audio.')
            features_dict['{}_{}'.format(start, end)] = features[seg[0]:seg[1]]
        write_txt_matrix(os.path.join(os.path.abspath(MFCC_dir),"{}{}".format(file_name.split('/')[-1],".mfcc")),features_dict)
    return

def generateEmbeddings(MFCC_dir, config_name, embendding_dir):
    if not os.path.isdir(embendding_dir):
        os.mkdir(embendding_dir)

    config = Utils.read_config(config_name)
    config_embedding_extractor = config['EmbeddingExtractor']
    embedding_extractor = KaldiXVectorExtraction(
        nnet=os.path.abspath(config_embedding_extractor['nnet']), use_gpu=config_embedding_extractor['use_gpu'],
        min_chunk_size=config_embedding_extractor['min_chunk_size'],
        chunk_size=config_embedding_extractor['chunk_size'],
        cache_capacity=config_embedding_extractor['cache_capacity'])
    embs=[]
    files = [line.rstrip('\n') for line in open("lists/list.scp")]
    for file in files:
        file_name=file.split('/')[-1]
        features_dict=read_txt_matrix(os.path.join(os.path.abspath(MFCC_dir),"{}{}".format(file_name,".mfcc")))
        embedding_set = EmbeddingSet()
        emb_file, featur_file, embeddings = embedding_extractor.features2embeddings(features_dict)

        os.remove(emb_file.name)
        os.remove(featur_file.name)

        for embedding_key in embeddings:
            start, end = embedding_key.split('_')
            embedding_set.add(embeddings[embedding_key], window_start=int(float(start)), window_end=int(float(end)))
        write_txt_matrix(os.path.join(os.path.abspath(embendding_dir), "{}{}".format(file_name, ".xv")), embeddings)

    return


def getEmbedings(file_path):
    logger.info('Running `{}`.'.format(' '.join(sys.argv)))

    # initialize extractor
    config = Utils.read_config("vbdiar.yml")

    config_mfcc = config['MFCC']
    config_path = os.path.abspath(config_mfcc['config_path'])
    if not os.path.isfile(config_path):
        raise ValueError('Path to MFCC configuration `{}` not found.'.format(config_path))
    features_extractor = KaldiMFCCFeatureExtraction(
        config_path=config_path, apply_cmvn_sliding=config_mfcc['apply_cmvn_sliding'],
        norm_vars=config_mfcc['norm_vars'], center=config_mfcc['center'], cmn_window=config_mfcc['cmn_window'])

    config_embedding_extractor = config['EmbeddingExtractor']
    embedding_extractor = KaldiXVectorExtraction(
        nnet=os.path.abspath(config_embedding_extractor['nnet']), use_gpu=config_embedding_extractor['use_gpu'],
        min_chunk_size=config_embedding_extractor['min_chunk_size'],
        chunk_size=config_embedding_extractor['chunk_size'],
        cache_capacity=config_embedding_extractor['cache_capacity'])


    embeddings = process_file(
        file_name=file_path, wav_dir="wav", vad_dir="vad",
        features_extractor=features_extractor, embedding_extractor=embedding_extractor,
        max_size=3000, tolerance=0, wav_suffix=".wav",
        vad_suffix=".lab.gz", out_dir="")
    return embeddings



if __name__ == '__main__':
    generateMFCCs(MFCC_dir="MFCCs", config_name="vbdiar.yml", regen=True,
                  vad_dir="vad", wav_dir="wav", max_size=3000, tolerance=0)
    generateEmbeddings(MFCC_dir="MFCCs", config_name="vbdiar.yml", embendding_dir="embs")







