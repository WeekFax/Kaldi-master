import os
import h5py
import librosa
import numpy as np

from extractors import VoiceActivityDetector
from extractors import KaldiMFCCFeatureExtraction
from extractors import KaldiXVectorExtraction
from utils import Utils
from utils import write_wav

from plda.classifier import Classifier


def wav_to_vad(wav_file, vad_file, sr=8000):
    audio, rate = librosa.load(wav_file, sr=sr)
    v = VoiceActivityDetector()
    write_wav(vad_file, v.get_speech(audio), rate)


def vad_all_files(to_dir_path, from_dir_path):
    if not os.path.exists(to_dir_path):
        os.mkdir(to_dir_path)

    for file in os.listdir(from_dir_path):
        if file.endswith(".wav"):
            print("VADing file:",file)
            wav_to_vad(os.path.join(from_dir_path,file),os.path.join(to_dir_path,file))
            print("Done")


def get_mfcc_from_file(file_name, mfcc_dir=None, config_name="vbdiar.yml"):
    if mfcc_dir is not None:
        if not os.path.isdir(mfcc_dir):
            os.mkdir(mfcc_dir)

    config = Utils.read_config(config_name)
    config_mfcc = config['MFCC']
    config_path = os.path.abspath(config_mfcc['config_path'])
    if not os.path.isfile(config_path):
        raise ValueError('Path to MFCC configuration `{}` not found.'.format(config_path))
    features_extractor = KaldiMFCCFeatureExtraction(
        config_path=config_path, apply_cmvn_sliding=config_mfcc['apply_cmvn_sliding'],
        norm_vars=config_mfcc['norm_vars'], center=config_mfcc['center'], cmn_window=config_mfcc['cmn_window'])

    pp=os.path.abspath(file_name)
    tempfile, features = features_extractor.audio2features(os.path.abspath(file_name))
    os.remove(tempfile)

    if mfcc_dir is not None:
        hf = h5py.File(os.path.join(mfcc_dir, "{}.{}".format(file_name.split('/')[-1], "h5")), 'w')
        hf.create_dataset('mfcc', data=features)
        hf.close()

    return features


def mfcc_all_files(to_dir_path, from_dir_path):
    if not os.path.exists(to_dir_path):
        os.mkdir(to_dir_path)

    for file in os.listdir(from_dir_path):
        if file.endswith(".wav"):
            print("Get MFCC from file:", file)
            get_mfcc_from_file(os.path.join(from_dir_path,file),mfcc_dir=to_dir_path)
            print("Done")


def get_embeddings_from_features(features, config_name="vbdiar.yml"):
    config = Utils.read_config(config_name)
    config_embedding_extractor = config['EmbeddingExtractor']
    embedding_extractor = KaldiXVectorExtraction(
        nnet=os.path.abspath(config_embedding_extractor['nnet']), use_gpu=config_embedding_extractor['use_gpu'],
        min_chunk_size=config_embedding_extractor['min_chunk_size'],
        chunk_size=config_embedding_extractor['chunk_size'],
        cache_capacity=config_embedding_extractor['cache_capacity'])

    feat_dict={1:features}
    emb_file, featur_file, embeddings = embedding_extractor.features2embeddings(feat_dict)

    os.remove(emb_file.name)
    os.remove(featur_file.name)

    return embeddings['1']


def get_embeddings_from_file(features_file):
    hf = h5py.File(features_file, 'r')
    features=np.array(hf.get("mfcc"))
    return get_embeddings_from_features(features)


def embedding_all_file(to_dir_path, from_dir_path):
    if not os.path.exists(to_dir_path):
        os.mkdir(to_dir_path)
    for file in os.listdir(from_dir_path):
        if file.endswith(".wav.h5"):
            print("Get embedding from file:", file)
            emb=get_embeddings_from_file(os.path.join(from_dir_path,file))
            hf = h5py.File(os.path.join(to_dir_path, file.split('/')[-1]), 'w')
            hf.create_dataset('embeddings', data=emb)
            hf.close()
            print("Done")

def plda():
    hf = h5py.File("embs/fe_03_00001-a.wav.h5", 'r')
    embs = np.array(hf.get("embeddings"))

    classifier = Classifier()
    classifier.fit_model()
    U_model = classifier.model.transform(embs, from_space='D', to_space='U_model')


    pass

if __name__ == '__main__':
    #vad_all_files(from_dir_path="wav", to_dir_path="vad")
    #mfcc_all_files(from_dir_path="vad", to_dir_path="MFCCs")
    #embedding_all_file(from_dir_path="MFCCs", to_dir_path="embs")

    plda()

    pass
