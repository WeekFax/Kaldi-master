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


def get_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr = 16000)
    mfcc=[]
    while audio.shape[0]>=(sr*3):
        audio_frame=audio[:(sr*3)]
        mfcc.append(librosa.feature.mfcc(y=audio_frame, sr = sr, n_mfcc=23).T)
        audio=audio[int(sr*3/2):]
    audio_frame=np.zeros(sr*3,dtype=np.float32)
    audio_frame[:audio.shape[0]]=audio
    mfcc.append(librosa.feature.mfcc(y=audio_frame, sr=sr, n_mfcc=23).T)
    return mfcc


def compute_eer(probabilities, target, n=10000):
    start_k = probabilities.min()
    stop_k = probabilities.max()
    proba_space = np.linspace(start_k, stop_k, n)

    human_score = probabilities[np.logical_not(target == 0)]
    spoof_score = probabilities[target == 0]

    n_human = human_score.size
    n_spoof = spoof_score.size

    frr = np.empty(n)
    far = np.empty(n)
    n_eer = n // 2
    eer = 1.0

    min_gap = np.inf
    for m, proba in enumerate(proba_space):
        frr[m] = len(np.where(human_score >= proba)[0]) / n_human
        far[m] = len(np.where(spoof_score < proba)[0]) / n_spoof

        gap = np.abs(far[m] - frr[m])

        if gap < min_gap:
            min_gap = gap
            n_eer = m
            eer = (far[m] + frr[m]) / 2

    return (n_eer, eer), frr, far, proba_space


if __name__ == '__main__':

    """
    files = [line.rstrip('\n') for line in open("ASVspoof2017_dev.trl.txt")]
    for file in files:
        file=file.split(' ')[0]
        mfccs = get_mfcc(os.path.join("../ASVspoof2017_dev",file))
        emb=[]
        for mfcc in mfccs:
            emb.append(get_embeddings_from_features(mfcc, config_name="vbdiar.yml"))
        xv=np.array(emb)
        xv=np.mean(xv,axis=0)
        hf = h5py.File(os.path.join("embs/dev","{}.{}".format(file,"h5")))
        hf.create_dataset('xv',data=xv)
        hf.close()
        print(file)
    """
    classifier=Classifier()

    train_data=[]
    train_label=[]

    files = [line.rstrip('\n') for line in open("ASVspoof2017_train.trn.txt")]
    for file in files:
        file_sp=file.split(' ')
        file_name=file_sp[0]
        file_label=file_sp[1]
        file_M=file_sp[2]
        file_S=file_sp[3]

        hf=h5py.File(os.path.join("embs/train","{}.{}".format(file_name,"h5")),'r')
        xv=np.array(hf.get("xv"))
        hf.close()

        train_data.append(xv)
        if(file_label=="spoof"):
            train_label.append(0)
        if (file_label == "genuine"):
            train_label.append(1)

    train_label=np.array(train_label)
    train_data=np.array(train_data)


    test_data=[]
    test_label=[]
    files = [line.rstrip('\n') for line in open("ASVspoof2017_dev.trl.txt")]
    for file in files:
        file_sp = file.split(' ')
        file_name = file_sp[0]
        file_label = file_sp[1]
        file_M = file_sp[2]
        file_S = file_sp[3]

        hf = h5py.File(os.path.join("embs/dev", "{}.{}".format(file_name, "h5")), 'r')
        xv = np.array(hf.get("xv"))
        hf.close()

        test_data.append(xv)
        if (file_label == "spoof"):
            test_label.append(0)
        if (file_label == "genuine"):
            test_label.append(1)

    test_label = np.array(test_label)
    test_data = np.array(test_data)

    classifier.fit_model(train_data,train_label,n_principal_components=90)

    predictions, log_p_predictions = classifier.predict(test_data)
    (n_eer, eer), frr, far, proba_space=compute_eer(log_p_predictions,test_label)

    print('Accuracy: {}'.format((test_label == predictions).mean()))
    print('Eer {}'.format(eer))

    pass
