
import logging
import os
import pickle


import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CDIR = os.path.dirname(os.path.realpath(__file__))

def extract_embeddings(features_dict, embedding_extractor):
    """ Extract embeddings from multiple segments.

    Args:
        features_dict (Dict): dictionary with segment range as key and features as values
        embedding_extractor (Any):

    Returns:
        EmbeddingSet: extracted embedding in embedding set
    """
    embedding_set = EmbeddingSet()
    emb_file, featur_file, embeddings = embedding_extractor.features2embeddings(features_dict)

    os.remove(emb_file.name)
    os.remove(featur_file.name)

    for embedding_key in embeddings:
        start, end = embedding_key.split('_')
        embedding_set.add(embeddings[embedding_key], window_start=int(float(start)), window_end=int(float(end)))
    return embedding_set


class Embedding(object):
    """ Class for basic i-vector operations.

    """

    def __init__(self):
        """ Class constructor.

        """
        self.data = None
        self.features = None
        self.window_start = None
        self.window_end = None


class EmbeddingSet(object):
    """ Class for encapsulating ivectors set.

    """

    def __init__(self):
        """ Class constructor.

        """
        self.name = None
        self.num_speakers = None
        self.embeddings = []

    def __iter__(self):
        current = 0
        while current < len(self.embeddings):
            yield self.embeddings[current]
            current += 1

    def __getitem__(self, key):
        return self.embeddings[key]

    def __setitem__(self, key, value):
        self.embeddings[key] = value

    def __len__(self):
        return len(self.embeddings)

    def get_all_embeddings(self):
        """ Get all ivectors.

        """
        a = []
        for i in self.embeddings:
            a.append(i.data.flatten())
        return np.array(a)

    def get_longer_embeddings(self, min_length):
        """ Get i-vectors extracted from longer segments than minimal length.

        Args:
            min_length (int): minimal length of segment in miliseconds

        Returns:
            np.array: i-vectors
        """
        a = []
        for embedding in self.embeddings:
            if embedding.window_end - embedding.window_start >= min_length:
                a.append(embedding.data.flatten())
        return np.array(a)

    def add(self, data, window_start, window_end, features=None):
        """ Add embedding to set.

        Args:
            data (np.array): embeding data
            window_start (int): start of the window [ms]
            window_end (int): end of the window [ms]
            features (np.array): features from which embedding was extracted
        """
        i = Embedding()
        i.data = data
        i.window_start = window_start
        i.window_end = window_end
        i.features = features
        self.__append(i)

    def __append(self, embedding):
        """ Append embedding to set of embedding.

        Args:
            embedding (Embedding):
        """
        ii = 0
        for vp in self.embeddings:
            if vp.window_start > embedding.window_start:
                break
            ii += 1
        self.embeddings.insert(ii, embedding)

    def save(self, path):
        """ Save embedding set as pickled file.

        Args:
            path (string_types): output path
        """
        mkdir_p(os.path.dirname(path))
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def get_vad(file_name, fea_len):
    """ Load .lab file as bool vector.

    Args:
        file_name (str): path to .lab file
        fea_len (int): length of features

    Returns:
        np.array: bool vector
    """

    logger.info('Loading VAD from file `{}`.'.format(file_name))


    return load_vad_lab_as_bool_vec(file_name)[:fea_len]


def load_vad_lab_as_bool_vec(lab_file):
    """

    Args:
        lab_file:

    Returns:

    """
    lab_cont = np.atleast_2d(np.loadtxt(lab_file, dtype=object))

    if lab_cont.shape[1] == 0:
        return np.empty(0), 0, 0

    if lab_cont.shape[1] == 3:
        lab_cont = lab_cont[lab_cont[:, 2] == 'sp', :][:, [0, 1]]

    n_regions = lab_cont.shape[0]
    ii = 0
    while True:
        try:
            start1, end1 = float(lab_cont[ii][0]), float(lab_cont[ii][1])
            jj = ii + 1
            start2, end2 = float(lab_cont[jj][0]), float(lab_cont[jj][1])
            if end1 >= start2:
                lab_cont = np.delete(lab_cont, ii, axis=0)
                ii -= 1
                lab_cont[jj - 1][0] = str(start1)
                lab_cont[jj - 1][1] = str(max(end1, end2))
            ii += 1
        except IndexError:
            break

    vad = np.round(np.atleast_2d(lab_cont).astype(np.float).T * 100).astype(np.int)
    vad[1] += 1  # Paja's bug!!!

    if not vad.size:
        return np.empty(0, dtype=bool)

    npc1 = np.c_[np.zeros_like(vad[0], dtype=bool), np.ones_like(vad[0], dtype=bool)]
    npc2 = np.c_[vad[0] - np.r_[0, vad[1, :-1]], vad[1] - vad[0]]

    out = np.repeat(npc1, npc2.flat)

    n_frames = sum(out)

    return out, n_regions, n_frames



def process_file(wav_dir, vad_dir, out_dir, file_name, features_extractor, embedding_extractor,
                 max_size, tolerance, wav_suffix='.wav', vad_suffix='.lab.gz'):
    """ Process single audio file.

    Args:
        wav_dir (str): directory with wav files
        vad_dir (str): directory with vad files
        out_dir (str): output directory
        file_name (str): name of the file
        features_extractor (Any): intialized object for feature extraction
        embedding_extractor (Any): initialized object for embedding extraction
        max_size (int): maximal size of window in ms
        tolerance (int): accept given number of frames as speech even when it is marked as silence
        wav_suffix (str): suffix of wav files
        vad_suffix (str): suffix of vad files

    Returns:
        EmbeddingSet
    """
    logger.info('Processing file {}.'.format(file_name.split()[0]))
    num_speakers = None
    if len(file_name.split()) > 1:  # number of speakers is defined
        file_name, num_speakers = file_name.split()[0], int(file_name.split()[1])

    wav_dir, vad_dir = os.path.abspath(wav_dir), os.path.abspath(vad_dir)
    if out_dir:
        out_dir = os.path.abspath(out_dir)

    # extract features
    tempfile, features = features_extractor.audio2features(os.path.join(wav_dir, '{}{}'.format(file_name, wav_suffix)))
    os.remove(tempfile)

    # load voice activity detection from file
    vad, _, _ = get_vad('{}{}'.format(os.path.join(vad_dir, file_name), vad_suffix), features.shape[0])

    # parse segments and split features
    features_dict = {}
    for seg in get_segments(vad, max_size, tolerance):
        start, end = get_num_segments(seg[0]), get_num_segments(seg[1])
        if seg[0] > features.shape[0] - 1 or seg[1] > features.shape[0] - 1:
            raise ValueError('Unexpected features dimensionality - check VAD input or audio.')
        features_dict['{}_{}'.format(start, end)] = features[seg[0]:seg[1]]

    # extract embedding for each segment

    embedding_set = extract_embeddings(features_dict, embedding_extractor)
    embedding_set.name = file_name
    embedding_set.num_speakers = num_speakers


    return embedding_set

RATE = 8000
SOURCERATE = 1250
TARGETRATE = 100000

LOFREQ = 120
HIFREQ = 3800

ZMEANSOURCE = True
WINDOWSIZE = 250000.0
USEHAMMING = True
PREEMCOEF = 0.97
NUMCHANS = 24
CEPLIFTER = 22
NUMCEPS = 19
ADDDITHER = 1.0
RAWENERGY = True
ENORMALISE = True

deltawindow = accwindow = 2

cmvn_lc = 150
cmvn_rc = 150

fs = 1e7 / SOURCERATE


def get_segments(vad, max_size, tolerance):
    """ Return clustered speech segments.

        :param vad: list with labels - voice activity detection
        :type vad: list
        :param max_size: maximal size of window in ms
        :type max_size: int
        :param tolerance: accept given number of frames as speech even when it is marked as silence
        :type tolerance: int
        :returns: clustered segments
        :rtype: list
    """
    clusters = get_clusters(vad, tolerance)
    segments = []
    max_frames = get_num_frames(max_size)
    for item in clusters.values():
        if item[1] - item[0] > max_frames:
            for ss in split_segment(item, max_frames):
                segments.append(ss)
        else:
            segments.append(item)
    return segments


def split_segment(segment, max_size):
    """ Split segment to more with adaptive size.

        :param segment: input segment
        :type segment: tuple
        :param max_size: maximal size of window in ms
        :type max_size: int
        :returns: splitted segment
        :rtype: list
    """
    size = segment[1] - segment[0]
    num_segments = int(np.math.ceil(size / max_size))
    size_segment = size / num_segments
    for ii in range(num_segments):
        yield (segment[0] + ii * size_segment, segment[0] + (ii + 1) * size_segment)


def get_num_frames(n):
    """ Get number of frames from ms.

        :param n: number of ms
        :type n: int
        :returns: number of frames
        :rtype: int

        >>> get_num_frames(25)
        1
        >>> get_num_frames(35)
        2
    """
    assert n >= 0, 'Time must be at least equal to 0.'
    if n < 25:
        return 0
    return int(1 + (n - WINDOWSIZE / 10000) / (TARGETRATE / 10000))


def get_num_segments(n):
    """ Get count of ms from number of frames.

        :param n: number of frames
        :type n: int
        :returns: number of ms
        :rtype: int

        >>> get_num_segments(1)
        25
        >>> get_num_segments(2)
        35

    """
    return int(n * (TARGETRATE / 10000) - (TARGETRATE / 10000) + (WINDOWSIZE / 10000))


def get_clusters(vad, tolerance=10):
    """ Cluster speech segments.

        :param vad: list with labels - voice activity detection
        :type vad: list
        :param tolerance: accept given number of frames as speech even when it is marked as silence
        :type tolerance: int
        :returns: clustered speech segments
        :rtype: dict
    """
    num_prev = 0
    in_tolerance = 0
    num_clusters = 0
    clusters = {}
    for ii, frame in enumerate(vad):
        if frame:
            num_prev += 1
        else:
            in_tolerance += 1
            if in_tolerance > tolerance:
                if num_prev > 0:
                    clusters[num_clusters] = (ii - num_prev, ii)
                    num_clusters += 1
                num_prev = 0
                in_tolerance = 0
    clusters[num_clusters] = (ii - num_prev, ii) #uncoment this, if you want to take last part, idk
    return clusters


def split_seq(seq, size):
    """ Split up seq in pieces of size.

    Args:
        seq:
        size:

    Returns:

    """
    return [seq[i:i + size] for i in range(0, len(seq), size)]

