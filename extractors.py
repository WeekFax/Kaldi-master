import os
import logging
import tempfile
import subprocess


KALDI_ROOT_PATH = '/home/weekfax/kaldi/'

bin_path = os.path.join(KALDI_ROOT_PATH, 'src', 'bin')
featbin_path = os.path.join(KALDI_ROOT_PATH, 'src', 'featbin')
nnet3bin_path = os.path.join(KALDI_ROOT_PATH, 'src', 'nnet3bin')


logger = logging.getLogger(__name__)

from utils import write_txt_matrix
from utils import read_txt_matrix
from utils import read_txt_vectors


class KaldiXVectorExtraction(object):

    def __init__(self, nnet, binary_path=nnet3bin_path, use_gpu=False,
                 min_chunk_size=25, chunk_size=10000, cache_capacity=64):
        """ Initialize Kaldi x-vector extractor.

        Args:
            nnet (string_types): path to neural net
            use_gpu (bool):
            min_chunk_size (int):
            chunk_size (int):
            cache_capacity (int):
        """
        self.nnet3_xvector_compute = os.path.join(binary_path, 'nnet3-xvector-compute')
        if not os.path.exists(self.nnet3_xvector_compute):
            raise ValueError(
                'Path to nnet3-xvector-compute - `{}` does not exists.'.format(self.nnet3_xvector_compute))
        self.nnet3_copy = os.path.join(binary_path, 'nnet3-copy')
        if not os.path.exists(self.nnet3_copy):
            raise ValueError(
                'Path to nnet3-copy - `{}` does not exists.'.format(self.nnet3_copy))
        if not os.path.isfile(nnet):
            raise ValueError('Invalid path to nnet `{}`.'.format(nnet))
        else:
            self.nnet = nnet
        self.binary_path = binary_path
        self.use_gpu = use_gpu
        self.min_chunk_size = min_chunk_size
        self.chunk_size = chunk_size
        self.cache_capacity = cache_capacity

    def features2embeddings(self, data_dict):
        """ Extract x-vector embeddings from feature vectors.

        Args:
            data_dict (Dict):

        Returns:

        """
        with tempfile.NamedTemporaryFile(delete=False) as xvec_ark, tempfile.NamedTemporaryFile(delete=False) as mfcc_ark:

            write_txt_matrix(path=mfcc_ark.name, data_dict=data_dict)


            args = [self.nnet3_xvector_compute,
                    '--use-gpu={}'.format('yes' if self.use_gpu else 'no'),
                    '--min-chunk-size={}'.format(str(self.min_chunk_size)),
                    '--chunk-size={}'.format(str(self.chunk_size)),
                    '--cache-capacity={}'.format(str(self.cache_capacity)),
                    self.nnet, 'ark,t:{}'.format(mfcc_ark.name), 'ark,t:{}'.format(xvec_ark.name)]

            logger.info('Extracting x-vectors from {} feature vectors to `{}`.'.format(len(data_dict), xvec_ark.name))
            process = subprocess.Popen(
                args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd=self.binary_path, shell=False)
            _, stderr = process.communicate()
            if process.returncode != 0:
                raise ValueError('`{}` binary returned error code {}.{}{}'.format(
                    self.nnet3_xvector_compute, process.returncode, os.linesep, stderr))

            #os.remove(xvec_ark.name)
            #os.remove(mfcc_ark.name)
            return xvec_ark, mfcc_ark, read_txt_vectors(xvec_ark.name)


class KaldiMFCCFeatureExtraction(object):

    def __init__(self, config_path, binary_path=featbin_path, apply_cmvn_sliding=True,
                 norm_vars=False, center=True, cmn_window=300):
        """ Initialize Kaldi MFCC extraction component. Names of the arguments keep original Kaldi convention.

        Args:
            config_path (string_types): path to config file
            binary_path (string_types): path to directory containing binaries
            apply_cmvn_sliding (bool): apply cepstral mean and variance normalization
            norm_vars (bool): normalize variances
            center (bool): center window
            cmn_window (int): window size
        """
        self.binary_path = binary_path
        self.config_path = config_path
        self.apply_cmvn_sliding = apply_cmvn_sliding
        self.norm_vars = norm_vars
        self.center = center
        self.cmn_window = cmn_window
        self.compute_mfcc_feats_bin = os.path.join(binary_path, 'compute-mfcc-feats')
        if not os.path.exists(self.compute_mfcc_feats_bin):
            raise ValueError('Path to compute-mfcc-feats - {} does not exists.'.format(self.compute_mfcc_feats_bin))
        self.copy_feats_bin = os.path.join(binary_path, 'copy-feats')
        if not os.path.exists(self.copy_feats_bin):
            raise ValueError('Path to copy-feats - {} does not exists.'.format(self.copy_feats_bin))
        self.apply_cmvn_sliding_bin = os.path.join(binary_path, 'apply-cmvn-sliding')
        if not os.path.exists(self.apply_cmvn_sliding_bin):
            raise ValueError('Path to apply-cmvn-sliding - {} does not exists.'.format(self.apply_cmvn_sliding_bin))

    def __str__(self):
        return '<mfcc_config={}>'.format(self.config_path)

    def audio2features(self, input_path):
        """ Extract features from list of files into list of numpy.arrays

        Args:
            input_path (string_types): audio file path

        Returns:
            Tuple[string_types, np.array]: path to Kaldi ark file containing features and features itself
        """
        with tempfile.NamedTemporaryFile() as wav_scp, tempfile.NamedTemporaryFile(delete=False) as mfcc_ark:
            # dump list of file to wav.scp file
            wav_scp.write('{} {}{}'.format(input_path, input_path, os.linesep).encode())
            wav_scp.flush()

            # run fextract
            args = [self.compute_mfcc_feats_bin,
                    '--config={}'.format(self.config_path),
                    'scp:{}'.format(wav_scp.name),
                    'ark{}'.format(',t:{}'.format(mfcc_ark.name) if not self.apply_cmvn_sliding else ':-')]
            logger.info('Extracting MFCC features from `{}`.'.format(input_path))
            compute_mfcc_feats = subprocess.Popen(
                args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd=self.binary_path, shell=False)
            if not self.apply_cmvn_sliding:
                # do not apply cmvn, so just simply compute features
                _, stderr = compute_mfcc_feats.communicate()
                if compute_mfcc_feats.returncode != 0:
                    raise ValueError('`{}` binary returned error code {}.{}{}'.format(
                        self.compute_mfcc_feats_bin, compute_mfcc_feats.returncode, os.linesep, stderr))
            else:
                args2 = [self.apply_cmvn_sliding_bin,
                         '--norm-vars={}'.format(str(self.norm_vars).lower()),
                         '--center={}'.format(str(self.center).lower()),
                         '--cmn-window={}'.format(str(self.cmn_window)),
                         'ark:-', 'ark,t:{}'.format(mfcc_ark.name)]
                apply_cmvn_sliding = subprocess.Popen(args2, stdin=compute_mfcc_feats.stdout,
                                                      stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
                _, stderr = apply_cmvn_sliding.communicate()
                if apply_cmvn_sliding.returncode == 0:
                    pass
                else:
                    raise ValueError('`{}` binary returned error code {}.{}{}'.format(
                        self.apply_cmvn_sliding_bin, compute_mfcc_feats.returncode, os.linesep, stderr))



            return mfcc_ark.name, list(read_txt_matrix(mfcc_ark.name).values())[0]
