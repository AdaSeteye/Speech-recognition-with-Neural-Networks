from imports import *
from dictionary import PHONEMES
import config


BATCH_SIZE = config["batch_size"] 
root = "/kaggle/working/" 
path = "/kaggle/input/dataset/"

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, partition='train-clean-100', augment=config['augment']):
        self.PHONEMES = PHONEMES
        self.subset = config['subset']
        self.time_mask = tat.TimeMasking(time_mask_param=config['time_mask_param'])
        self.freq_mask = tat.FrequencyMasking(freq_mask_param=config['freq_mask_param'])
        self.augment = augment
        self.mfcc_folder = os.path.join(path, partition, 'mfcc')
        self.transcript_folder = os.path.join(path, partition, 'transcript')

        self.mfcc_files = sorted(os.listdir(path=self.mfcc_folder))
        self.transcript_files = sorted(os.listdir(path=self.transcript_folder))

        dataset_size = int(self.data_ratio * len(self.mfcc_files))

        self.mfcc_files = self.mfcc_files[:dataset_size]
        self.transcript_files = self.transcript_files[:dataset_size]

        self.dataset_length = dataset_size

        self.normalized_mfccs, self.transcript_labels = [], []

        for idx in tqdm(range(self.dataset_length)):
            mfcc_data = np.load(os.path.join(self.mfcc_folder, self.mfcc_files[idx]))
            mfcc_mean = np.mean(mfcc_data, axis=0)
            mfcc_std = np.std(mfcc_data, axis=0)
            mfcc_normalized = (mfcc_data - mfcc_mean) / (mfcc_std + 1e-8)

            transcript_data = np.load(os.path.join(self.transcript_folder, self.transcript_files[idx]))[1:-1]
            transcript_indices = [self.PHONEME_LIST.index(t) for t in transcript_data]
            transcript_indices = torch.tensor(transcript_indices, dtype=torch.int64)

            mfcc_normalized = torch.tensor(mfcc_normalized, dtype=torch.float32)

            self.normalized_mfccs.append(mfcc_normalized)
            self.transcript_labels.append(transcript_indices)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        mfcc_sample = self.normalized_mfccs[index]
        transcript_sample = self.transcript_labels[index]

        return mfcc_sample, transcript_sample

    def collate_fn(self, batch):
        batch_mfcc, batch_transcript = zip(*batch)

        mfcc_lengths = torch.tensor([len(mfcc) for mfcc in batch_mfcc])
        transcript_lengths = torch.tensor([len(transcript) for transcript in batch_transcript])

        padded_mfcc = pad_sequence(batch_mfcc, batch_first=True, padding_value=0)
        padded_transcript = pad_sequence(batch_transcript, batch_first=True, padding_value=0)

        if self.augment:
            padded_mfcc = padded_mfcc.transpose(1, 2)
            padded_mfcc = self.freq_mask(padded_mfcc)
            padded_mfcc = self.time_mask(padded_mfcc)
            padded_mfcc = padded_mfcc.transpose(1, 2)

        return padded_mfcc, padded_transcript, torch.tensor(mfcc_lengths), torch.tensor(transcript_lengths)
    

    

class AudioDatasetTest(torch.utils.data.Dataset):

    def __init__(self):
        self.PHONEMES = PHONEMES
        self.subset = config['subset']

        self.mfcc_folder = os.path.join(path, 'test-clean', 'mfcc')
        self.mfcc_files = sorted(os.listdir(path=self.mfcc_folder))

        self.dataset_size = len(self.mfcc_files)

        self.mfcc_data = []

        for idx in tqdm(range(self.dataset_size)):
            mfcc_sample = np.load(os.path.join(path, 'test-clean', 'mfcc', self.mfcc_files[idx]))
            mfcc_mean = np.mean(mfcc_sample, axis=0)
            mfcc_std = np.std(mfcc_sample, axis=0)
            mfcc_normalized = (mfcc_sample - mfcc_mean) / (mfcc_std + 1e-8)

            mfcc_normalized = torch.tensor(mfcc_normalized, dtype=torch.float32)

            self.mfcc_data.append(mfcc_normalized)

        self.dataset_length = len(self.mfcc_data)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        mfcc_sample = self.mfcc_data[index]
        return mfcc_sample

    def collate_fn(self, batch):
        batch_mfcc = batch

        padded_mfcc = pad_sequence(batch_mfcc, batch_first=True, padding_value=0)
        mfcc_lengths = torch.tensor([len(mfcc) for mfcc in batch_mfcc])

        return padded_mfcc, torch.tensor(mfcc_lengths)
