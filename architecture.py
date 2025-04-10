from imports import *
from dictionary import PHONEMES
import config


class Permute(torch.nn.Module):
    '''
    Used to transpose/permute the dimensions of an MFCC tensor.
    '''
    def forward(self, x):
        return x.transpose(1, 2)
    


class pBLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()

        self.blstm = nn.LSTM(
            input_size=input_dim * 2,  
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, packed_input):
        packed_input, input_lens = pad_packed_sequence(packed_input, batch_first=True)

        packed_input, input_lens = self.trunc_reshape(packed_input, input_lens)

        packed_input = pack_padded_sequence(packed_input, input_lens, batch_first=True, enforce_sorted=False)

        packed_output, _ = self.blstm(packed_input)

        return packed_output

    def trunc_reshape(self, x, x_lens):
        batch_size, seq_len, feat_dim = x.size()

        if seq_len % 2 != 0:
            x = x[:, :-1, :]
            seq_len = seq_len - 1

        x = x.contiguous().view(batch_size, seq_len // 2, feat_dim * 2)

        x_lens = torch.div(x_lens, 2, rounding_mode='floor')

        return x, x_lens

class LSTMWrapper(torch.nn.Module):
    '''
    Used to get only output of lstm, not the hidden states.
    '''
    def __init__(self, lstm):
        super(LSTMWrapper, self).__init__()
        self.lstm = lstm

    def forward(self, x):
        output, _ = self.lstm(x)
        return output
    


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.skip_conn = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0) if in_ch != out_ch else nn.Identity()

    def forward(self, x_in):
        out = self.conv1(x_in)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        skip = self.skip_conn(x_in)

        out += skip
        out = self.relu(out)

        return out
    

class LockedDropout(nn.Module):
    
    def __init__(self, dropout_rate=0.2):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x_packed_seq):
        if not self.training or self.dropout_rate == 0:
            return x_packed_seq

        x_data, seq_lengths = nn.utils.rnn.pad_packed_sequence(x_packed_seq, batch_first=True)

        batch_sz, seq_len, hidden_dim = x_data.size()
        mask = x_data.new_empty(batch_sz, 1, hidden_dim).bernoulli_(1 - self.dropout_rate) / (1 - self.dropout_rate)
        mask = mask.expand(-1, seq_len, -1)

        x_data = x_data * mask

        x_packed_seq = nn.utils.rnn.pack_padded_sequence(
            x_data, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        return x_packed_seq


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, enc_hidden_dim):
        super(Encoder, self).__init__()

        self.embedding_layer = nn.Sequential(
            Permute(),
            Block(input_dim, 128),
            Block(128, 256),
            Block(256, 512),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            Block(512, 256),

            nn.Dropout(config['encoder_dropout']),
            Permute()
        )
        

        self.bi_lstm = LSTMWrapper(
            torch.nn.LSTM(
                input_size=config['embed_size'],
                hidden_size=enc_hidden_dim,
                num_layers=2,
                bidirectional=True,
                dropout=0.3,  
                batch_first=True
            )
        )

        self.pyramidal_blstms = torch.nn.Sequential(
            pBLSTM(input_size=2*enc_hidden_dim, hidden_size=enc_hidden_dim),
            LockedDropout(dropout=0.3),
            pBLSTM(input_size=2*enc_hidden_dim, hidden_size=enc_hidden_dim),
            LockedDropout(dropout=0.3)
        )

    def forward(self, input_data, input_lengths):
        # CNN feature extraction
        input_data = self.embedding_layer(input_data)

        packed_input = nn.utils.rnn.pack_padded_sequence(
            input_data, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        blstm_output = self.bi_lstm(packed_input)

        pyramidal_blstm_output = self.pyramidal_blstms(blstm_output)

        
        encoder_outputs, encoder_lengths = nn.utils.rnn.pad_packed_sequence(
            pyramidal_blstm_output, batch_first=True
        )

        return encoder_outputs, encoder_lengths



class Decoder(torch.nn.Module):

    def __init__(self, embed_size, output_size=41):
        super(Decoder, self).__init__()

        self.mlp = torch.nn.Sequential(
            Permute(),
            torch.nn.BatchNorm1d(2 * embed_size),
            Permute(),
            torch.nn.Linear(2 * embed_size, 3072),
            torch.nn.ReLU(),
            Permute(),
            torch.nn.BatchNorm1d(3072),
            Permute(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(3072, 1024),
            torch.nn.ReLU(),
            Permute(),
            torch.nn.BatchNorm1d(1024),
            Permute(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 768),
            torch.nn.ReLU(),
            Permute(),
            torch.nn.BatchNorm1d(768),
            Permute(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            Permute(),
            torch.nn.BatchNorm1d(512),
            Permute(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 360),
            torch.nn.ReLU(),
            Permute(),
            torch.nn.BatchNorm1d(360),
            Permute(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(360, 256),
            torch.nn.ReLU(),
            Permute(),
            torch.nn.BatchNorm1d(256),
            Permute(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, output_size)
        )

        self.softmax = torch.nn.LogSoftmax(dim=2)


    def forward(self, encoder_out):
        logits = self.mlp(encoder_out)
        out = self.softmax(logits)
        return out



class ASRModel(torch.nn.Module):

    def __init__(self, input_size, embed_size=192, output_size=len(PHONEMES)):
        super().__init__()

        # Initialize encoder and decoder
        self.encoder = Encoder(input_size=input_size, encoder_hidden_size=embed_size)
        self.decoder = Decoder(embed_size=embed_size, output_size=output_size)

    def forward(self, x, lengths_x):

        encoder_out, encoder_lens = self.encoder(x, lengths_x)
        decoder_out = self.decoder(encoder_out)

        return decoder_out, encoder_lens
