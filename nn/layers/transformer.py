import numpy as np
from ._base import Module, Sequential
from .basic import Linear, LayerNorm, Embedding
from .activations import ReLU, Softmax
from .attention import MultiHeadAttention, SelfAttention


def generate_tril_mask(shape):
    return np.zeros(shape) + np.tril(np.ones(shape[-2:])) # broadcast

def generate_pos_encodings(shape):
    E = np.zeros(shape[-2:], dtype=float)
    d_model = E.shape[-1]
    positions = np.arange(E.shape[0], dtype=float).reshape(-1, 1)
    sin_chan = np.arange(np.ceil(d_model / 2), dtype=float)
    cos_chan = np.arange(np.floor(d_model / 2), dtype=float)
    E[:, 0::2] = np.sin(positions / (100000 ** (2 * sin_chan / d_model)))
    E[:, 1::2] = np.cos(positions / (100000 ** (2 * cos_chan / d_model)))
    return np.zeros(shape) + E # broadcast


class SubBlock(Module):

    def __init__(self, sublayer):
        super().__init__()
        self.sublayer = sublayer
        self.norm = LayerNorm()

    def forward(self, input):
        x = input
        x = self.sublayer(x)
        x = x + input
        return self.norm(x)
    
    def backward(self, next):
        d = self.norm.backward(next)
        d_sb = self.sublayer.backward(d)
        return d + d_sb
    

class CrossSubBlock(Module):

    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.norm = LayerNorm()

    def forward(self, input):
        x_e, x_d = input
        x = self.attention([x_e, x_e, x_d])
        x = x + x_d
        return self.norm(x)
    
    def backward(self, next):
        d = self.norm.backward(next)
        ds = self.attention.backward(d)
        d_e, d_d = np.sum(ds[:-1], axis=0), ds[-1]
        return d_e, (d_d + d)
    

class FeedForward(Sequential):

    def __init__(self, d_model, d_ff):
        super().__init__([
            Linear(d_model, d_ff),
            ReLU(),
            Linear(d_ff, d_model, bias=False)
        ])


class EncoderBlock(Sequential):
    
    def __init__(self, n_heads, d_model, d_hidden, d_ff):
        super().__init__([
            SubBlock(
                SelfAttention(n_heads, d_model, d_hidden)
            ),
            SubBlock(
                FeedForward(d_model, d_ff)
            )
        ])


class DecoderBlock(Module):
    
    def __init__(self, n_heads, d_model, d_hidden, d_ff):
        super().__init__()
        self.sb1 = SubBlock(SelfAttention(n_heads, d_model, d_hidden, generate_tril_mask))
        self.sb2 = CrossSubBlock(MultiHeadAttention(n_heads, d_model, d_hidden))
        self.sb3 = SubBlock(FeedForward(d_model, d_ff))

    def forward(self, input):
        memory, x = input
        x = self.sb1(x)
        x = self.sb2([memory, x])
        return self.sb3(x)

    def backward(self, next):
        d = self.sb3.backward(next)
        d_mem, d = self.sb2.backward(d)
        d = self.sb1.backward(d)
        return d_mem, d


class Encoder(Sequential):
    
    def __init__(self, n_layers, n_heads, d_model, d_hidden, d_ff):
        super().__init__([
            EncoderBlock(n_heads, d_model, d_hidden, d_ff)
            for _ in range(n_layers)
        ])


class Decoder(Module):
    
    def __init__(self, n_layers, n_heads, d_model, d_hidden, d_ff):
        super().__init__()
        self.blocks = [
            DecoderBlock(n_heads, d_model, d_hidden, d_ff) 
            for _ in range(n_layers)
        ]

    def forward(self, input): # input: [memory, decoder]
        memory, x = input
        for block in self.blocks:
            x = block([memory, x])
        return x
    
    def backward(self, next):
        d = next
        d_mems = []
        for block in reversed(self.blocks):
            d_mem, d = block.backward(d)
            d_mems.append(d_mem)
        return np.sum(d_mems, axis=0), d


class Transformer(Module):
    
    def __init__(self, vocab_size, n_layers, n_heads, d_model, d_hidden, d_ff):
        super().__init__()
        self.encoder_emb = Embedding(vocab_size, d_model)
        self.decoder_emb = Embedding(vocab_size, d_model)
        self.encoder = Encoder(n_layers, n_heads, d_model, d_hidden, d_ff)
        self.decoder = Decoder(n_layers, n_heads, d_model, d_hidden, d_ff)
        self.lin = Linear(d_model, vocab_size)
        self.sm = Softmax()
        self.pos_encoding = generate_pos_encodings

    def forward(self, input):
        enc_tokens, dec_tokens = input

        # encoder pass
        x_enc = self.encoder_emb(enc_tokens)
        x_enc = x_enc + self.pos_encoding(x_enc.shape)
        memory = self.encoder(x_enc)

        # decoder pass
        x_dec = self.decoder_emb(dec_tokens)
        x_dec = x_dec + self.pos_encoding(x_dec.shape)
        x = self.decoder([memory, x_dec])
        x = self.lin(x)
        return self.sm(x)

    def backward(self, next):
        d = self.sm.backward(next)
        d = self.lin.backward(d)
        d_enc, d_dec = self.decoder.backward(d)
        d_enc = self.encoder.backward(d_enc)
        self.decoder_emb.backward(d_dec)
        self.encoder_emb.backward(d_enc)
