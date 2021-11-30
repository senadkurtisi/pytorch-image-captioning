import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer, TransformerDecoder


class PositionalEncodings(nn.Module):
    """Attention is All You Need positional encoding layer"""

    def __init__(self, seq_len, d_model, p_dropout):
        """Initializes the layer."""
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=seq_len).view(-1, 1)
        dim_positions = torch.arange(start=0, end=d_model).view(1, -1)
        angles = token_positions / (10000 ** ((2 * dim_positions) / d_model))

        encodings = torch.zeros(1, seq_len, d_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Performs forward pass of the module."""
        x = x + self.positional_encodings
        x = self.dropout(x)
        return x


class CaptionDecoder(nn.Module):
    """Decoder for image captions.

    Generates prediction for next caption word given the prviously
    generated word and image features extracted from CNN.    
    """

    def __init__(self, config):
        """Initializes the model."""
        super(CaptionDecoder, self).__init__()
        decoder_layers = config["decoder_layers"]
        attention_heads = config["attention_heads"]
        d_model = config["d_model"]
        ff_dim = config["ff_dim"]
        dropout = config["dropout"]

        embedding_dim = config["embeddings"]["size"]
        vocab_size = config["vocab_size"]
        img_feature_channels = config["image_specs"]["img_feature_channels"]

        # Load pretrained word embeddings
        word_embeddings = torch.Tensor(np.loadtxt(config["embeddings"]["path"]))
        self.embedding_layer = nn.Embedding.from_pretrained(
            word_embeddings,
            freeze=True,
            padding_idx=config["PAD_idx"]
        )

        # Modules used for mapping image features and word tokens to transformer embedding dimension
        self.entry_mapping_words = nn.Linear(embedding_dim, d_model)
        self.entry_mapping_img = nn.Linear(img_feature_channels, d_model)

        self.positional_encodings = PositionalEncodings(config["max_len"], d_model, dropout)
        transformer_decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=ff_dim,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(transformer_decoder_layer, decoder_layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, image_features, padd_mask=None):
        """Performs forward pass of the module.

        Arguments:
            x: Input word tokens - previously generated words
            image_features: Features acquired by encoder CNN for
                image for which we generate caption
            padd_mask: Mask for ignoring padding tokens in @x during attention        
        """
        # Adapt the dimensionality of the features for image patches
        image_features = self.entry_mapping_img(image_features)
        image_features = image_features.permute(1, 0, 2)

        # Entry mapping for word tokens
        x = self.embedding_layer(x)
        x = self.entry_mapping_words(x)
        x = self.positional_encodings(x)

        # Get output from the decoder
        x = x.permute(1, 0, 2)
        x = self.decoder(tgt=x, memory=image_features, tgt_key_padding_mask=padd_mask)
        x = x.permute(1, 0, 2)

        x = self.classifier(x)
        return x
