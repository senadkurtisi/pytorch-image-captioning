import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer, TransformerDecoder


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

        # Load pretrained word embeddings
        word_embeddings = torch.Tensor(np.loadtxt(config["embeddings"]["path"]))
        self.embedding_layer = nn.Embedding.from_pretrained(
            word_embeddings,
            freeze=True,
            padding_idx=config["PAD_idx"]
        )

        self.entry_mapping = nn.Linear(embedding_dim, d_model)

        transformer_decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=ff_dim,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(transformer_decoder_layer, decoder_layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, image_features, target_pos, padd_mask=None):
        """Performs forward pass of the module."""
        x = self.embedding_layer(x)
        x = self.entry_mapping(x)
        # Get output from the decoder
        x = x.permute(1, 0, 2)
        x = self.decoder(tgt=x, memory=image_features, tgt_key_padding_mask=padd_mask)
        x = x.permute(1, 0, 2)

        # Extract the prediction of relevant token
        x = x[:, target_pos, :]
        x = self.classifier(x)
        return x
