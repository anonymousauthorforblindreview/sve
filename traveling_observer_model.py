# Copyright (c) 2021 Cognizant Digital Business, Cognizant AI Labs
# Issued under this Academic Public License: github.com/cognizant-ai-labs/tom-release/LICENSE.

"""
Class for TOM implemented as in Section 4 of https://arxiv.org/pdf/2010.02354.pdf.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import SparseAttention
from core_res_block import CoreResBlock
from film_layer import VEFilmLayer
from film_res_block import FilmResBlock
from inv_film_layer import InvVEFilmLayer



class TravelingObserverModel(nn.Module):

    def __init__(self,
                 context_size,
                 context_std,
                 entmax_bisect_alpha,
                 num_variables,
                 basis_elements,
                 hidden_size,
                 num_encoder_layers,
                 num_core_layers,
                 num_decoder_layers,
                 dropout=0.0):
        super(TravelingObserverModel, self).__init__()


        # Create Encoder
        self.encoder_film_layer = VEFilmLayer(1,
                                              hidden_size,
                                              context_size)
        self.encoder_blocks = nn.ModuleList([])
        for i in range(num_encoder_layers - 1):
            encoder_block = FilmResBlock(context_size,
                                         hidden_size,
                                         dropout)
            self.encoder_blocks.append(encoder_block)

        # Create Core
        self.core_blocks = nn.ModuleList([])
        for i in range(num_core_layers):
            core_block = CoreResBlock(hidden_size,
                                      dropout)
            self.core_blocks.append(core_block)

        # Create Decoder
        self.decoder_blocks = nn.ModuleList([])
        for i in range(num_decoder_layers - 1):
            decoder_block = FilmResBlock(context_size,
                                         hidden_size,
                                         dropout)
            self.decoder_blocks.append(decoder_block)

        self.decoder_film_layer = InvVEFilmLayer(hidden_size,
                                                 1,
                                                 context_size)

        # Create dropout layer
        self.dropout = nn.Dropout(dropout)

        self.dataset_contexts = nn.Embedding(num_variables, context_size)
        self.dataset_contexts.weight.data.normal_(mean=0., std=context_std)

        self.basis_context = nn.Embedding(basis_elements, context_size)
        self.basis_context.weight.data.normal_(mean=0., std=context_std)

        self.attn = SparseAttention()
        self.entmax_bisect_alpha = nn.Parameter(torch.tensor(entmax_bisect_alpha))


    def forward(self, input_batch, var_indices, input_var_indices, output_var_indices):
        contexts = self.dataset_contexts(var_indices)
        input_contexts = contexts[input_var_indices,:].unsqueeze(0).transpose(-2, -1)
        output_contexts = contexts[output_var_indices,:].unsqueeze(0).transpose(-2, -1)

        device_idx = self.basis_context.weight.get_device()
        attn_context, _ = self.attn(input_contexts.transpose(-2, -1), self.basis_context.weight.transpose(-2, -1), self.basis_context.weight.transpose(-2, -1), self.entmax_bisect_alpha.to(device_idx))
        input_contexts = attn_context.transpose(-2, -1)

        # Setup encoder inputs
        batch_size = input_batch.shape[0]
        x = input_batch.unsqueeze(1)
        z = input_contexts.expand(batch_size, -1, -1)

        # Apply encoder
        x = self.encoder_film_layer(x, z)
        x = self.dropout(x)
        for block in self.encoder_blocks:
            x = block(x, z)

        # Aggregate state over variables
        x = torch.sum(x, dim=-1, keepdim=True)

        # Apply model core
        for block in self.core_blocks:
           x = block(x)

        # Setup decoder inputs
        x = x.expand(-1, -1, output_contexts.shape[-1])
        z = output_contexts.expand(batch_size, -1, -1)

        # Apply decoder
        for block in self.decoder_blocks:
            x = block(x, z)
        x = self.dropout(x)
        x = self.decoder_film_layer(x, z)

        # Remove unnecessary channels dimension
        x = torch.squeeze(x, dim=1)

        return x
