import torch
import torch.nn as nn
import torch.nn.functional as F


def position_encoding(d:int, n:int, dmodel:int, device:str="cpu"):
    """  
    @params:
    --------
    - d:        number of sequences
    - n:        sequence length
    - dmodel:   number of features 
    - device:   "cpu" or "cuda" depending on cuda availability

    @returns:
    ---------
    - encoding: (d,n,dmodel) dimensional positional encoding
    """
    assert(device in ["cpu", "cuda"]), "DeviceError: Invalid device '"+device+"'"

    pos = torch.arange(n)
    encoding = torch.zeros(n, dmodel) 

    power = torch.true_divide(torch.arange(0,dmodel,2), dmodel).unsqueeze(0).repeat(n,1)  # (n, dmodel/2)
    denom = torch.pow(10000, power)
    pos = pos.unsqueeze(1).repeat(1,dmodel//2)  # (n,dmodel/2)

    encoding[:,0::2] = torch.sin( torch.true_divide(pos,denom) )  # (n, dmodel/2)
    encoding[:,1::2] = torch.cos( torch.true_divide(pos,denom) )  # (n, dmodel/2)
    encoding = encoding.unsqueeze(0).repeat(d,1,1).to(device)  # (d,n,dmodel)

    return encoding


class TransformerEncoder(nn.Module):

    def __init__(self, dmodel:int, dq:int, dk:int, dv:int, heads:int, feedforward:int, num_encoders:int=1):
        """  
        @params:
        --------
        - dmodel:           number of features in the input
        - dq:               number of features in the queries
        - dk:               number of features in the keys
        - dv:               number of features in the values
        - heads:            number of heads for multi-head attention
        - feedforward:      number of units in the 2nd layer of the 3-layered feedforward network
        - num_encoders:     number of stacked encoders 
        """
        super(TransformerEncoder, self).__init__()

        # Intialize the parameters
        self.dmodel, self.dq, self.dk, self.dv = dmodel, dq, dk, dv
        self.heads = heads
        self.feedforward = feedforward
        self.num_encoders = num_encoders

        # Multi-Head Attention
        self.Wq = nn.ModuleList([nn.Linear( self.dmodel, self.heads*self.dq ) for _ in range(num_encoders)])
        self.Wk = nn.ModuleList([nn.Linear( self.dmodel, self.heads*self.dk ) for _ in range(num_encoders)])
        self.Wv = nn.ModuleList([nn.Linear( self.dmodel, self.heads*self.dv ) for _ in range(num_encoders)])
        self.unify = nn.ModuleList([nn.Linear( self.heads*self.dv, self.dmodel ) for _ in range(num_encoders)])

        # Normalization
        self.norm1 = nn.ModuleList([nn.LayerNorm(self.dmodel) for _ in range(num_encoders)])

        # Feedforward
        self.ff = nn.ModuleList([nn.Sequential(
                        nn.Linear(self.dmodel, self.feedforward),
                        nn.ReLU(),
                        nn.Linear(self.feedforward, self.dmodel))  for _ in range(num_encoders)])

        # Normalization
        self.norm2 = nn.ModuleList([nn.LayerNorm(self.dmodel) for _ in range(num_encoders)])


    def attention(self, x, enc_num):
        """  
        @params:
        --------- 
        - x:        input torch tensor (d,n,dmodel)
        - enc_num:  index of the encoder in the encoder stack
        """
        queries = self.Wq[enc_num](x)  # (d,n,h*dq)
        keys = self.Wk[enc_num](x)  # (d,n,h*dk)
        values = self.Wv[enc_num](x)  # (d,n,h*dv)

        scores = F.softmax(torch.bmm(queries, keys.transpose(1,2))/self.dk**0.5, dim=-1) # (d,n,n)

        attn = torch.bmm(scores, values)  # (d,n,h*dv)
        unified_attn = self.unify[enc_num](attn)  # (d,n,dmodel)

        return unified_attn, keys, values

    
    def forward(self, x, return_keys_and_values=False):
        """ 
        @params: 
        --------
        - x:                        input torch tensor (d,n,dmodel)
        - return_keys_and_values:   if true returns the keys and valued to be given as input to the decoder
        """
        for i in range(self.num_encoders):
            attn, keys, values = self.attention(x, enc_num=i) # (d,n,dmodel)
            norm1 = self.norm1[i](x + attn) # (d,n,dmodel)
            feedfwd = self.ff[i](norm1) # (d,n,dmodel)
            y = self.norm2[i](norm1 + feedfwd) # (d,n,dmodel)

        if return_keys_and_values:
            return y, keys, values
        else:
            return y


class TransformerDecoder(nn.Module):
    
    def __init__(self, dmodel:int, dq:int, dk:int, dv:int, heads:int, feedforward:int, num_decoders:int=1):
        """  
        @params:
        --------
        - dmodel:           number of features in the output
        - dq:               number of features in the queries
        - dk:               number of features in the keys
        - dv:               number of features in the values
        - heads:            number of heads for multi-head attention
        - feedforward:      number of units in the 2nd layer of the 3-layered feedforward network
        - num_decoders:     number of stacked decoders 
        """
        super(TransformerDecoder, self).__init__()

        # Initialize the parameters
        self.dmodel, self.dq, self.dk, self.dv = dmodel, dq, dk, dv
        self.heads = heads
        self.feedforward = feedforward
        self.num_decoders = num_decoders

        # Multi-Head Attention
        self.Wq = nn.ModuleList([nn.Linear( self.dmodel, self.heads*self.dq ) for _ in range(num_decoders)])
        self.Wk = nn.ModuleList([nn.Linear( self.dmodel, self.heads*self.dk ) for _ in range(num_decoders)])
        self.Wv = nn.ModuleList([nn.Linear( self.dmodel, self.heads*self.dv ) for _ in range(num_decoders)])
        self.unify = nn.ModuleList([nn.Linear( self.heads*self.dv, self.dmodel ) for _ in range(num_decoders)])

        # Normalization
        self.norm1 = nn.ModuleList([nn.LayerNorm(self.dmodel) for _ in range(num_decoders)])

        # Encoder Decoder Attention
        self.Wedq = nn.ModuleList([nn.Linear( self.dmodel, self.heads*self.dq ) for _ in range(num_decoders)])
        self.ed_unify = nn.ModuleList([nn.Linear( self.heads*self.dv, self.dmodel ) for _ in range(num_decoders)])

        # Normalization
        self.norm2 = nn.ModuleList([nn.LayerNorm(self.dmodel) for _ in range(num_decoders)])

        # Feedforward
        self.ff = nn.ModuleList([nn.Sequential(
                        nn.Linear(self.dmodel, self.feedforward),
                        nn.ReLU(),
                        nn.Linear(self.feedforward, self.dmodel))  for _ in range(num_decoders)])

        # Normalization
        self.norm3 = nn.ModuleList([nn.LayerNorm(self.dmodel) for _ in range(num_decoders)])


    def attention(self, x, dec_num):
        """  
        @params:
        --------- 
        - x:        input torch tensor (d,n,dmodel)
        - dec_num:  index of the decoder in the decoder stack
        """
        queries = self.Wq[dec_num](x)  # (d,n,h*dq)
        keys = self.Wk[dec_num](x)  # (d,n,h*dk)
        values = self.Wv[dec_num](x)  # (d,n,h*dv)

        scores = F.softmax(torch.bmm(queries, keys.transpose(1,2))/self.dk**0.5, dim=-1) # (d,n,n)
        attn = torch.bmm(scores, values)  # (d,n,h*dv)
        unified_attn = self.unify[dec_num](attn)  # (d,n,dmodel)

        return unified_attn


    def encoder_decoder_attention(self, x, keys, values, dec_num):
        """  
        @params:
        -------- 
        - x:        input torch tensor (d,n,dmodel)
        - keys:     encoder keys tensor (d,n,h*dk)
        - values:   encoder values tensor (d,n,h*dv)
        - dec_num:  index of the decoder in the decoder stack
        """
        queries = self.Wedq[dec_num](x)  # (d,n,h*dq)
        scores = F.softmax(torch.bmm(queries, keys.transpose(1,2))/self.dk**0.5, dim=-1)  # (d,n,n)
        attn = torch.bmm(scores, values)  # (d,n,h*dv)
        unified_attn = self.ed_unify[dec_num](attn)  # (d,n,dmodel)

        return unified_attn, scores
    

    def forward(self, x, keys, values):
        """  
        @params:
        --------- 
        - x:        input torch tensor (d,m,dmodel)
        - keys:     encoder keys tensor (d,m,h*dk)
        - values:   encoder values tensor (d,m,h*dv)
        """
        for i in range(self.num_decoders):
            attn = self.attention(x, dec_num=i) # (d,m,dmodel)
            norm1 = self.norm1[i](x + attn) # (d,m,dmodel)
            enc_dec_attn, scores = self.encoder_decoder_attention(norm1, keys, values, dec_num=i) # (d,m,dmodel), (d,m,m)
            norm2 = self.norm2[i](enc_dec_attn + norm1) # (d,m,dmodel)
            feedfwd = self.ff[i](norm2) # (d,m,dmodel)
            y = self.norm3[i](feedfwd + norm2) # (d,m,dmodel)

        return y, scores


class Seq2SeqTransformer(nn.Module):

    def __init__(self, dmodel, dq, dk, dv, heads, feedforward, vocab_size, num_encoders=1, num_decoders=1, device="cpu"):
        """  
        @params:
        --------
        - dmodel:           number of features in the output
        - dq:               number of features in the queries
        - dk:               number of features in the keys
        - dv:               number of features in the values
        - heads:            number of heads for multi-head attention
        - feedforward:      number of units in the 2nd layer of the 3-layered feedforward network
        - vocab_size:       size of the vocabulary excluding the UNK, PAD, EOS and SOS tokens
        - num_encoders:     number of stacked encoders
        - num_decoders:     number of stacked decoders 
        """
        super(Seq2SeqTransformer, self).__init__()

        # Initialize the parameters
        self.dmodel, self.dq, self.dk, self.dv = dmodel, dq, dk, dv
        self.heads = heads
        self.feedforward = feedforward
        self.vocab_size = vocab_size+4 # vocab_size + padding + unk + sos + eos
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.device= device
        self.UNK_IDX = 1
        # self.PAD_IDX = 0
        # self.SOS_IDX = 2
        # self.EOS_IDX = 3

        self.embedding = nn.Embedding(self.vocab_size, dmodel) 
        
        self.encoders = TransformerEncoder(self.dmodel, self.dq, self.dk, self.dv, self.heads, \
                        self.feedforward, num_encoders=self.num_encoders)
        self.decoders = TransformerDecoder(self.dmodel, self.dq, self.dk, self.dv, self.heads, \
                        self.feedforward, num_decoders=self.num_decoders)

        self.Wcoverage = nn.Linear(2,1)
        self.Wpgen = nn.Linear(3*self.dmodel, 1)
        self.Wvocab = nn.Linear(2*self.dmodel, self.vocab_size)


    def forward_through_encoder(self, input_indices):
        """ 
        @params:
        --------
        - input_indices:    input data torch tensor of dimension (d,n)
        """
        d,n = input_indices.shape

        # Replace OOV word indices with the UNK index
        input_indices_with_unk_tokens = input_indices.clone()
        input_indices_with_unk_tokens[input_indices>=self.vocab_size] = self.UNK_IDX

        # Convert the input indices into embeddings
        x = self.embedding(input_indices_with_unk_tokens) + position_encoding(d,n,self.dmodel).to(self.device) # (d,n,dmodel)

        # Forward the embeddings through the encoder stack
        enc_states, enc_keys, enc_values = self.encoders(x, return_keys_and_values=True) # (d,n,dmodel), (d,n,h*dk), (d,n,h*dv)
        
        return enc_states, enc_keys, enc_values


    def forward_through_decoder(self, input_indices, output_indices, enc_states, enc_keys, enc_values, coverage):
        """ 
        @params:
        --------
        - input_indices:    torch tensor containing indices of input words (d,n)
        - output_indices:   torch tensor containing indices of input words (d,n)
        - enc_states:       torch tensor holding the encoder states output from the last encoder (d,n,dmodel)
        - enc_keys:         torch tensor holding the encoder keys output from the last encoder (d,n,h*dk)
        - enc_values:       torch tensor holding the encoder values output from the last encoder (d,n,h*dv)
        - coverage:         summed up attention distributions from previous timesteps (d,n)

        @note:
        ------
        - OOV words in the input have indices beyond vocab_size starting from vocab_size for each document
        """
        d,n = input_indices.shape

        # input_indices_with_unk_tokens = input_indices.clone()
        # input_indices_with_unk_tokens[input_indices>self.vocab_size] = self.UNK_IDX
        # x = self.embedding(input_indices_with_unk_tokens) + position_encoding(d,n,self.dmodel).to(self.device) # (d,n,dmodel)

        # Convert the output indices into embeddings
        y = self.embedding(output_indices) # (d,m,dmodel) 
        
        # Forward the output embeddings through the decoder stack
        dec_states, dec_scores = self.decoders(enc_states, enc_keys, enc_values) # (d,n,dmodel), (d,m,n)

        # Calculate the attention distribution over the input sequence
        attn_dist = self.Wcoverage(torch.cat((dec_scores[:,-1,:].view(d,n,1), coverage.view(d,n,1)), axis=-1)).view(d,n) # (d,n)
        extended_attn_dist = attn_dist.view(d,n,1).repeat(1,1,self.dmodel) # (d,n,dmodel) 

        # Generate the context vector from the attention distribution over the input sequence
        context_vec = torch.sum(torch.mul(extended_attn_dist, enc_states), dim=1) # (d,dmodel)

        # Generate the switch probability to switch between choosing from the training vocabulary or the input vocabulary
        p_gen = self.Wpgen(torch.cat((context_vec, dec_states[:,-1,:], y[:,-1,:]), dim=-1)) # (d,1)
        
        # Generate the probability distribution over the training vocabulary
        p_vocab = self.Wvocab(torch.cat((context_vec, dec_states[:,-1,:]), dim=-1)) # (d, vocab_size)

        # Determine the probability distribution over the input vocabulary
        p_copy = attn_dist # (d,n)

        combined_dist = torch.zeros(d, n+self.vocab_size).to(self.device)
        
        # Create indices for vocabulary from 0 to vocab_size for all sequences (i.e d-dimension)
        vocab_indices = torch.LongTensor(range(self.vocab_size)).view(1,-1).repeat(d,1).to(self.device) # (d, vocab_size)

        # scatter_add adds all values from the source tensor into self at the indices specified in the index tensor
        combined_dist = combined_dist.scatter_add(dim=1, index=vocab_indices, src=torch.mul(p_gen, p_vocab))
        combined_dist = combined_dist.scatter_add(dim=1, index=input_indices, src=torch.mul(1-p_gen, p_copy))

        coverage += attn_dist

        return combined_dist, coverage, attn_dist



