from sequence_models import Seq2SeqTransformer
from vocab import *
import torch

class Eval():

    def __init__(self, model:Seq2SeqTransformer, vocab:Vocab, tokenizer:object, device:str="cpu") -> None:
        """  
        @params:
        --------
        - model:        Trained Seq2SeqTransformer model
        - vocab:        Vocab class object that manages the vocabulary
        - tokenizer:    python function which takes a string as input and tokenizes it 
                        into a list of sub strings according to the tokenization level
        - device:       "cpu" or "cuda" depending on cuda availability
        """
        self.device = device
        self.vocab = vocab
        self.input_vocab = dict()
        self.idx_to_word = dict()
        self.combined_vocab = dict()
        self.combined_vocab.update(self.vocab.word_to_idx)
        self.tokenizer = tokenizer
        self.model = model
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.SOS_IDX = 2
        self.EOS_IDX = 3


    def get_indices(self, input_text:str):
        """  
        @params:
        --------
        - input_text: raw python string
        """
        # Tokenize the input text
        input_tokens = self.tokenizer(input_text)
        last_index = self.vocab.index

        # Convert the input tokens into indices
        for i in range(len(input_tokens)):
            idx = self.vocab.word_to_idx.get(input_tokens[i].lower(), None)
            if idx is None:
                input_tokens[i] = last_index
                self.input_vocab[input_tokens[i]] = last_index
                last_index += 1
            else:
                input_tokens[i] = idx

        output_tokens = [self.SOS_IDX, ] # Consider the start token to be UNK for the time being.
        
        input_indices = torch.LongTensor(input_tokens).view(1,-1) # (1,n)
        output_indices = torch.LongTensor(output_tokens).view(1,-1) # (1,m)

        # Combine the training vocabulary and the input vocabulary
        self.combined_vocab.update(self.input_vocab)
        self.idx_to_word = {v: k for k, v in self.combined_vocab.items()}

        return input_indices, output_indices


    def eval(self, input_text:str, max_len:int=20, beam_width:int=3):
        """  
        @params:
        --------
        - input_text:   raw python string
        - max_len:      maximum number of tokens to output in the summary
        """
        output = []

        # Prepare the input text
        input_indices, output_indices = self.get_indices(input_text)
        input_indices = input_indices.to(self.device)
        output_indices = output_indices.to(self.device)

        coverage = torch.zeros(input_indices.shape[0], input_indices.shape[1]).to(self.device)
        ts = 0

        # Forward the input through the encoder stack
        enc_states, enc_keys, enc_values = self.model.forward_through_encoder(input_indices)
        
        # Decode the output summary token at each timestep
        while ts!=max_len:

            combined_dist, coverage, _ = self.model.forward_through_decoder(input_indices, output_indices[:,0:ts+1], \
                                                                    enc_states, enc_keys, enc_values, coverage)
            
            idx = torch.argmax(combined_dist, dim=-1).view(1,1) # (1,1)

            if idx in (self.EOS_IDX, self.PAD_IDX):
                break

            output_indices = torch.cat((output_indices, idx), dim=-1)

            output_word = self.idx_to_word.get(idx.item(), None)
            if output_word is None:
                print(f"Error in decoding with index {idx.item()}")
            else:
                output.append(output_word)
            ts += 1

        output_string = " ".join(output)

        return output_string

