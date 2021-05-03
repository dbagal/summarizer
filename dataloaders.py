from torch.utils.data import Dataset
import torch
import json, os
from vocab import Vocab


class SummarizationDataset(Dataset):
    

    def __init__(self, folder_path:str, vocab:Vocab, input_seq_len:int, output_seq_len:int, tokenizer:object) -> None:
        """  
        @params:
        --------
        - folder_path:      path to the 'data' folder
        - vocab:            Vocab class object that manages the vocabulary
        - input_seq_len:    length of the input sequence used for truncation/padding
        - output_seq_len:   length of the output sequence used for truncation/padding
        - tokenizer:        python function which takes a string as input and tokenizes it 
                            into a list of sub strings according to the tokenization level
        """

        super(SummarizationDataset, self).__init__()
        raw_data = {"inputs":[], "outputs":[]}
        
        # Load the data from the JSON files in a python dictionary
        fnames =[fname for fname in os.listdir(folder_path) if fname.endswith(".json")] 
        for fname in fnames:
            with open(os.path.join(folder_path, fname), "r") as fp:
                json_data = json.load(fp)
                raw_data["outputs"] += json_data["outputs"]
                raw_data["inputs"] += json_data["inputs"]

        # Tokenize the raw text data
        outputs = [tokenizer(op) for op in raw_data["outputs"]]
        inputs = [tokenizer(ip) for ip in raw_data["inputs"]]

        # Set the padding index, unknown word index, 
        # start-of-sentence index and end-of-sentence index
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.SOS_IDX = 2
        self.EOS_IDX = 3

        # The start and end indices are the SOS and EOS respectively
        output_seq_len -= 2

        # The OOV words in the input are assigned indices starting from the last index in the vocabulary
        # For input sequence do the truncation before hand only so that input OOV words get appropriate indices after truncation
        for i in range(len(inputs)):
            if len(inputs[i]) > input_seq_len:
                inputs[i] =  inputs[i][0:input_seq_len]

        # Convert words to indices for the input
        for i,abstract in enumerate(inputs):
            last_index = vocab.index
            for j,word in enumerate(abstract):
                idx = vocab.word_to_idx.get(word.lower(), None)
                if idx is None:
                    inputs[i][j] = last_index
                    last_index += 1
                else:
                    inputs[i][j] = idx

            if len(inputs[i]) < input_seq_len:
                inputs[i] += [self.PAD_IDX,]*(input_seq_len - len(inputs[i]))

        # Convert words to indices for the output
        for i,claim in enumerate(outputs):
            for j,word in enumerate(claim):
                idx = vocab.word_to_idx.get(word.lower(), None)
                if idx is None:
                    outputs[i][j] = self.UNK_IDX
                else:
                    outputs[i][j] = idx

            if len(outputs[i]) > output_seq_len:
                outputs[i] =  outputs[i][0:output_seq_len]
            elif len(outputs[i]) < output_seq_len:
                outputs[i] += [self.PAD_IDX,]*(output_seq_len - len(outputs[i]))
            
            outputs[i] = [self.SOS_IDX,] + outputs[i] + [self.EOS_IDX]

        self.outputs = torch.LongTensor(outputs)
        self.inputs = torch.LongTensor(inputs)


    def __len__(self):
        return self.outputs.shape[0]


    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


