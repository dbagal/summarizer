from collections import  Counter, defaultdict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class GloVeDataset(Dataset):


    def __init__(self, vocab:object, tokenizer:object, window_size:int=5):
        super(GloVeDataset).__init__()
        """  
        @params
        - vocab => instance of the Vocab class containing word to index mapping
        - tokenizer => tokenizes the input string and returns a list of tokens
        - window_size => coocurrence distance 
        """

        # End of the line marker which will be substituted for \n in the text
        self.eol_marker=" "
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.vocab = vocab
        self.window_size = window_size
        self.tokenizer= tokenizer

    
    def generate(self, folder:str):
        content = self.extractContent(folder)
        self.buildMatrix(content)


    def extractContent(self, folder:str):
        paths = [os.path.join(folder,fname) for fname in os.listdir(folder) if fname.endswith(".txt")]
        content = ""
        for path in paths:
            with open(path, "r") as fp:
                text = fp.read().lower().replace("\n", self.eol_marker)
            content += self.eol_marker+text
        return content


    def buildMatrix(self, content:str):
        tokens = self.tokenizer(content)
        token_indices = [self.vocab.word_to_idx.get(token, self.UNK_IDX) for token in tokens]
        num_tokens = len(tokens)

        # defaultdict adds a default value for the key when it doesn't exist
        matrix = defaultdict(Counter)  

        # Create a sparse matrix with cooccurrence counts of only the cooccuring words based on the window_size
        for i, token_idx in enumerate(token_indices):
            context_window_begin = max(i - self.window_size, 0)
            context_window_end = min(i + self.window_size + 1, num_tokens)
            for j in range(context_window_begin, context_window_end):
                if i!=j:
                    context_word_idx = token_indices[j]
                    matrix[token_idx][context_word_idx] += 1

        # Convert the cooccurrence counts into probabilities
        matrix = {word_idx:{context_word_idx:count/sum(matrix[word_idx].values()) \
                for context_word_idx, count in matrix[word_idx].items()} for word_idx in matrix.keys()}

        word_indices = []
        context_word_indices = []
        word_cooccurrence_probs = []
        self.num_indices = 0

        for word_idx, context_word_indices_and_probs in matrix.items():
            for context_word_idx, prob in context_word_indices_and_probs.items():
                word_indices.append(word_idx)
                context_word_indices.append(context_word_idx)
                word_cooccurrence_probs.append(prob)
                self.num_indices+=1

        self.word_indices = torch.LongTensor(word_indices)
        self.context_word_indices = torch.LongTensor(context_word_indices)
        self.word_cooccurrence_probs = torch.FloatTensor(word_cooccurrence_probs)


    def __len__(self):
        return self.num_indices

    
    def __getitem__(self, idx):
        return self.word_indices[idx], self.context_word_indices[idx], self.word_cooccurrence_probs[idx]



class GloVeModel(nn.Module):
    
    
    def __init__(self, num_embeddings:int, embedding_dim:int):
        super(GloVeModel, self).__init__()
        """ 
        @params
        - num_embeddings => offset plus number of words in the vocabulary
        - embedding_dim => number of features in the word embedding 
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim), requires_grad=True)
        self.context_word_embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim), requires_grad=True)
        self.word_bias = nn.Parameter(torch.zeros(num_embeddings,1), requires_grad=True)
        self.context_word_bias = nn.Parameter(torch.zeros(num_embeddings,1), requires_grad=True)


    def save(self, dir:str):
        self.embeddings = self.word_embeddings + self.context_word_embeddings
        fname = "glove_"+str(self.num_embeddings)+"_"+str(self.embedding_dim)+".pt"
        torch.save(self.embeddings, os.path.join(dir, fname))

    
    def loss(self, cooccurrence_probs:torch.FloatTensor, model_outputs:torch.FloatTensor, xmax:int=100, alpha:float=0.75):
        device = model_outputs.device
        weights = torch.pow(cooccurrence_probs/xmax, alpha)
        weights = torch.min(weights, torch.ones(weights.shape).to(device))
        loss = torch.mul(weights, torch.pow(model_outputs - torch.log(cooccurrence_probs), 2))
        loss = torch.sum(loss)
        return loss


    def forward(self, word_indices:torch.LongTensor, context_word_indices:torch.LongTensor):
        wi = self.word_embeddings[word_indices]
        wj = self.context_word_embeddings[context_word_indices]
        bi = self.word_bias[word_indices].view(-1)
        bj = self.context_word_bias[context_word_indices].view(-1)
        x = torch.sum(torch.mul(wi,wj), dim=1) + bi + bj
        
        return x


    def train(self, train_loader:DataLoader, optimizer:optim, num_epochs:int, save_dir:str, device:str="cuda"):
        
        progress_bar = tqdm(range(num_epochs), position=0, leave=True)
        
        for epoch in progress_bar:     
            epoch_loss = 0.0   
            train_total = 0
            progress_bar.set_description(f"Epoch {epoch} ")
            
            for step, data in enumerate(train_loader):
            
                word_indices, ctxt_word_indices, cooccurrence_probs = data
                
                word_indices = word_indices.to(device)
                ctxt_word_indices = ctxt_word_indices.to(device)
                cooccurrence_probs = cooccurrence_probs.to(device)

                optimizer.zero_grad()

                output = self(word_indices, ctxt_word_indices)
                
                loss = self.loss(cooccurrence_probs, output)
                
                epoch_loss += loss.item()
                train_total += 1
                
                b_loss = str(round(loss.item(), 8))
                t_loss = str(round(epoch_loss/train_total, 8))
                progress_bar.set_postfix({'Batch':step,'Batch_Loss': b_loss, 'Train_Loss':t_loss})
                
                loss.backward()
                optimizer.step()

        self.save(save_dir)