# %%
import math
from typing import Text
import pytorch_lightning as pl
from torch._C import device
import torch.nn.functional as F
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torchtext
from torchtext.data.utils import get_tokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader

import io
import torch
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning import seed_everything
seed_everything(42)


class AlarmDataset(Dataset):
    def __init__(self,data,seq_len,batch_size):
        self.length = len(data)//seq_len # how much data i have         
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
       
    def __getitem__(self, index: int):
        x = self.data[index*self.seq_len:(index*self.seq_len)+seq_len]
        y = self.data[1+index*self.seq_len:1+(index*self.seq_len)+seq_len]
        return x,y
    
    def __len__(self) -> int:
        return self.length

class MyDataModule(pl.LightningDataModule):
    
    def __init__(self, data_path:str, batch_size:int, seq_len:int):
        super().__init__()
        self.batch_size = batch_size
                
        url = data_path
        test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(self.tokenizer,iter(io.open(train_filepath,encoding="utf8"))))
      
        train_data = self.data_process(iter(io.open(train_filepath, encoding="utf8")))
        val_data = self.data_process(iter(io.open(valid_filepath, encoding="utf8")))
        test_data = self.data_process(iter(io.open(test_filepath, encoding="utf8")))
    
        self.train_dataset = AlarmDataset(train_data, seq_len,self.batch_size)
        self.valid_dataset = AlarmDataset(val_data,seq_len,self.batch_size)
        self.test_dataset = AlarmDataset(test_data, seq_len,self.batch_size)
    
    def data_process(self, raw_text_iter):
        data = [torch.tensor([self.vocab[token] for token in self.tokenizer(item)],dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


    def setup(self, stage: None):
        return None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,num_workers=1,drop_last=True, pin_memory=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,num_workers=1,drop_last=True, pin_memory=True)





class TransformerModel(pl.LightningModule):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, seq_len=None):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.ntoken = ntoken
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = torch.nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = torch.nn.Linear(ninp, ntoken)
        self.src_mask = self.generate_square_subsequent_mask(seq_len)
        self.seq_len = seq_len 
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src_mask = src_mask.to(self.device)
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        # print(src.device)
        # print(self.device)
        src_mask = src_mask.to(self.device)
      
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        # print("done")
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0000001)
        return optimizer
    
    def training_step(self,batch,batch_idx):
        x,y = batch
        x = x.T
        y = y.T.reshape(-1)

        # print("Training Shape: ", x.size(),y.size())
        
        if x.size(0) != self.seq_len:
           self.src_mask =  self.generate_square_subsequent_mask(x.size(0))
        
        y_hat = self(x,self.src_mask)

        loss = F.cross_entropy(y_hat.view(-1, self.ntoken),y)
        self.log('train_loss', loss,on_step=True, prog_bar=True, logger=True)
        self.log("train_ppl",math.exp(loss.item()),on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self,batch, batch_idx):
        x,y = batch
        x = x.T
        y = y.T.reshape(-1)

        # print("Validation Shape: ", x.size(),y.size())

        if x.size(0) != self.seq_len:
           self.src_mask =  self.generate_square_subsequent_mask(x.size(0))
        
        y_hat = self(x,self.src_mask)
        # print("> y-hat",y_hat.size())
        loss = F.cross_entropy(y_hat.view(-1, self.ntoken),y)
        self.log('val_loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log("val_ppl",math.exp(loss.item()),on_step=True, prog_bar=True, logger=True)
        return {'val_loss':loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([d['loss']  for d in outputs]).mean()
        print(f"> Avg Training loss = {avg_loss}")
        
    def validation_epoch_end(self, outputs):
        # print(outputs)
        avg_loss = torch.stack([d['val_loss'] for d in outputs]).mean()
        print(f"> Average Valid Loss = {avg_loss}")

    

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)






    
url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'

bsize = 20
seq_len = 35


dm = MyDataModule(url,bsize,seq_len)

ntokens = len(dm.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout,seq_len=seq_len)
trainer = Trainer(gpus=1,max_epochs=1,check_val_every_n_epoch=1,deterministic=True, gradient_clip_val=0.5)
trainer.fit(model,dm.train_dataloader(),dm.val_dataloader())

# %%
