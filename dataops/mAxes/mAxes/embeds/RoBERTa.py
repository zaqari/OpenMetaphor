import torch.nn as nn
import torch
from transformers import *
import numpy as np

class axComp(nn.Module):

    def __init__(self):
        super(axComp, self).__init__()

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        config = RobertaConfig.from_pretrained('roberta-base', output_hidden_states=True)
        self.mod = RobertaModel.from_pretrained('roberta-base', config=config)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def translate(self, lexeme_text):
        self.mod.eval()
        with torch.no_grad():
            tokens = torch.tensor(self.tokenizer.encode(lexeme_text, add_special_tokens=False)).unsqueeze(0)
            outputs = self.mod(tokens)

            return outputs[2][0].sum(dim=1)
            #return outputs[2][1:][10].sum(dim=1)

    def translate_(self, lexeme_text):
        self.mod.eval()
        with torch.no_grad():
            tokens = torch.tensor(self.tokenizer.encode(lexeme_text, add_special_tokens=False)).unsqueeze(0)
            outputs = self.mod(tokens)

            return outputs[2][0]#[1:][10]

    def translate_chunk(self, text, layer_number=0):
        """

        :param text:
        :param layer_number: the layer that you want results from. 0 is the base encoding layer, all after are attention layers.
        :return:
        """
        self.mod.eval()
        with torch.no_grad():
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            outputs = self.mod(torch.tensor(ids).unsqueeze(0))

            return outputs[2][layer_number].squeeze(0), np.array(self.tokenizer.convert_ids_to_tokens(ids))

    def make_axes(self, col1, col2, df):
        return [self.translate(phrase[0])[0] - self.translate(phrase[1])[0] for phrase in df[[col1, col2]].values]

    def df_targets(self, col1, col2, df):
        data = [' '.join(df[[col1, col2]].loc[i].values.tolist()) for i in df.index]
        return [self.translate(phrase)[0].sum(dim=0) for phrase in data]
    
    def target(self, phrase):
        return self.translate(phrase)[0].sum(dim=0)

    def lexeme(self, word):
        return self.translate(word)

    def adhoc_axis(self, lista, listb):
        axes = [self.translate(i) - self.translate(k) for k in listb for i in lista]
        return torch.cat(axes, dim=0).mean(dim=0)

vecs = axComp()
