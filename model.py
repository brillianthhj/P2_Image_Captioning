import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        
        captions = captions[:, :-1]
        embedding = self.embed(captions)
        embedding = torch.cat((features.unsqueeze(dim = 1), embedding), dim = 1)
        lstm_out, hidden = self.lstm(embedding)
        outputs = self.linear(lstm_out)

#         print("features shape: ", features.shape)
#         print("captions shape: ", captions.shape)
#         print("embeddings shape: ", embeddings.shape)
#         print("hiddens shape: ", lstm_out.shape)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        for i in range(20):                                    # maximum sampling length
            lstm_out, states = self.lstm(inputs, states)        # (batch_size, 1, hidden_size),
            outputs = self.linear(lstm_out.squeeze(1))          # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                       # (batch_size, 1, embed_size)
        
        return sampled_ids