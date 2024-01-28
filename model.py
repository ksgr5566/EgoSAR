import torch
from torch import nn
from torchvision import models

NUM_NOUNS = 51
NUM_VERBS = 19
NUM_ACTIONS = 106

class ActionClassifier(nn.Module):
    def __init__(self):
        super(ActionClassifier, self).__init__()

        num_ftrs = 2048
        num_ftrs_aux = 768
        num_lstm_out = 512
        num_lstm_out_aux = 256

        # Noun Inception
        self.noun_model = models.inception_v3(pretrained=True)
        self.noun_model.fc = nn.Identity()
        self.noun_model.AuxLogits.fc = nn.Identity()
        self.noun_fc = nn.Linear(num_ftrs, NUM_NOUNS)

        # Verb Inception
        self.verb_model = models.inception_v3(pretrained=True)
        self.verb_model.transform_input=False
        self.verb_model.Conv2d_1a_3x3.conv = nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.verb_model.Conv2d_1a_3x3.bn = nn.BatchNorm2d(32)
        self.verb_model.AuxLogits.conv1 = nn.Conv2d(128, 768, kernel_size=(3, 3), stride=(2, 2))
        self.verb_model.fc = nn.Identity()
        self.verb_model.AuxLogits.fc = nn.Identity()
        self.lstm = nn.LSTM(num_ftrs, num_lstm_out, batch_first=True)
        self.lstm_aux = nn.LSTM(num_ftrs_aux, num_lstm_out_aux, batch_first=True)
        self.verb_fc = nn.Linear(512, NUM_VERBS)

        # Action classification layers
        self.fc = nn.Linear(num_ftrs + num_lstm_out, NUM_ACTIONS)
        self.aux_fc = nn.Linear(num_ftrs_aux + num_lstm_out_aux, NUM_ACTIONS)

    def forward(self, x, y):
        if self.training:
          noun_out, noun_aux = self.noun_forward(x)
          verb_out, verb_aux = self.verb_forward(y)
          out_aux = torch.cat((noun_aux, verb_aux), dim=1)
          action_logits_aux = self.aux_fc(out_aux)
        else:
          noun_out = self.noun_forward(x)
          verb_out = self.verb_forward(y)

        out = torch.cat((noun_out, verb_out), dim=1)
        noun_logits = self.noun_fc(noun_out)
        verb_logits = self.verb_fc(verb_out)
        action_logits = self.fc(out)

        if self.training:
          return action_logits, action_logits_aux, noun_logits, verb_logits
        return action_logits, noun_logits, verb_logits

    def noun_forward(self, x):
        batch_size, temporal_dim, _, _, _ = x.shape
        x = x.view(batch_size * temporal_dim, x.size(2), x.size(3), x.size(4)) # bs*T, 3, 224, 224

        if self.training:
          x, aux = self.noun_model(x)
          aux = aux.view(batch_size, temporal_dim, -1)
          aux = torch.mean(aux, dim=1)
        else:
          x = self.noun_model(x) # bs*T, 2048
        x = x.view(batch_size, temporal_dim, -1) # bs, T, 2048
        x = torch.mean(x, dim=1)

        if self.training:
          return x, aux
        return x

    def verb_forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        x = x.view(batch_size * sequence_length, C, H, W)

        if self.training:
          features, aux = self.verb_model(x)
          aux = aux.view(batch_size, sequence_length, -1)
          aux_out, _ = self.lstm_aux(aux)
          aux_out = aux_out[:, -1, :]
        else:
          features = self.verb_model(x)

        features = features.view(batch_size, sequence_length, -1)
        out, _ = self.lstm(features)
        out = out[:, -1, :]

        if self.training:
          return out, aux_out
        return out