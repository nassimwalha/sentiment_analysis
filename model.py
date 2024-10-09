import torch
from transformers import RobertaModel
class RobertaClassifier(torch.nn.Module):
    def __init__(self):
        super(RobertaClassifier, self).__init__()
        self.roberta_model = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768) #Linear layer applied on the output embedding
        self.dropout = torch.nn.Dropout(0.3) #Dropout for regularization
        self.classifier = torch.nn.Linear(768, 4) #Output layer: 4 classes

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0] #Get last hidden state
        pooler = hidden_state[:, 0] #Get the hidden state corresponding to the first token <s> used in RoBERTa for classification
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output