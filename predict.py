import json
import torch
from torch import nn
from transformers import BertTokenizer, BertModel

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Set up parameters
bert_model_name = config['bert_model_name']
num_classes = config['num_classes']
max_length = config['max_length']
model_path = "./models/bertlaw_classifier.pth"

# Build custom BERT classifier
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Function to preprocess the input text
def preprocess_text(text, tokenizer, max_length):
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    return encoding['input_ids'].to(device), encoding['attention_mask'].to(device)

# Function to make predictions
def predict(text, model, tokenizer, max_length):

    class_dict = {0: 'affirmed',
                  1: 'applied',
                  2: 'approved',
                  3: 'cited',
                  4: 'considered',
                  5: 'discussed',
                  6: 'distinguished',
                  7: 'followed',
                  8: 'referred to',
                  9: 'related'}

    input_ids, attention_mask = preprocess_text(text, tokenizer, max_length)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, dim=1)
    return conf, class_dict[preds.item()]

# Example usage
if __name__ == "__main__":
    text = """On the question of the level of unreasonableness necessary to attract the discretion, I respectfully agree with the comment of Sackville J in Seven Network Limited v News Limited (2007) 244 ALR 374 at [62] questioning the utility of substituting a requirement that rejection be "plainly unreasonable" for the requirement that it be "unreasonable". Given the evaluative character of the judgment involved the addition of the word "plainly" which is itself evaluative, has no useful function. 37 At the time that Sirtex made its offer to UWA the prospect of UWA succeeding against Sirtex depended critically upon: 1. UWA establishing its case against Dr Gray and, in particular, that he had breached his fiduciary duty. 2. UWA establishing that Sirtex was accessorially liable in relation to that breach, a position that depended upon establishing that Sirtex was aware of facts constituting (and which would have indicated to a reasonable person) the breach of fiduciary duties owed by Dr Gray to UWA. 38 It cannot be said that UWA acted unreasonably in proceeding on the basis that it had a reasonable cause of action against Dr Gray. True it is that the case as framed and presented depended upon an important proposition of law as to the existence of an implied term in the contract of Dr Gray's employment with UWA. But the correctness of that proposition had not previously been tested in Australia in circumstances of the kind which arose in this case. This is not a case, in my opinion, in which it is appropriate to take a hindsight test to the facts known to UWA at the time of Sirtex's offer and conclude that it ought to have known that the law was against it. 39 There were of course other hazards in the way of UWA's path to success against Dr Gray and therefore against Sirtex. The question whether the relevant inventions were made while Dr Gray was an employee of UWA was one issue upon which findings adverse to UWA were made on all but the DOX-Spheres technology. There was also a finding adverse to UWA that none of the Sirtex directors, apart from Dr Gray, were on notice of a potential claim. To establish any cause of action against Sirtex based on knowing involvement in his alleged breaches of fiduciary duty would have depended entirely upon his role as a director of Sirtex and whether his knowledge could be attributed to that company. In addition, UWA faced substantial defences by Sirtex based on UWA's delay in commencing proceedings after it first became aware of the facts relevant to its claimed causes of action. 40 The preceding factors may be seen as weighing to some degree in favour of the Sirtex motion. On the other hand the offer came as the trial commenced. That is a factor, given the focus on the trial process which would then have existed, that militates against a finding of unreasonableness on the part of UWA in refusing the offer. That conclusion is not affected by the fact that Sirtex was making a counter-offer. The counter-offer was not a variation on a theme opened by UWA's offers. It was quite different and could have been proposed earlier."""
    conf, prediction = predict(text, model, tokenizer, max_length)
    print(f"Predicted label: {prediction}, Confidence: {conf}")
