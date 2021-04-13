from model import BertCrfForNer
from transformers import BertTokenizer
import torch


def convert_examples_to_features(example:str,max_seq_length:int,tokenizer:BertTokenizer):
    tokens = []
    valid_mask = []
    for word in example:
        word_tokens = tokenizer.tokenize(word)
        # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
        for i, word_token in enumerate(word_tokens):
            if i == 0:
                valid_mask.append(1)
            else:
                valid_mask.append(0)
            tokens.append(word_token)

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        valid_mask = valid_mask[: (max_seq_length - special_tokens_count)]

    tokens += [tokenizer.sep_token]
    valid_mask.append(1)

    tokens = [tokenizer.cls_token] + tokens
    valid_mask.insert(0, 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)

    input_ids += [tokenizer.pad_token_id] * padding_length
    input_mask += [0] * padding_length
    valid_mask += [0] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(valid_mask) == max_seq_length

    conver_tensor = lambda x:torch.tensor([x], dtype=torch.long)

    input_ids = conver_tensor(input_ids)
    input_mask = conver_tensor(input_mask)
    valid_mask = conver_tensor(valid_mask)

    inputs = {"input_ids": input_ids, "attention_mask": input_mask, "valid_mask": valid_mask, "decode": True}
    return inputs,tokens


class NerPredict(object):
    def __init__(self,model_name_or_path):
        self.model = BertCrfForNer.from_pretrained(model_name_or_path)
        self.id2label = self.model.config.id2label
        self.max_seq_length = self.model.config.max_position_embeddings
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.n_gpu = torch.cuda.device_count()
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        print(self.id2label)

    def predict(self, text:str):
        inputs, tokens = convert_examples_to_features(text, self.max_seq_length, self.tokenizer)
        decode = inputs.pop("decode")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        inputs["decode"] = decode

        outputs = self.model(**inputs)[0]
        output_labels = [self.id2label[id] for id in outputs[0].detach().cpu().numpy() if id in self.id2label]
        print(output_labels)
        print(tokens)

        print(len(output_labels))
        print(len(tokens))

        for i in range(len(output_labels)):
            print("%s\t%s" % (tokens[i], output_labels[i]))

        return tokens,output_labels


if __name__ == '__main__':
    model_name_or_path = "checkpoint"

    pre = NerPredict(model_name_or_path)

    text = "那不勒斯vs锡耶纳以及桑普vs热那亚之上呢？"
    preicd = pre.predict(text)

    print(preicd)



