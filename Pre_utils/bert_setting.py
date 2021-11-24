import transformers
from transformers import BertTokenizer,AutoTokenizer


class BertPreSetting:
    def __init__(self, args, overal_maxlen):
        if args.model_type == r'ernie':
            self.path = './Pre-training/ernie'
            self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        if args.model_type == r'roberta':
            self.path = './Pre-training/roberta'
            self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        else:
            self.path = r'./Pre-training/BERT_base'
            self.tokenizer = BertTokenizer.from_pretrained(self.path)
        self.args = args
        self.max_length = overal_maxlen

    def get_inputs(self, args, input_pre_x):
        encoded_inputs = self.tokenizer(input_pre_x, return_tensors='tf', padding=True, truncation=True,
                                        max_length=self.max_length)

        inputs_ids = encoded_inputs.get('input_ids')
        inputs_mask = encoded_inputs.get('attention_mask')
        inputs_tokentype = encoded_inputs.get('token_type_ids')

        return inputs_ids, inputs_mask, inputs_tokentype
