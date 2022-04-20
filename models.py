# import tensorflow.keras.optimizers as opt
import numpy as np
import tensorflow.keras.optimizers as opt
import tensorflow as tf
from my_layers import GlobalMaxPooling1D, GlobalAveragePooling1D, MultiHeadAttention

'''
    篇章级特征：句法结构信息，构建一个图，通过图神经网络获得向量。来拼
    句法级特征。
'''


class ModelsConfigs:
    # 获得优化器
    def get_optimizer(self, args):
        clipvalue = 0
        clipnorm = 10
        if args.algorithm == 'rmsprop':
            optimizer = opt.RMSprop(learning_rate=args.learning_rate, rho=0.9, epsilon=1e-06, clipnorm=clipnorm,
                                    clipvalue=clipvalue)

        return optimizer

    # 获得损失函数和评估方式
    def get_loss_metric(self, args):
        if args.loss == 'mse':
            # 均方误差
            loss = 'mean_squared_error'
            metric = 'mean_absolute_error'
        else:
            loss = 'mean_absolute_error'
            metric = 'mean_squared_error'

        return loss, metric


class Models:
    def __init__(self, args, initial_mean_value, overal_maxlen, vocab):
        if initial_mean_value.ndim == 0:
            self.initial_mean_value = np.expand_dims(initial_mean_value, axis=0)

        self.num_outpus = len(self.initial_mean_value)

    def get_model(self, args, overal_maxlen, vocab):
        if args.model_type == 'ClassModule':
            '''
                消融实验
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten
            import transformers

            from transformers import BertTokenizer, TFBertModel
            path = 'Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)

            #冻结encoder部分参数
            for layer in bert_model.layers[:]:
                if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
                    for idx, layer in enumerate(layer.encoder.layer):
                        if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                            layer.trainable = False
            bert_model.summary()

            # 第一个输入
            input1 = Input(shape=(overal_maxlen,), dtype='int32')
            # emb_out1.shape == (None,600,300)
            emb_out1 = Embedding(args.vocab_size, args.emb_dim, name='emb')(input1)

            # trm_out.shape == (None,600,300)
            x1 = MultiHeadAttention(3, 100)(emb_out1)

            # 第二个输入
            # shape == (None,600)
            input_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_tokentype = Input(shape=(overal_maxlen,), dtype='int32')
            dim_out2 = bert_model(
                {"input_ids": input_ids, "token_type_ids": input_tokentype, "attention_mask": input_mask})
            # bert_last_hidden_output = dim_out2.last_hidden_state
            bert_pooler_output = dim_out2.last_hidden_state
            # x2.shape == (None,600,768)
            x2 = bert_pooler_output

            # out.shape == (None,600,1368)
            x_feature = concatenate([x1, x2], axis=-1)
            # 把Bert的输出作为初始化门偏置
            matrix = Dense(1068, activation='sigmoid')(emb_out1)
            out = x_feature * matrix


            max = GlobalMaxPooling1D()(out)
            avg = GlobalAveragePooling1D()(out)

            x = concatenate([max, avg], axis=-1)

            x = Dropout(0.1)(x)
            x = Dense(100, activation='swish')(x)
            x = Dense(100, activation='swish')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=score)
            model.summary()

            return model


'''
	BERT的最后输出有两种
last_hidden_state：维度【batch_size, seq_length, hidden_size】，这是训练后每个token的词向量。
pooler_output：维度是【batch_size, hidden_size】，每个sequence第一个位置CLS的向量输出，用于分类任务。
'''
