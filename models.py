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
        if args.model_type == 'Only bert_input':
            '''
                bert只作为输入和输出，中间什么都没有
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten

            from transformers import BertTokenizer, TFBertModel
            path = 'Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)

            # 第一个输入
            input1 = Input(shape=(overal_maxlen,), dtype='int32')
            # emb_out1.shape == (None,600,300)
            emb_out1 = Embedding(args.vocab_size, args.emb_dim, name='emb')(input1)

            # 第二个输入
            input_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_tokentype = Input(shape=(overal_maxlen,), dtype='int32')
            dim_out2 = bert_model(
                {"input_ids": input_ids, "token_type_ids": input_tokentype, "attention_mask": input_mask})
            # ★★★★★ 后面一定要加.pooler_output
            # dim_out2.last_hidden_state.shape == (None,600,768)
            bert_output = dim_out2.last_hidden_state

            trm_out = MultiHeadAttention(3, 100)(emb_out1)
            # trm_out = add([trm_out, trm_out_tem])
            trm_out = Dense(200, activation='relu')(trm_out)

            max1 = GlobalMaxPooling1D()(trm_out)
            avg = GlobalAveragePooling1D()(trm_out)
            x = concatenate([avg, max1], axis=-1)
            x = Dropout(0.1)(x)
            x = Dense(100, activation='relu')(x)
            x = Dense(100, activation='relu')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=score)
            model.summary()

            return model

        if args.model_type == 'bert_pool':
            '''
                取bert的pooler_output  不进行池化，直接和第一个输入池化后拼接的x1 进行拼接
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten

            from transformers import BertTokenizer, TFBertModel
            path = 'Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)

            # 第一个输入
            input1 = Input(shape=(overal_maxlen,), dtype='int32')
            # emb_out1.shape == (None,600,300)
            emb_out1 = Embedding(args.vocab_size, args.emb_dim, name='emb')(input1)

            # trm_out.shape == (None,600,300)
            trm_out = MultiHeadAttention(3, 100)(emb_out1)
            # trm_out.shape == (None,600,200)
            trm_out = Dense(200, activation='relu')(trm_out)

            # max1.shape == avg1.shape == (None,200)
            max1 = GlobalMaxPooling1D()(trm_out)
            avg1 = GlobalAveragePooling1D()(trm_out)
            # x1.shape == (None,400)
            x1 = concatenate([avg1, max1], axis=-1)

            # 第二个输入
            # shape == (None,600)
            input_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_tokentype = Input(shape=(overal_maxlen,), dtype='int32')
            dim_out2 = bert_model(
                {"input_ids": input_ids, "token_type_ids": input_tokentype, "attention_mask": input_mask})
            # bert_last_hidden_output = dim_out2.last_hidden_state
            bert_pooler_output = dim_out2.pooler_output

            x2 = bert_pooler_output

            # bert_out = Dense(200,activation='relu')(bert_output)

            # max2.shape == avg2.shape == (None,768)
            # max2 = GlobalMaxPooling1D()(bert_out)
            # avg2 = GlobalAveragePooling1D()(bert_out)
            # x2.shape == 968 (★：为什么不是768+768 = 1536 ?)
            # x2 = concatenate([avg2,max2],axis=-1)

            x = concatenate([x1, x2], axis=-1)
            x = Dropout(0.1)(x)
            x = Dense(100, activation='relu')(x)
            x = Dense(100, activation='relu')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=score)
            model.summary()

            return model

        if args.model_type == 'bert_hiddenState':
            '''
                取bert的last_hidden_state,和经过多头注意力后的第一个数据进行拼接。然后进行双池化，输出
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten

            from transformers import BertTokenizer, TFBertModel
            path = 'Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)

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

            x2 = bert_pooler_output

            # 两个输入进行拼接
            out = concatenate([x1, x2], axis=-1)
            # 双池化
            max = GlobalMaxPooling1D()(out)
            avg = GlobalAveragePooling1D()(out)
            # 线性，输出
            x = concatenate([max, avg], axis=-1)
            x = Dropout(0.1)(x)
            x = Dense(100, activation='relu')(x)
            x = Dense(100, activation='relu')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=score)
            model.summary()

            return model

        if args.model_type == 'freeze':
            '''
                同上。冻结bert参数
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten
            import transformers

            from transformers import BertTokenizer, TFBertModel
            path = 'Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)
            ### 冻结参数
            # 全部冻结参数
            # for k, v in bert_model._get_trainable_state().items():
            #     k.trainable = False
            # bert_model.summary()

            # 冻结embeddings参数
            # for layer in bert_model.layers[:]:
            #     if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            #         layer.embeddings.trainable = False
            # bert_model.summary()
            # 冻结encoder部分参数
            # for layer in bert_model.layers[:]:
            #     if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            #         for idx, layer in enumerate(layer.encoder.layer):
            #             # if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            #             if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11]:
            #                 layer.trainable = False
            # bert_model.summary()

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

            x2 = bert_pooler_output

            # 两个输入进行拼接
            out = concatenate([x1, x2], axis=-1)
            # 双池化
            max = GlobalMaxPooling1D()(out)
            avg = GlobalAveragePooling1D()(out)
            # 线性，输出
            x = concatenate([max, avg], axis=-1)
            x = Dropout(0.1)(x)
            x = Dense(100, activation='relu')(x)
            x = Dense(100, activation='relu')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=score)
            model.summary()

            return model

        if args.model_type == 'only_input':
            '''
                同上。冻结bert参数
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten
            import transformers

            from transformers import BertTokenizer, TFBertModel
            path = 'Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)
            ### 冻结参数
            # 全部冻结参数
            for k, v in bert_model._get_trainable_state().items():
                k.trainable = False
            bert_model.summary()

            # 冻结embeddings参数
            # for layer in bert_model.layers[:]:
            #     if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            #         layer.embeddings.trainable = False
            # bert_model.summary()
            # # 冻结encoder部分参数
            # for layer in bert_model.layers[:]:
            #     if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            #         for idx, layer in enumerate(layer.encoder.layer):
            #             # if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            #             if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11]:
            #                 layer.trainable = False
            bert_model.summary()

            # 第一个输入
            input1 = Input(shape=(overal_maxlen,), dtype='int32')
            # emb_out1.shape == (None,600,300)
            # emb_out1 = Embedding(args.vocab_size, args.emb_dim, name='emb')(input1)
            # # args.emb_dim == 300
            # # 维度降低
            # emb_out1 = Embedding(args.vocab_size, 50, name='emb')(input1)

            # # trm_out.shape == (None,600,300)
            # # output_dim == emb_dim == heads * head_size == 3 *100 == 300
            # x1 = MultiHeadAttention(3, 100)(emb_out1)
            # x1 = MultiHeadAttention(2, 25)(emb_out1)

            # 第二个输入
            # shape == (None,600)
            input_ids = Input(shape=(overal_maxlen,), dtype='int32')
            input_mask = Input(shape=(overal_maxlen,), dtype='int32')
            input_tokentype = Input(shape=(overal_maxlen,), dtype='int32')
            dim_out2 = bert_model(
                {"input_ids": input_ids, "token_type_ids": input_tokentype, "attention_mask": input_mask})
            # bert_last_hidden_output = dim_out2.last_hidden_state
            bert_pooler_output = dim_out2.last_hidden_state

            x2 = bert_pooler_output

            # 不拼接了，只用随机初始化
            # # 两个输入进行拼接
            # out = concatenate([x1, x2], axis=-1)
            out = x2

            # 双池化
            max = GlobalMaxPooling1D()(out)
            avg = GlobalAveragePooling1D()(out)
            # 线性，输出
            x = concatenate([max, avg], axis=-1)
            x = Dropout(0.1)(x)
            x = Dense(100, activation='elu')(x)
            x = Dense(100, activation='elu')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=score)
            model.summary()

            return model

        if args.model_type == 'test':
            '''
                同上。冻结bert参数
            '''
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                Concatenate, Convolution1D, Dropout, MaxPool1D \
                , LSTM, LeakyReLU, concatenate, Flatten
            import transformers

            from transformers import BertTokenizer, TFBertModel
            path = 'Pre-training/BERT_base'
            bert_model = TFBertModel.from_pretrained(path)
            ### 冻结参数
            # 全部冻结参数
            # for k, v in bert_model._get_trainable_state().items():
            #     k.trainable = False
            # bert_model.summary()

            #######   冻结11层
            # 冻结embeddings参数
            for layer in bert_model.layers[:]:
                if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
                    layer.embeddings.trainable = False
            # bert_model.summary()
            # 冻结encoder部分参数
            for layer in bert_model.layers[:]:
                if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
                    for idx, layer in enumerate(layer.encoder.layer):
                        # if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                        if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                            layer.trainable = False
            bert_model.summary()

            # 第一个输入
            input1 = Input(shape=(overal_maxlen,), dtype='int32')
            # emb_out1.shape == (None,600,300)
            emb_out1 = Embedding(args.vocab_size, args.emb_dim, name='emb')(input1)

            # trm_out.shape == (None,600,300)
            x1 = MultiHeadAttention(3, 100)(emb_out1)
            # one_merge = Dense(300, activation='elu', kernel_initializer='he_normal')(x1)
            # two_merge = Dense(200, activation='elu', kernel_initializer='he_normal')(one_merge)
            # three_merge = Dense(100, activation='elu', kernel_initializer='he_normal')(two_merge)
            # # concate
            # concat = concatenate([one_merge, two_merge, three_merge], axis=-1)
            # x1= concat

            '''
            one_merge=Dense(512,activation='elu',kernel_initializer='he_normal')(concat)
            two_merge=Dense(256,activation='elu',kernel_initializer='he_normal')(one_merge)
            three_merge=Dense(64,activation='elu',kernel_initializer='he_normal')(two_merge)
            #concate
            concat = concatenate([one_merge,two_merge,three_merge], axis=-1)
            '''

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

            # # 分别双池化，然后门控，然后全连接
            # max_x1 = GlobalMaxPooling1D()(x1)
            # avg_x1 = GlobalAveragePooling1D()(x1)
            # # x1.shape == (None,300+300) == (None,600)
            # x1 = concatenate([max_x1,avg_x1],axis=-1)
            #
            # max_x2 = GlobalMaxPooling1D()(x2)
            # avg_x2 = GlobalAveragePooling1D()(x2)
            # # x2.shape == (None,768+768) == (None,1536)
            # x2 = concatenate([max_x2, avg_x2], axis=-1)
            #
            # # x3:x1和x2两个双池化一拼(上下),x3.shape == (None,1536+600) == (None,2136)
            # x_feature = concatenate([x1,x2],axis=-1)
            #
            # matrix = Dense(2136,activation='sigmoid')(input1)
            #
            # x = x_feature * matrix

            #
            # # 两个输入进行拼接
            # out = concatenate([x1, x2], axis=-1)
            # # 双池化
            # max = GlobalMaxPooling1D()(out)
            # avg = GlobalAveragePooling1D()(out)
            # # 线性，输出
            # x = concatenate([max, avg], axis=-1)

            x = Dropout(0.1)(x)
            x = Dense(100, activation='swish')(x)
            x = Dense(100, activation='swish')(x)
            x = Dropout(0.1)(x)

            score = Dense(self.num_outpus, activation='sigmoid')(x)

            model = Model(inputs=[input1, input_ids, input_mask, input_tokentype], outputs=score)
            model.summary()

            return model

        if args.model_type == 'Ablation_experiment':
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
            ### 冻结参数
            # 全部冻结参数
            # for k, v in bert_model._get_trainable_state().items():
            #     k.trainable = False
            # bert_model.summary()

            #######   冻结10层
            # 冻结embeddings参数
            for layer in bert_model.layers[:]:
                if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
                    layer.embeddings.trainable = False
            bert_model.summary()
            # 冻结encoder部分参数
            for layer in bert_model.layers[:]:
                if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
                    for idx, layer in enumerate(layer.encoder.layer):
                        # if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
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
