#!/usr/bin/env python
import logging
import random
import numpy as np
from time import time
import os
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import sequence
from Pre_utils.bert_setting import BertPreSetting
from Pre_utils.data_process import DatasetProcess
from config import Config
from models import ModelsConfigs, Models
from Evaluate.evaluator import Evaluator

# 查看可用设备
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 使用第一, 三块GPU
# 最好的验证性能
best_result_fold = []
# 平均性能
mean_fold = []

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
#######################################################################################################################
## Parse arguments
config = Config()
args = config.get_parser()

out_dir = args.out_dir_path

prompts = [1, 2, 3, 4, 5, 6, 7, 8]
datas = [0]
# prompts = [1,2,3]
# datas = [0,1]

prompt = config.get_prompt()

# 第i折的训练
for i in datas:
    # 最好的数据集性能
    best_result_data = []
    # 第j个数据
    for j in prompt:
        logger.info('################################################# ' + 'this is training: fold ' + str(
            i) + ' and prompt' + str(j) + ' #################################################')
        args.prompt_id = j
        # 指定数据集的路径
        args.train_path = 'data/fold_' + str(i) + '/train.tsv'
        args.dev_path = 'data/fold_' + str(i) + '/dev.tsv'
        args.test_path = 'data/fold_' + str(i) + '/test.tsv'

        # # 3090专用
        # if prompt == 8:
        #     args.batch_size = 8
        # elif prompt == 3:
        #     args.batch_size = 16
        # elif prompt == 4:
        #     args.batch_size = 16
        # elif prompt == 5:
        #     args.batch_size = 16
        # else:
        #     args.batch_size = 8




        # 设定numpy随机种子
        if args.seed > 0:
            random.seed(args.seed)  # 为python设置随机种子
            np.random.seed(args.seed)  # 为numpy设置随机种子
            tf.random.set_seed(args.seed)  # tf cpu fix seed
            # 使用GPU，需要pip install tensorflow-determinism
            os.environ['TF_DETERMINISTIC_OPS'] = '1'

        # 设定文本长度
        maxlen_array = [600, 600, 300, 300, 300, 400, 500, 800]
        overal_maxlen = maxlen_array[args.prompt_id - 1]

        # 获得数据
        get_Data = DatasetProcess(args)

        # 准备第一组数据
        (train_x, train_y, train_pmt), \
        (dev_x, dev_y, dev_pmt), \
        (test_x, test_y, test_pmt), \
        vocab, vocab_size, num_outputs \
            = get_Data.get_data(args, args.vocab_size, tokenize_text=True, vocab_path=args.vocab_path)

        # 对序列长度进行padding,转为ndarray格式
        train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
        dev_x = sequence.pad_sequences(dev_x, maxlen=overal_maxlen)
        test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)

        train_y = np.array(train_y, dtype=K.floatx())
        dev_y = np.array(dev_y, dtype=K.floatx())
        test_y = np.array(test_y, dtype=K.floatx())

        train_pmt = np.array(train_pmt, dtype='int32')
        dev_pmt = np.array(dev_pmt, dtype='int32')
        test_pmt = np.array(test_pmt, dtype='int32')

        # 准备第二组数据
        (train_pre_x, train_pre_y, train_pre_prompts), (dev_pre_x, dev_pre_y, dev_pre_prompts), (
            test_pre_x, test_pre_y, test_pre_prompts), vocab, vocab_size = get_Data.get_pre_data(args)
        # 获得第二组数据
        bert_inputs = BertPreSetting(args, overal_maxlen)
        inputs_train_ids, inputs_train_mask, inputs_train_tokentype = bert_inputs.get_inputs(args, train_pre_x)
        inputs_dev_ids, inputs_dev_mask, inputs_dev_tokentype = bert_inputs.get_inputs(args, dev_pre_x)
        inputs_test_ids, inputs_test_mask, inputs_test_tokentype = bert_inputs.get_inputs(args, test_pre_x)

        # 计算均值和标准差
        train_mean = train_y.mean(axis=0)  # 计算每一列的均值
        train_std = train_y.std(axis=0)  # 计算每一列的标准差
        dev_mean = dev_y.mean(axis=0)
        dev_std = dev_y.std(axis=0)
        test_mean = test_y.mean(axis=0)
        test_std = test_y.std(axis=0)

        # 我们需要原始规模的开发和测试集进行评估
        dev_y_org = dev_y.astype(get_Data.get_ref_dtype())
        test_y_org = test_y.astype(get_Data.get_ref_dtype())

        # 将分数转换为 [0 1] 的边界以进行训练和评估（损失计算）
        train_y = get_Data.get_model_friendly_scores(train_y, train_pmt)
        dev_y = get_Data.get_model_friendly_scores(dev_y, dev_pmt)
        test_y = get_Data.get_model_friendly_scores(test_y, test_pmt)

        # 建立模型
        creat_model = Models(args, train_y.mean(axis=0), overal_maxlen, vocab)
        models_config = ModelsConfigs()
        # optimizer = models_config.get_optimizer(args)
        loss, metric = models_config.get_loss_metric(args)

        model = creat_model.get_model(args, overal_maxlen, vocab)
        # model.compile(optimizer='rmsprop', loss=loss, metrics=metric)
        # model.compile(optimizer='adam', loss=loss, metrics=metric)
        model.compile(optimizer='Nadam', loss=loss, metrics=metric)


        # 评价指标
        evl = Evaluator(get_Data, args, out_dir, dev_x, inputs_dev_ids, inputs_dev_mask, inputs_dev_tokentype, test_x,
                        inputs_test_ids, inputs_test_mask, inputs_test_tokentype, dev_y, test_y, dev_y_org, test_y_org)

        # 训练
        total_train_time = 0
        total_eval_time = 0
        t1 = time()

        for epoch in range(args.epochs):
            # Training
            t0 = time()
            train_history = model.fit([train_x, inputs_train_ids, inputs_train_mask, inputs_train_tokentype], train_y,
                                      batch_size=args.batch_size, epochs=1, verbose=0)
            tr_time = time() - t0
            total_train_time += tr_time

            # Evaluate
            t0 = time()
            evl.evaluate(model, epoch)
            evl_time = time() - t0
            total_eval_time += evl_time
            total_time = time() - t1

            # Print information
            logger.info(
                'Epoch %d, train: %is, evaluation: %is, total_time: %is' % (epoch, tr_time, evl_time, total_time))
            train_loss = train_history.history['loss'][0]
            train_metric = train_history.history[metric][0]
            logger.info('[Train] loss: %.4f, metric: %.4f' % (train_loss, train_metric))
            evl.print_info()

        # 结果总结
        logger.info('Training:   %i seconds in total' % total_train_time)
        logger.info('Evaluation: %i seconds in total' % total_eval_time)

        evl.print_final_info()

        best_statistics = evl.get_best_statistics()
        best_result_data.append(best_statistics)
        logger.removeHandler(logging.StreamHandler)

    best_result_fold.append(best_result_data)

mean_fold.append(np.mean(best_result_fold, axis=0))

import time

print(np.mean(best_result_fold, axis=0))
with open("实验数据.txt", "a+", encoding="utf-8") as f:
    f.seek(0)
    f.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + '：'+args.model_type+'，'+args.explain+'\n')
    f.write(str(best_result_fold))
    f.write("\n")
    f.write('均值：')
    f.write(str(mean_fold))
    f.write("\n")
    f.write("平均QWK：" + str(np.mean(mean_fold)))
    f.write("\n")
    f.write("\n")
