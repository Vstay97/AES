# 用法: cd /data1/songchao/My_newModel/
# 切换到 run.sh 文件目录下,输入: sh run.sh
# 结构说明:
# nohup : 后台运行
# /home/admin123/miniconda3/envs/songchao-tf/bin/python : 所使用的conda虚拟环境解释器
# main.py : 运行的py文件名称
# -tr data/fold_0/train.tsv .... : main.py所用的参数
# > filelog.txt 2>&1 & : 将控制台的输出写入到 filelog.txt 文件中
nohup /home/admin123/miniconda3/envs/songchao-tf/bin/python main.py -tr data/fold_0/train.tsv -tu=data/fold_0/dev.tsv -ts=data/fold_0/test.tsv --emb ./SWEM/glove.6B.300d.txt --vocab-path output_dir/vocab.pkl --tsp data/training_set_rel3.xls --pp data/prompt.xlsx --swp data/stopword.txt -o output_dir -b=2 -p=0 -g=1 -lr=1e-5 -t ChinaAI --epochs=20 --explain="冻结emb+10" > filelog.txt 2>&1 &