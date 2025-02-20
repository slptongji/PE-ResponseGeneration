import argparse
import os
import sys
import keras
import sklearn

from keras.optimizers import adam_v2
from sklearn.metrics import classification_report
from tqdm import tnrange

from data_utils.prepare_dialogue_data import read_training_file, read_total_embeddings, get_word_count, \
    read_emotion_words, construct_vocab, construct_word_dict, get_word_list, read_stop_words, read_word_embeddings, \
    read_emotion_label, filter_sentence_length, align_sentence_length, get_predict_train_response_data, \
    align_batch_size, get_train_response_data, read_personality_file, filter_sentence_length_peld, \
    align_peld_batch_size, filter_test_peld_sentence_length, align_test_peld_batch_size
from peld_src.dataload import load_data,load_test_data
import tensorflow.compat.v1 as tf
from src.configuration import ChatConfig
from tensorflow import losses
from peld_src.utils import Emotion_dict
import numpy as np
import pandas as pd
# 是model.py里面的Emo_Generation类
from peld_src.model import Emo_Generation
from keras.optimizers import adam_v2
from transformers import RobertaConfig, RobertaModel, PreTrainedModel, get_linear_schedule_with_warmup
# 是tain.py里面的train_model函数
# from train import train_model

device = tf.device("cpu")


# CONFIG
# 配置项目

# 数据路径
DATA_PATH = '../Dyadic_all.tsv'
TEST_DATA_PATH = '../Dyadic_all.tsv'


# parse.add_argument("--result_name",type=str,default=args.Senti_or_Emo+'_Mode_'+str(args.mode)+'_'+args.loss_function+'_Epochs_'+str(args.epochs)+'.csv')

parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.mode = 3
args.Senti_or_Emo = 'Emotion'
args.loss_function = 'CE' # CE or MSE or Focal
args.device = 0
args.SEED = 42
args.MAX_LEN = 64
args.batch_size = 64
args.lr = 1e-5
args.adam_epsilon = 1e-8
args.epochs = 5
args.result_name = args.Senti_or_Emo+'_Mode_'+str(args.mode)+'_'+args.loss_function+'_Epochs_'+str(args.epochs)+'.csv'

def train(config_file, pre_train_word_count_file, emotion_words_dir,embedding_file,data_path,test_data_path, session, checkpoint_dir, max_vocab_size,log_name):
    # chat_config = ChatConfig(config_file)
    #
    # print("Now prepare data!\n")
    # print("Read stop words!\n")
    # stop_words = read_stop_words(FLAGS.stop_words_file)
    # # 构造通用词汇的词汇表
    # print("Construct vocab first\n")
    # # 得到词嵌入（float组成的向量），词嵌入中词的index，词嵌入中所有的词
    # total_embeddings, total_word2id, total_word_list = read_total_embeddings(embedding_file, max_vocab_size)
    # #[情感词]=word count
    # pre_word_count = get_word_count(pre_train_word_count_file, chat_config.word_count)
    # # 构造好的情感词典dic[情感类别代号][情感词]=情感词word_count?
    # emotion_words_dict = read_emotion_words(emotion_words_dir, pre_word_count)
    # # 处理情感词和通用词，得到全部词的list
    # word_list = construct_vocab(total_word_list, emotion_words_dict, chat_config.generic_word_size,
    #                             chat_config.emotion_vocab_size, FLAGS.unk)
    # # 全部词的词典，0是unk,1是start letter,2是end letter
    # word_dict = construct_word_dict(word_list, FLAGS.unk, FLAGS.start_symbol, FLAGS.end_symbol)
    # # 根据index来查Word
    # id2words = {idx: word for word, idx in word_dict.items()}
    # # 其实word_unk_id就是0
    # word_unk_id = word_dict[FLAGS.unk]
    # # word_start_id是1
    # word_start_id = word_dict[FLAGS.start_symbol]
    # # word_end_id是2
    # word_end_id = word_dict[FLAGS.end_symbol]
    # final_word_list = get_word_list(id2words)
    #
    # print("Read word embeddings!\n")
    # # 读所有的词向量
    # embeddings = read_word_embeddings(total_embeddings, total_word2id, final_word_list, chat_config.embedding_size)

    # Read the training file
    train_length, train_dataloader, valid_dataloader= load_data(args, DATA_PATH)
    test_dataloader = load_test_data(args, DATA_PATH)
    args.train_length = train_length

    model_train(args, pre_train_word_count_file, emotion_words_dir,embedding_file, session,checkpoint_dir, max_vocab_size,log_name,train_dataloader, valid_dataloader, test_dataloader)

def vad_to_emo(emotion, Emotion_dict):
    label_list = []
    for emo in list(emotion):
        min_index = 0
        min_mse = 1000
        cnt = 0
        for k,v in Emotion_dict.items():
            mse = sklearn.metrics.mean_squared_error(list(emo), v)
            if mse < min_mse:
                min_mse = mse
                min_index = cnt
            cnt += 1
        label_list.append(min_index)
    return np.array(label_list)

def model_train(args, pre_train_word_count_file, emotion_words_dir,  embedding_file, session,
                checkpoint_dir, max_vocab_size,
                log_name,train_dataloader, valid_dataloader, test_dataloader):
    # Start training
    num_warmup_steps = int(0.05 * args.train_length)
    # 整个训练过程的总步数=训练集数量*epoch个数
    num_training_steps = len(train_dataloader) * args.epochs
    # # 模型训练时的优化器，lr是学习率
    # optimizer = adam_v2.Adam(learning_rate=args.lr,epsilon=args.adam_epsilon)  # To reproduce BertAdam specific behavior set correct_bias=False
    # # 学习率预热，在预热期间，从0慢慢增加到adamW中的学习率，在预热阶段之后创建一个schedule，使其学习率从优化器中的初始lr线性降低到0
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,num_training_steps=num_training_steps)  # PyTorch scheduler
    optimizer=adam_v2.Adam(lr=1e-5)
    train_logs = []
    valid_logs = []
    test_logs  = []

    model = Emo_Generation.from_pretrained('roberta-base', mode=3)
    model.zero_grad()

    # num_train_batch = int(len(train_utt1) / chat_config.batch_size)
    # emotion_vocab_size = chat_config.emotion_vocab_size
    # num_generic_word=chat_config.generic_word_size+3
    # # 读取训练的epoch数量
    # train_epochs = chat_config.epochs
    print ("check here")
    for _ in tnrange(1, args.epochs +1, desc='Epoch'):
        print("<" + "=" * 22 + F" Epoch {_} " + "=" * 22 + ">")
        # Calculate total loss for this epoch
        batch_loss = 0
        train_accuracy, nb_train_steps = 0, 0

        pred_list = np.array([])
        labels_list = np.array([])
        # 打乱数据没做
        # 开始batch的循环
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t for t in batch)
            b_input_ids_1, b_input_ids_2, b_input_ids_3, b_personality, b_init_emo, b_response_emo, b_labels = batch
            logits = model(b_input_ids_1, b_input_ids_2, b_input_ids_3, personality=b_personality, init_emo=b_init_emo)
            # 均方误差，参数估计值与参数真值之差平方的期望值，也就是简单的预测值和真实值之间的差的平方
            if args.loss_function == 'MSE':
                loss = losses.mean_squared_error(logits, b_response_emo)
                # 返回的就是一个新的tensor，但是没有梯度了
                pred_emotion = logits.detach().to('cpu').detach().numpy()
                # 模型得到的预测情感
                pred_flat = vad_to_emo(pred_emotion, Emotion_dict).flatten()
                # 数据集中的真实情感
                labels_flat = vad_to_emo(b_response_emo.to('cpu'), Emotion_dict).flatten()
            else:
                # 交叉熵损失函数

                #计算损失
                loss= tf.nn.sigmoid_cross_entropy_with_logits(logits, b_labels)
                logits= logits.detach().to('cpu').detach().numpy()
                label_ids= b_labels.to('cpu').detach().numpy()
                # 从logits里选可能性最大的那个情感,flatten返回一个折叠成一维的数组
                pred_flat   = np.argmax(logits, axis=1).flatten()
                labels_flat = label_ids.flatten()


            # 把这一次的batch得到的prediction的情感放入list
            pred_list   = np.append(pred_list, pred_flat)
            labels_list = np.append(labels_list, labels_flat)
            df_metrics  = pd.DataFrame({'Epoch':args.epochs,'Actual_class':labels_flat,'Predicted_class':pred_flat})

            nb_train_steps += 1
            '''
            # Backward pass
            loss.backward()

            # Clip the norm of the gradients to 1.0
            # Gradient clipping is not in AdamW anymore
            # 用梯度下降更新各类参数
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()
            '''
            tvars=model.get_vars()
            all_grads=tf.gradients(loss,tvars)
            grads,_=tf.clip_by_global_norm(all_grads,1)
            optimizer.apply_gradients(zip(grads,tvars))

            # # Update learning rate schedule
            # scheduler.step()

            # Clear the previous accumulated gradients
            optimizer.zero_grad()

            # Update tracking variables
            batch_loss += loss.item()

        #  Calculate the average loss over the training data.
        avg_train_loss = batch_loss / len(train_dataloader)

        # store the current learning rate
        for param_group in optimizer.param_groups:
            print("\n\tCurrent Learning rate: ", param_group['lr'])

        # 此函数用于显示主要分类指标的文本报告，返回的是精确度、召回值和F1值
        result = classification_report(pred_list, labels_list, digits=4, output_dict=True)
        # 将结果的准确性等等输出到train_log中
        for key in result.keys():
            if key != 'accuracy':
                try:
                    train_logs.append([
                        labelencoder.classes_[int(eval(key))],
                        result[key]['precision'],
                        result[key]['recall'],
                        result[key]['f1-score'],
                        result[key]['support']
                    ])
                except:
                    train_logs.append([
                        key,
                        result[key]['precision'],
                        result[key]['recall'],
                        result[key]['f1-score'],
                        result[key]['support']
                    ])

        # 进行验证和测试，和train基本一样，只是损失函数用的交叉熵函数，同时没有更新参数
        valid_logs = eval_model(model, valid_dataloader, args, valid_logs)
        test_logs = test_model(model, test_dataloader, args, test_logs)

    df_train_logs = pd.DataFrame(train_logs,
                                 columns=['label', 'precision', 'recall', 'f1-score', 'support']).add_prefix('train_')
    df_valid_logs = pd.DataFrame(valid_logs, columns=['precision', 'recall', 'f1-score', 'support']).add_prefix(
        'valid_')
    df_test_logs = pd.DataFrame(test_logs, columns=['precision', 'recall', 'f1-score', 'support']).add_prefix('test_')

    # 将所有的拼成dataframe，
    df_all = pd.concat([df_train_logs, df_valid_logs, df_test_logs], axis=1)
    df_all.to_csv(args.result_name, index=False)


def eval_model(model, valid_dataloader, args, valid_logs):
    # Validation

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    pred_list = np.array([])
    labels_list = np.array([])

    for batch in valid_dataloader:
        batch = tuple(t.cuda(args.device) for t in batch)
        b_input_ids_1, b_input_ids_2, b_input_ids_3, b_personality, b_init_emo, b_response_emo, b_labels = batch

        # Forward pass, calculate logit predictions
        logits = model(b_input_ids_1, b_input_ids_2, b_input_ids_3, personality=b_personality, init_emo=b_init_emo)

        # if MSE:
        # loss_fct = nn.MSELoss()
        # loss = loss_fct(logits, b_response_emo)

        # if cross entropy:

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, b_labels)
        logits = logits.detach().to('cpu').detach().numpy()
        label_ids = b_labels.to('cpu').detach().numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()

        # if MSE:
        # pred_emotion = logits.detach().to('cpu').numpy()
        # pred_flat = vad_to_emo(pred_emotion, Emotion_dict).flatten()
        # labels_flat = vad_to_emo(b_response_emo.to('cpu'), Emotion_dict).flatten()

        pred_list = np.append(pred_list, pred_flat)
        labels_list = np.append(labels_list, labels_flat)
    result = classification_report(pred_list, labels_list, digits=4, output_dict=True)
    for key in result.keys():
        if key != 'accuracy':
            valid_logs.append([
                result[key]['precision'],
                result[key]['recall'],
                result[key]['f1-score'],
                result[key]['support']
            ])
    return valid_logs


def test_model(model, test_dataloader, args, test_logs):
    # Test

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    pred_list = np.array([])
    labels_list = np.array([])

    for batch in test_dataloader:
        batch = tuple(t.cuda(args.device) for t in batch)
        b_input_ids_1, b_input_ids_2, b_input_ids_3, b_personality, b_init_emo, b_response_emo, b_labels = batch

        # Forward pass, calculate logit predictions
        logits = model(b_input_ids_1, b_input_ids_2, b_input_ids_3, personality=b_personality, init_emo=b_init_emo)

        # if MSE:
        # loss_fct = nn.MSELoss()
        # loss = loss_fct(logits, b_response_emo)

        # if cross entropy:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, b_labels)
        logits = logits.detach().to('cpu').detach().numpy()
        label_ids = b_labels.to('cpu').detach().numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()

        # if MSE:
        # pred_emotion = logits.detach().to('cpu').numpy()
        # pred_flat = vad_to_emo(pred_emotion, Emotion_dict).flatten()
        # labels_flat = vad_to_emo(b_response_emo.to('cpu'), Emotion_dict).flatten()

        pred_list = np.append(pred_list, pred_flat)
        labels_list = np.append(labels_list, labels_flat)

    result = classification_report(pred_list, labels_list, digits=4, output_dict=True)
    for key in result.keys():
        if key != 'accuracy':
            test_logs.append([
                result[key]['precision'],
                result[key]['recall'],
                result[key]['f1-score'],
                result[key]['support']
            ])
    return test_logs

def main(_):
    with tf.device("/gpu:0"):
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))
        train(FLAGS.config_file, FLAGS.pre_train_word_count_file, FLAGS.emotion_words_dir,FLAGS.embedding_file, FLAGS.data_path,FLAGS.test_data_path ,sess,FLAGS.checkpoint_dir, FLAGS.max_vocab_size, FLAGS.log_name)



if __name__ == '__main__':
    model_path = os.path.dirname(os.path.dirname(os.path.abspath("tf_main.py")))
    data_dir = os.path.join(model_path, "data")
    parse = argparse.ArgumentParser()

    # Add the config of the model
    parse.add_argument("--config_file", type=str, default=os.path.join(model_path, "conf/peld.conf"),
                       help="configuration file path")
    parse.add_argument("--loss_function",type=str,default='CE',help="Chooose the loss function to use")
    parse.add_argument("--device",type=int,default=0,help="Choose the device to use")
    parse.add_argument("--data_path",type=str,default="Dyadic_PELD.tsv")
    parse.add_argument("--test_data_path", type=str, default="Dyadic_all.tsv")
    parse.add_argument("--embedding_file", type=str,
                       default=os.path.join(data_dir, "embedding/7_classes_trans_metric.txt"),
                       help="word embedding file path")
    parse.add_argument("--unk", type=str, default="</s>", help="symbol for unk words")
    parse.add_argument("--start_symbol", type=str, default="<ss>", help="symbol for response sentence start")
    parse.add_argument("--end_symbol", type=str, default="<es>", help="symbol for response sentence end")
    parse.add_argument("--checkpoint_dir", type=str, default=os.path.join(model_path, "data_utils/check_path"),
                       help="saving checkpoint directory")
    parse.add_argument("--stop_words_file", type=str,
                       default=os.path.join(data_dir, "stop_words/stop-word-zh.txt"),
                       help="stop word file path")
    parse.add_argument("--max_vocab_size", type=int, default=50000, help="maximum vocabulary size")
    parse.add_argument("--log_name", type=str, default="dialogue", help="log file name")
    parse.add_argument("--restore_model", type=str, default="dialogue-model-0",
                       help="name of restore model from checkpoints")
    parse.add_argument("--pre_train_word_count_file", type=str,
                       default=os.path.join(data_dir, "emotion_words_human/word.count.7.120.CN.txt"),
                       help="nlp cc word count file")
    parse.add_argument("--emotion_words_dir", type=str,
                       default=os.path.join(data_dir, "emotion_words_human/7_class_CN_120"),
                       help="emotion words directory")


    FLAGS, unparsed = parse.parse_known_args()
    # tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)

    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)








