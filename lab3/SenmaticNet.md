# ��Ȼ���Դ����Ӧ�á�������mindspore����з���ʵ��

## ����

��з�������Ȼ���Դ������ı�����������Ӽ���������Ȼ���Դ����������Ӧ�á����ǶԴ��и���ɫ�ʵ��������ı����з���������Ĺ��̣�������˵���˵�̬�ȣ����������滹�Ƿ��档

> ͨ������£����ǻ���������Ϊ���桢������������ࡣ��Ȼ�����ޱ��顱������Ҳ�в��٣��������󲿷�ʱ���ֻ��������ͷ���İ�������ѵ��������������ݼ����Ǻܺõ����ӡ�

��ͳ���ı������������ĵ��Ͳο����ݼ�Ϊ[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)�������ݼ���20������������ɣ�����Լ20000�������ĵ��� �������б�����Щ�������ݱȽ����ƣ�����comp.sys.ibm.pc.hardware��comp.sys.mac.hardware���Ǻ͵���ϵͳӲ����ص���Ŀ�����ƶȱȽϸߡ�����Щ�����������������˵�ͺ��޹���������misc.forsale��soc.religion.christian��

�����籾����ԣ��ı�������������ṹ����з��������ṹ�������ơ�����������з���������ι���֮�󣬺����׿��Թ���һ�����Ƶ����磬�������μ��������ı������������

����ҵ�������Ĳ࣬�ı���������Ƿ����ı����۵Ŀ͹����ݣ�����з�����Ҫ���ı��еõ����Ƿ�֧��ĳ�ֹ۵����Ϣ�����磬�����������������Ǻÿ����ˣ�ӰƬ������ȷ����������������仰�����ı����������Ҫ�����Ϊ���Ϊ����Ӱ�����⣬����з�����Ҫ�ھ����һӰ����̬�������滹�Ǹ��档

����ڴ�ͳ���ı�������࣬��з����Ϊ�򵥣�ʵ����Ҳ��ǿ�������Ĺ�����վ����Ӱ��վ�����Բɼ�����Ը����������ݼ���Ҳ�����׸�ҵ������������档���磬���Խ�����������ģ��Զ������ض����Ϳͻ��Ե�ǰ��Ʒ����������Է�������û����Ͷ���н��з�������������ԵĴ����������ڴ˽�һ���Ƽ���Ʒ�����ת���ʣ��������ߵ���ҵ���档

���������У�ĳЩ�Ǽ��Դ�Ҳ��ֱ�����û���������򣬱�������ʹ��APPʱ���������ˡ���������̫���ˡ��ͱ�����û��ĸ���������򣻹�Ʊ�����У������ǡ�����ţ�С����ľ����û�����������������ԣ������ϣ�����ϣ��ģ���ܹ��ڴ�ֱ�����У��ھ��һЩ����ı���Ϊ���Դʸ���з���ϵͳʹ�ã�

��ֱ���Դ�=ͨ�ü��Դ�+�������м��Դʴ�ֱ���Դ�=ͨ�ü��Դ�+�������м��Դ�

���մ����ı������Ȳ�ͬ����з����ɷ�Ϊ���Ｖ�����Ｖ�����Ӽ������伶�Լ�ƪ�¼��ȼ����о���Ρ������ԡ����伶��Ϊ��������Ϊһ�����䣬���ΪӰ�������滹�Ǹ������Ϣ��

����ʵ�飬��IMDBӰ����з�������MindSpore����Ȼ���Դ����ϵ�Ӧ�á�

## ��������

1. ׼�����ڡ�
2. �������ݼ����������ݴ���
3. �������硣
4. �����Ż�������ʧ������
5. ʹ������ѵ�����ݣ�����ģ�͡�
6. �õ�ģ��֮��ʹ����֤���ݼ����鿴ģ�;��������

## ׼������

### ���ݼ�

����ʵ�����IMDBӰ�����ݼ���Ϊʵ�����ݡ�

1. ����[IMDBӰ�����ݼ�](http://ai.stanford.edu/~amaas/data/sentiment/)��

   �����Ǹ���Ӱ����Negative��������Ӱ����Positive���İ�����

| Review                                                       | Label    |
| ------------------------------------------------------------ | -------- |
| "Quitting" may be as much about exiting a pre-ordained identity as about drug withdrawal. As a rural guy coming to Beijing, class and success must have struck this young artist face on as an appeal to separate from his roots and far surpass his peasant parents' acting success. Troubles arise, however, when the new man is too new, when it demands too big a departure from family, history, nature, and personal identity. The ensuing splits, and confusion between the imaginary and the real and the dissonance between the ordinary and the heroic are the stuff of a gut check on the one hand or a complete escape from self on the other. | Negative |
| This movie is amazing because the fact that the real people portray themselves and their real life experience and do such a good job it's like they're almost living the past over again. Jia Hongsheng plays himself an actor who quit everything except music and drugs struggling with depression and searching for the meaning of life while being angry at everyone especially the people who care for him most. | Positive |

&emsp;&emsp;�����غõ����ݼ���ѹ�����ڵ�ǰ����Ŀ¼�µ�`datasets`Ŀ¼�£�ÿ��ѹ1000���ļ����ڵײ�׷�Ӵ�ӡһ���ڵ㡣

?       ����[GloVe�ļ�](http://ai.stanford.edu/~amaas/data/sentiment/) ���ز���ѹGloVe�ļ�����ǰ����Ŀ¼�µ�`datasets`Ŀ¼�£���������Glove�ļ���ͷ�����������ʾ�µ�һ�У���˼���ܹ���ȡ400000�����ʣ�ÿ��������300γ�ȵĴ�������ʾ��

```shell
400000 300
```

�����ݼ���ѹ����ǰ����Ŀ¼�£������ṹ������ʾ��

> ������ ckpt
>
> ������ datasets
>
> ��   ������ aclImdb
>
> ��   ��   ������ imdbEr.txt
>
> ��   ��   ������ imdb.vocab
>
> ��   ��   ������ README
>
> ��   ��   ������ test
>
> ��   ��   ������ train
>
> ��   ������ glove
>
> ��       ������ glove.6B.100d.txt
>
> ��       ������ glove.6B.200d.txt
>
> ��       ������ glove.6B.300d.txt
>
> ��       ������ glove.6B.50d.txt
>
> ������ nlp_application.ipynb
>
> ������ preprocess
>
> 
>
> 7 directories, 10 files

### ȷ�����۱�׼

��Ϊ���͵ķ������⣬��з�������۱�׼���Ա�����ͨ�ķ������⴦�������ľ��ȣ�Accuracy������׼�ȣ�Precision�����ٻ��ʣ�Recall����F_beta������������Ϊ�ο���

���ȣ�????????��=������ȷ��������Ŀ/��������Ŀ���ȣ�Accuracy��=������ȷ��������Ŀ/��������Ŀ

��׼�ȣ�?????????��=������������Ŀ/����Ԥ�����Ϊ���Ե�������Ŀ��׼�ȣ�Precision��=������������Ŀ/����Ԥ�����Ϊ���Ե�������Ŀ

�ٻ��ʣ�??????��=������������Ŀ/������ʵ���Ϊ���Ե�������Ŀ�ٻ��ʣ�Recall��=������������Ŀ/������ʵ���Ϊ���Ե�������Ŀ

?1����=(2?????????????????)/(?????????+??????)F1����=(2?Precision?Recall)/(Precision+Recall)

��IMDB������ݼ��У�������������𲻴󣬿��Լ򵥵��þ��ȣ�accuracy����Ϊ�������ĺ�����׼��

### ȷ������

����ʹ�û���LSTM������SentimentNet���������Ȼ���Դ���

> LSTM��Long short-term memory�������ڼ��䣩������һ��ʱ��ѭ�������磬�ʺ��ڴ����Ԥ��ʱ�������м�����ӳٷǳ�������Ҫ�¼��� ����ʵ������GPU��CPUӲ��ƽ̨��

### ����������Ϣ��SentimentNet�������

1. ʹ��`parser`ģ�鴫�����б�Ҫ����Ϣ��
   - `preprocess`���Ƿ�Ԥ�������ݼ���Ĭ��Ϊ��
   - `aclimdb_path`�����ݼ����·����
   - `glove_path`��GloVe�ļ����·����
   - `preprocess_path`��Ԥ�������ݼ��Ľ���ļ��С�
   - `ckpt_path`��CheckPoint�ļ�·����
   - `pre_trained`��Ԥ����CheckPoint�ļ���
   - `device_target`��ָ��GPU��CPU������
2. ����ѵ��ǰ����Ҫ���ñ�Ҫ����Ϣ������������Ϣ��ִ�е�ģʽ�������Ϣ��Ӳ����Ϣ��

��װ`easydict`��������

```shell
pip install easydict
```

��װ`gensim`��������

```shell
pip install gensim
```

��������һ�δ���������ѵ��������ز�������ϸ�Ľӿ�������Ϣ����μ�MindSpore����`context.set_context`API�ӿ�˵������

```python
import argparse
from mindspore import context
from easydict import EasyDict as edict


# LSTM CONFIG
lstm_cfg = edict({
    'num_classes': 2,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'num_epochs': 10,
    'batch_size': 64,
    'embed_size': 300,
    'num_hiddens': 100,
    'num_layers': 2,
    'bidirectional': True,
    'save_checkpoint_steps': 390,
    'keep_checkpoint_max': 10
})

cfg = lstm_cfg

parser = argparse.ArgumentParser(description='MindSpore LSTM Example')
parser.add_argument('--preprocess', type=str, default='false', choices=['true', 'false'],
                    help='whether to preprocess data.')
parser.add_argument('--aclimdb_path', type=str, default="./datasets/aclImdb",
                    help='path where the dataset is stored.')
parser.add_argument('--glove_path', type=str, default="./datasets/glove",
                    help='path where the GloVe is stored.')
parser.add_argument('--preprocess_path', type=str, default="./preprocess",
                    help='path where the pre-process data is stored.')
parser.add_argument('--ckpt_path', type=str, default="./models/ckpt/nlp_application",
                    help='the path to save the checkpoint file.')
parser.add_argument('--pre_trained', type=str, default=None,
                    help='the pretrained checkpoint file path.')
parser.add_argument('--device_target', type=str, default="GPU", choices=['GPU', 'CPU'],
                    help='the target device to run, support "GPU", "CPU". Default: "GPU".')
args = parser.parse_args(['--device_target', 'CPU', '--preprocess', 'true'])

context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=args.device_target)

print("Current context loaded:\n    mode: {}\n    device_target: {}".format(context.get_context("mode"), context.get_context("device_target")))
```

> Current context loaded:
>
> mode: 0
>
> device_target: CPU

# ���ݴ���

## Ԥ�������ݼ�

ִ�����ݼ�Ԥ����

- ����`ImdbParser`������ı����ݼ����������롢�ִʡ����롢����GloVeԭʼ���ݣ�ʹ֮�ܹ���Ӧ����ṹ��
- ����`convert_to_mindrecord`���������ݼ���ʽת��ΪMindRecord��ʽ������MindSpore��ȡ������`_convert_to_mindrecord`��`weight.txt`Ϊ����Ԥ������Զ����ɵ�weight������Ϣ�ļ���
- ����`convert_to_mindrecord`����ִ�����ݼ�Ԥ����

```python
import os
from itertools import chain
import numpy as np
import gensim
from mindspore.mindrecord import FileWriter
from main import argparse
from main import args
from main import cfg
from main import lstm_cfg

class ImdbParser():
    """
    parse aclImdb data to features and labels.
    sentence->tokenized->encoded->padding->features
    """

    def __init__(self, imdb_path, glove_path, embed_size=300):
        self.__segs = ['train', 'test']
        self.__label_dic = {'pos': 1, 'neg': 0}
        self.__imdb_path = imdb_path
        self.__glove_dim = embed_size
        self.__glove_file = os.path.join(glove_path, 'glove.6B.' + str(self.__glove_dim) + 'd.txt')

        # properties
        self.__imdb_datas = {}
        self.__features = {}
        self.__labels = {}
        self.__vacab = {}
        self.__word2idx = {}
        self.__weight_np = {}
        self.__wvmodel = None

    def parse(self):
        """
        parse imdb data to memory
        """
        self.__wvmodel = gensim.models.KeyedVectors.load_word2vec_format(self.__glove_file)

        for seg in self.__segs:
            self.__parse_imdb_datas(seg)
            self.__parse_features_and_labels(seg)
            self.__gen_weight_np(seg)

    def __parse_imdb_datas(self, seg):
        """
        load data from txt
        """
        data_lists = []
        for label_name, label_id in self.__label_dic.items():
            sentence_dir = os.path.join(self.__imdb_path, seg, label_name)
            for file in os.listdir(sentence_dir):
                with open(os.path.join(sentence_dir, file), mode='r', encoding='utf8') as f:
                    sentence = f.read().replace('\n', '')
                    data_lists.append([sentence, label_id])
        self.__imdb_datas[seg] = data_lists

    def __parse_features_and_labels(self, seg):
        """
        parse features and labels
        """
        features = []
        labels = []
        for sentence, label in self.__imdb_datas[seg]:
            features.append(sentence)
            labels.append(label)

        self.__features[seg] = features
        self.__labels[seg] = labels

        # update feature to tokenized
        self.__updata_features_to_tokenized(seg)
        # parse vacab
        self.__parse_vacab(seg)
        # encode feature
        self.__encode_features(seg)
        # padding feature
        self.__padding_features(seg)

    def __updata_features_to_tokenized(self, seg):
        tokenized_features = []
        for sentence in self.__features[seg]:
            tokenized_sentence = [word.lower() for word in sentence.split(" ")]
            tokenized_features.append(tokenized_sentence)
        self.__features[seg] = tokenized_features

    def __parse_vacab(self, seg):
        # vocab
        tokenized_features = self.__features[seg]
        vocab = set(chain(*tokenized_features))
        self.__vacab[seg] = vocab

        # word_to_idx: {'hello': 1, 'world':111, ... '<unk>': 0}
        word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
        word_to_idx['<unk>'] = 0
        self.__word2idx[seg] = word_to_idx

    def __encode_features(self, seg):
        """ encode word to index """
        word_to_idx = self.__word2idx['train']
        encoded_features = []
        for tokenized_sentence in self.__features[seg]:
            encoded_sentence = []
            for word in tokenized_sentence:
                encoded_sentence.append(word_to_idx.get(word, 0))
            encoded_features.append(encoded_sentence)
        self.__features[seg] = encoded_features

    def __padding_features(self, seg, maxlen=500, pad=0):
        """ pad all features to the same length """
        padded_features = []
        for feature in self.__features[seg]:
            if len(feature) >= maxlen:
                padded_feature = feature[:maxlen]
            else:
                padded_feature = feature
                while len(padded_feature) < maxlen:
                    padded_feature.append(pad)
            padded_features.append(padded_feature)
        self.__features[seg] = padded_features

    def __gen_weight_np(self, seg):
        """
        generate weight by gensim
        """
        weight_np = np.zeros((len(self.__word2idx[seg]), self.__glove_dim), dtype=np.float32)
        for word, idx in self.__word2idx[seg].items():
            if word not in self.__wvmodel:
                continue
            word_vector = self.__wvmodel.get_vector(word)
            weight_np[idx, :] = word_vector

        self.__weight_np[seg] = weight_np

    def get_datas(self, seg):
        """
        return features, labels, and weight
        """
        features = np.array(self.__features[seg]).astype(np.int32)
        labels = np.array(self.__labels[seg]).astype(np.int32)
        weight = np.array(self.__weight_np[seg])
        return features, labels, weight



def _convert_to_mindrecord(data_home, features, labels, weight_np=None, training=True):
    """
    convert imdb dataset to mindrecoed dataset
    """
    if weight_np is not None:
        np.savetxt(os.path.join(data_home, 'weight.txt'), weight_np)

    # write mindrecord
    schema_json = {"id": {"type": "int32"},
                   "label": {"type": "int32"},
                   "feature": {"type": "int32", "shape": [-1]}}

    data_dir = os.path.join(data_home, "aclImdb_train.mindrecord")
    if not training:
        data_dir = os.path.join(data_home, "aclImdb_test.mindrecord")

    def get_imdb_data(features, labels):
        data_list = []
        for i, (label, feature) in enumerate(zip(labels, features)):
            data_json = {"id": i,
                         "label": int(label),
                         "feature": feature.reshape(-1)}
            data_list.append(data_json)
        return data_list

    writer = FileWriter(data_dir, shard_num=4)
    data = get_imdb_data(features, labels)
    writer.add_schema(schema_json, "nlp_schema")
    writer.add_index(["id", "label"])
    writer.write_raw_data(data)
    writer.commit()


def convert_to_mindrecord(embed_size, aclimdb_path, preprocess_path, glove_path):
    """
    convert imdb dataset to mindrecoed dataset
    """
    parser = ImdbParser(aclimdb_path, glove_path, embed_size)
    parser.parse()

    if not os.path.exists(preprocess_path):
        print(f"preprocess path {preprocess_path} is not exist")
        os.makedirs(preprocess_path)

    train_features, train_labels, train_weight_np = parser.get_datas('train')
    _convert_to_mindrecord(preprocess_path, train_features, train_labels, train_weight_np)

    test_features, test_labels, _ = parser.get_datas('test')
    _convert_to_mindrecord(preprocess_path, test_features, test_labels, training=False)

if args.preprocess == "true":
    os.system("rm -f ./preprocess/aclImdb* weight*")
    print("============== Starting Data Pre-processing ==============")
    convert_to_mindrecord(cfg.embed_size, args.aclimdb_path, args.preprocess_path, args.glove_path)
    print("======================= Successful =======================")

```

> ============== Starting Data Pre-processing ==============
>
> ======================= Successful =======================

ת���ɹ������`preprocess`Ŀ¼������MindRecord�ļ���ͨ���ò��������ݼ����������£�����ÿ��ѵ����ִ�У���ʱ�鿴`preprocess`�ļ�Ŀ¼�ṹ��

��ʱ�ļ��ṹ���£�

> preprocess
>
> ������ aclImdb_test.mindrecord0
>
> ������ aclImdb_test.mindrecord0.db
>
> ������ aclImdb_test.mindrecord1
>
> ������ aclImdb_test.mindrecord1.db
>
> ������ aclImdb_test.mindrecord2
>
> ������ aclImdb_test.mindrecord2.db
>
> ������ aclImdb_test.mindrecord3
>
> ������ aclImdb_test.mindrecord3.db
>
> ������ aclImdb_train.mindrecord0
>
> ������ aclImdb_train.mindrecord0.db
>
> ������ aclImdb_train.mindrecord1
>
> ������ aclImdb_train.mindrecord1.db
>
> ������ aclImdb_train.mindrecord2
>
> ������ aclImdb_train.mindrecord2.db
>
> ������ aclImdb_train.mindrecord3
>
> ������ aclImdb_train.mindrecord3.db
>
> ������ weight.txt
>
> 
>
> 0 directories, 17 files

��ʱ`preprocess`Ŀ¼�µ��ļ�Ϊ��

- ���ư���`aclImdb_train.mindrecord`��Ϊת�����MindRecord��ʽ��ѵ�����ݼ���
- ���ư���`aclImdb_test.mindrecord`��Ϊת�����MindRecord��ʽ�Ĳ������ݼ���
- `weight.txt`ΪԤ������Զ����ɵ�weight������Ϣ�ļ���

����ѵ������

- ���崴�����ݼ�����`lstm_create_dataset`������ѵ����`ds_train`��
- ͨ��`create_dict_iterator`���������ֵ����������ȡ�Ѵ��������ݼ�`ds_train`�е����ݡ�

��������һ�δ��룬�������ݼ�����ȡ��1��`batch`�е�`label`�����б��͵�1��`batch`�е�1��Ԫ�ص�`feature`���ݡ�

```python
import os
import mindspore.dataset as ds


def lstm_create_dataset(data_home, batch_size, repeat_num=1, training=True):
    """Data operations."""
    ds.config.set_seed(1)
    data_dir = os.path.join(data_home, "aclImdb_train.mindrecord0")
    if not training:
        data_dir = os.path.join(data_home, "aclImdb_test.mindrecord0")

    data_set = ds.MindDataset(data_dir, columns_list=["feature", "label"], num_parallel_workers=4)

    # apply map operations on images
    data_set = data_set.shuffle(buffer_size=data_set.get_dataset_size())
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)
    data_set = data_set.repeat(count=repeat_num)

    return data_set

ds_train = lstm_create_dataset(args.preprocess_path, cfg.batch_size)

iterator = next(ds_train.create_dict_iterator())
first_batch_label = iterator["label"].asnumpy()
first_batch_first_feature = iterator["feature"].asnumpy()[0]
print(f"The first batch contains label below:\n{first_batch_label}\n")
print(f"The feature of the first item in the first batch is below 
```



## ��������

1. �����ʼ����������ģ�顣
2. ������Ҫ����LSTMС���Ӷѵ����豸���͡�
3. ����`lstm_default_state`��������ʼ���������������״̬��
4. ����`stack_lstm_default_state`��������ʼ��С���Ӷѵ���Ҫ�ĳ�ʼ���������������״̬��
5. ���CPU�������Զ��嵥��LSTMС���Ӷѵ�����ʵ�ֶ��LSTM�����ӹ��ܡ�
6. ʹ��`Cell`��������������ṹ��`SentimentNet`���磩��
7. ʵ����`SentimentNet`���������磬�����������м��صĲ�����

```python
import math
import numpy as np
from mindspore import Tensor, nn, context, Parameter, ParameterTuple
from mindspore.common.initializer import initializer
import mindspore.ops as ops

STACK_LSTM_DEVICE = ["CPU"]

# Initialize short-term memory (h) and long-term memory (c) to 0
def lstm_default_state(batch_size, hidden_size, num_layers, bidirectional):
    """init default input."""
    num_directions = 2 if bidirectional else 1
    h = Tensor(np.zeros((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32))
    c = Tensor(np.zeros((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32))
    return h, c

def stack_lstm_default_state(batch_size, hidden_size, num_layers, bidirectional):
    """init default input."""
    num_directions = 2 if bidirectional else 1

    h_list = c_list = []
    for _ in range(num_layers):
        h_list.append(Tensor(np.zeros((num_directions, batch_size, hidden_size)).astype(np.float32)))
        c_list.append(Tensor(np.zeros((num_directions, batch_size, hidden_size)).astype(np.float32)))
    h, c = tuple(h_list), tuple(c_list)
    return h, c


class StackLSTM(nn.Cell):
    """
    Stack multi-layers LSTM together.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 has_bias=True,
                 batch_first=False,
                 dropout=0.0,
                 bidirectional=False):
        super(StackLSTM, self).__init__()
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.transpose = ops.Transpose()

        # direction number
        num_directions = 2 if bidirectional else 1

        # input_size list
        input_size_list = [input_size]
        for i in range(num_layers - 1):
            input_size_list.append(hidden_size * num_directions)

        # layers
        layers = []
        for i in range(num_layers):
            layers.append(nn.LSTMCell(input_size=input_size_list[i],
                                      hidden_size=hidden_size,
                                      has_bias=has_bias,
                                      batch_first=batch_first,
                                      bidirectional=bidirectional,
                                      dropout=dropout))

        # weights
        weights = []
        for i in range(num_layers):
            # weight size
            weight_size = (input_size_list[i] + hidden_size) * num_directions * hidden_size * 4
            if has_bias:
                bias_size = num_directions * hidden_size * 4
                weight_size = weight_size + bias_size

            # numpy weight
            stdv = 1 / math.sqrt(hidden_size)
            w_np = np.random.uniform(-stdv, stdv, (weight_size, 1, 1)).astype(np.float32)

            # lstm weight
            weights.append(Parameter(initializer(Tensor(w_np), w_np.shape), name="weight" + str(i)))

        #
        self.lstms = layers
        self.weight = ParameterTuple(tuple(weights))

    def construct(self, x, hx):
        """construct"""
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        # stack lstm
        h, c = hx
        hn = cn = None
        for i in range(self.num_layers):
            x, hn, cn, _, _ = self.lstms[i](x, h[i], c[i], self.weight[i])
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        return x, (hn, cn)


class SentimentNet(nn.Cell):
    """Sentiment network structure."""

    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_hiddens,
                 num_layers,
                 bidirectional,
                 num_classes,
                 weight,
                 batch_size):
        super(SentimentNet, self).__init__()
        # Mapp words to vectors
        self.embedding = nn.Embedding(vocab_size,
                                      embed_size,
                                      embedding_table=weight)
        self.embedding.embedding_table.requires_grad = False
        self.trans = ops.Transpose()
        self.perm = (1, 0, 2)

        if context.get_context("device_target") in STACK_LSTM_DEVICE:
            # stack lstm by user
            self.encoder = StackLSTM(input_size=embed_size,
                                     hidden_size=num_hiddens,
                                     num_layers=num_layers,
                                     has_bias=True,
                                     bidirectional=bidirectional,
                                     dropout=0.0)
            self.h, self.c = stack_lstm_default_state(batch_size, num_hiddens, num_layers, bidirectional)
        else:
            # standard lstm
            self.encoder = nn.LSTM(input_size=embed_size,
                                   hidden_size=num_hiddens,
                                   num_layers=num_layers,
                                   has_bias=True,
                                   bidirectional=bidirectional,
                                   dropout=0.0)
            self.h, self.c = lstm_default_state(batch_size, num_hiddens, num_layers, bidirectional)

        self.concat = ops.Concat(1)
        if bidirectional:
            self.decoder = nn.Dense(num_hiddens * 4, num_classes)
        else:
            self.decoder = nn.Dense(num_hiddens * 2, num_classes)

    def construct(self, inputs):
        # input��(64,500,300)
        embeddings = self.embedding(inputs)
        embeddings = self.trans(embeddings, self.perm)
        output, _ = self.encoder(embeddings, (self.h, self.c))
        # states[i] size(64,200)  -> encoding.size(64,400)
        encoding = self.concat((output[0], output[499]))
        outputs = self.decoder(encoding)
        return outputs

embedding_table = np.loadtxt(os.path.join(args.preprocess_path, "weight.txt")).astype(np.float32)
network = SentimentNet(vocab_size=embedding_table.shape[0],
                       embed_size=cfg.embed_size,
                       num_hiddens=cfg.num_hiddens,
                       num_layers=cfg.num_layers,
                       bidirectional=cfg.bidirectional,
                       num_classes=cfg.num_classes,
                       weight=Tensor(embedding_table),
                       batch_size=cfg.batch_size)

print(network.parameters_dict(recurse=True))

```



## ѵ��������ģ��

��������һ�δ��룬�����Ż�������ʧ����ģ�ͣ�����ѵ�����ݼ���`ds_train`�������ú�`CheckPoint`������Ϣ��Ȼ��ʹ��`model.train`�ӿڣ�����ģ��ѵ����

```python
from mindspore import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor, LossMonitor
from mindspore.nn import Accuracy
from mindspore import nn

os.system("rm -f {0}/*.ckpt {0}/*.meta".format(args.ckpt_path))
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
model = Model(network, loss, opt, {'acc': Accuracy()})
loss_cb = LossMonitor(per_print_times=78)
print("============== Starting Training ==============")
config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps, keep_checkpoint_max=cfg.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=args.ckpt_path, config=config_ck)
time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
if args.device_target == "CPU":
    model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb], dataset_sink_mode=False)
else:
    model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb])
print("============== Training Success ==============")
```

> ============== Starting Training ==============
>
> epoch: 1 step: 78, loss is 0.2971678
>
> epoch: 1 step: 156, loss is 0.30519545
>
> ... ...
>
> epoch: 10 step: 312, loss is 0.050257515
>
> epoch: 10 step: 390, loss is 0.025655827
>
> Epoch time: 27546.935, per step time: 70.633
>
> ============== Training Success ==============

## ģ����֤

������������֤���ݼ���`ds_eval`����������**ѵ��**�����CheckPoint�ļ���������֤���鿴ģ��������

```python 
from mindspore import load_checkpoint, load_param_into_net
args.ckpt_path_saved = f'{args.ckpt_path}/lstm-{cfg.num_epochs}_390.ckpt'
print("============== Starting Testing ==============")
ds_eval = lstm_create_dataset(args.preprocess_path, cfg.batch_size, training=False)
param_dict = load_checkpoint(args.ckpt_path_saved)
load_param_into_net(network, param_dict)
if args.device_target == "CPU":
    acc = model.eval(ds_eval, dataset_sink_mode=False)
else:
    acc = model.eval(ds_eval)
print("============== {} ==============".format(acc))
```



### ѵ���������

��������һ�δ����������Կ������ھ�����10��epoch֮��ʹ����֤�����ݼ������ı�����з�����ȷ����85%���ң��ﵽһ����������Ľ����

## �ܽ�

���ϱ������MindSpore��Ȼ���Դ���Ӧ�õ����飬����ͨ����������ȫ���˽������ʹ��MindSpore������Ȼ�����д�����з������⣬��������ͨ������ͳ�ʼ������LSTM��`SentimentNet`�������ѵ��ģ�ͼ���֤��ȷ�ʡ�

## ʵ��Ҫ��

1. ��ʵ���ֲ����ʵ��

2. ��дʵ�鱨�棨���ݰ�����������ʵ�����ݡ�ʵ��˼·��ʵ�ֹ��̡������ͼչʾ��ʵ���ܽ�ȣ�

   ʵ�鱨��������ʽ��**ѧ��-����-ʵ�����**�����磺111702xxxxx -��xx-ʵ����

3. **5��29������12:00֮ǰ**��������루ipynb�ļ�����ʵ�鱨����������ϴ������䡣

  ѹ��������ͬʵ�鱨��  

  ���䣺**NLPSpring2022@163.com**

  �ʼ����⣺**����+ʵ�����** ������xxʵ����

 
