import json
import pickle
# 引入配置文件中的语料库地址
from CONFIG import word_dict_DatasetName

# ##############
# main--词典-- 完全的python字典结构
# 用来读取utf8格式文章库、json格式词典库、停用词；
# 可以转存为json形式的txt词典库；
# 初始化时默认调用load函数，读取dataset/unknown/words.txt
# ##############


class WordDictModel:
    def __init__(self):
        self.word_dict = {}
        self.data = None
        self.stop_words = {}
        #self.load()

    # 读取词库数据，词库为空格分隔的文章等。文件后缀应当为.utf8
    def load_data(self, filename):
        self.data = open(filename, "r", encoding="utf-8")

    # 更新新词库数据，把读取的词库转化成内部字典word_dict
    def update(self):
        # build word_dict
        for line in self.data:
            words = line.split(" ")
            for word in words:
                if word in self.stop_words:
                    continue
                if self.word_dict.get(word):
                    self.word_dict[word] += 1
                else:
                    self.word_dict[word] = 1

    # 把内部词典里面的数据，转化成txt本文文档，格式为key-value键值对，相对于压缩
    def save(self, filename="words.txt", code="txt"):
        fw = open(filename, 'w', encoding="utf-8")
        data = {
            "word_dict": self.word_dict
        }

        # encode and write
        if code == "json":
            txt = json.dumps(data)
            fw.write(txt)
        elif code == "pickle":
            pickle.dump(data, fw)
        if code == 'txt':
            for key in self.word_dict:
                tmp = "%s %d\n" % (key, self.word_dict[key])
                fw.write(tmp)
        fw.close()

    # 加载初始词库，是用save()方法保存的key-value形式的词库
    def load(self, filename=word_dict_DatasetName, code="txt"):
        fr = open(filename, 'r', encoding='utf-8')

        # load model
        model = {}
        if code == "json":
            model = json.loads(fr.read())
        elif code == "pickle":
            model = pickle.load(fr)
        elif code == 'txt':
            word_dict = {}
            for line in fr:
                tmp = line.split(" ")
                if len(tmp) < 2:
                    continue
                word_dict[tmp[0]] = int(tmp[1])
            model = {"word_dict": word_dict}
        fr.close()

        # update word dict
        word_dict = model["word_dict"]
        for key in word_dict:
            if self.word_dict.get(key):
                self.word_dict[key] += word_dict[key]
            else:
                self.word_dict[key] = word_dict[key]

# 先注释掉self.load()
# d = WordDictModel()
# d.load_data("../dataset/pku/pku_training.utf8")
# d.update()
# d.save()
