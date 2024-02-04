import json
import pickle
# 引入配置文件中的语料库地址
from wordSegmentation.dataset.CONFIG import base_trie_DatasetName


# ##############
# main--词典-- 基础的前缀树结构
# 用来读取utf8格式文章库、json格式词典库、停用词；
# 可以转存为json形式的txt词典库；
# 初始化时默认调用load函数，读取dataset/unknown/words.txt
# ##############


# 节点
class Node:

    def __init__(self, value, f, isEnd):
        self.value = value  # 当前节点的取值，也就是一个字符
        self.f = f  # 当前节点的路径构成的词的词频
        self.isEnd = isEnd  # 当前节点的路径，是否构成完全词 | 词尾标识
        self.children = {}  # 当前节点的子节点，dict来存储


# 用于存储单词的前缀树
class BaseTrie:

    def __init__(self):
        self.children = {'root': Node('', -1, False)}
        self.load()

    # 添加新词 - 构建trie树结构
    def add_new_word(self, words, f):
        # 获取根节点
        temp_node = self.children['root']
        # 对新词中的字依次遍历
        for i, word in enumerate(words):
            # 若这个字已经存在,有可能是较长字符串生成的.
            # 需要判断当前字是否终止,终止则赋值f|isEnd,否则把它当前新的根节点.
            if word in temp_node.children:
                if i == len(words) - 1:
                    temp_node.children[word].f = f
                    temp_node.children[word].isEnd = True
                else:
                    temp_node = temp_node.children[word]
            # 若这个字不存在
            # 判断当前字是否终止,终止则创建叶子结点,否则创建内部结点.
            else:
                if i == len(words) - 1:
                    new_node = Node(word, f, True)
                else:
                    new_node = Node(word, -1, False)
                temp_node.children[word] = new_node
                temp_node = temp_node.children[word]

    # 加载初始词库，是用save()方法保存的key-value形式的词库
    def load(self, filename=base_trie_DatasetName, code="txt"):
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
            self.add_new_word(key, word_dict[key])


data = []


# 递归地召回前缀树上的所有路径，用于打印
def get_pathes(root, now_path):
    if root.isEnd:
        data.append(now_path + root.value)
    if root.children == {}:
        pass
    else:
        now_path += root.value
        for child_node in root.children:
            get_pathes(root.children[child_node], now_path)

# d = BaseTrie()
# d.load()
# print(d.children['root'].children)

# import json
# from pyecharts import options as opts
# from pyecharts.charts import Page, Tree
#
# echart_data = {}
#
# def get_pathes(root, now_data):
#     now_data['name'] = root.value
#
#     if root.children == {}:
#         pass
#     else:
#         now_data["children"] = []
#         for child_node in root.children:
#             dict = {}
#             now_data["children"].append(dict)
#             get_pathes(root.children[child_node], dict)
#
# get_pathes(d.children['root'],echart_data)
# tree=(
#      Tree()
#         .add("", [echart_data],orient="TB")
#         .set_global_opts(title_opts=opts.TitleOpts(title="Tree"))
#     )
# tree.width = '3000px'
# tree.height='3000px'
# tree.render()