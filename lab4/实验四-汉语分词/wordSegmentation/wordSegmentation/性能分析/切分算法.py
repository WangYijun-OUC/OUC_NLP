"""
算法：正向最长匹配、逆向最长匹配、双向最长匹配
研究：切分效果、速度测评
"""
import time


# 正向最长匹配
def forward_segment(text, dic):
    word_list = []
    i = 0
    while i < len(text):
        longest_word = text[i]  # 当前扫描位置的单字
        for j in range(i + 1, len(text) + 1):  # 所有可能的结尾
            word = text[i:j]  # 从当前位置到结尾的连续字符串
            if word in dic:  # 是否在词典中
                if len(word) > len(longest_word):  # 且更长
                    longest_word = word  # 则优先输出
        word_list.append(longest_word)  # 输出最长词
        i += len(longest_word)  # 正向扫描
    return word_list


# 逆向最长匹配
def backward_segment(text, dic):
    word_list = []
    i = len(text) - 1
    while i >= 0:
        longest_word = text[i]  # 当前扫描位置的单字,作为词的终点
        for j in range(0, i):  # 所有可能的前缀
            word = text[j:i + 1]  # 从前缀到当前位置连续字符串
            if word in dic:  # 是否在词典中
                if len(word) > len(longest_word):  # 且更长
                    longest_word = word  # 则优先输出
        word_list.insert(0, longest_word)  # 输出最长词
        i -= len(longest_word)  # 后向扫描
    return word_list


# 双向最长匹配
def bidirectional_segment(text, dic):
    # 统计单字个数
    def count_single_char(word_list):
        return sum(1 for word in word_list if len(word) == 1)

    f = forward_segment(text, dic)
    b = backward_segment(text, dic)
    if len(f) < len(b):  # 词数更少优先级更高
        return f
    elif len(f) > len(b):
        return b
    else:
        if count_single_char(f) < count_single_char(b):  # 单字更少优先级更高
            return f
        else:  # 都相等时逆向匹配优先级更高
            return b


# 加载数据集
def load_data():
    dic = []
    data = open("words_2.txt", "r", encoding="utf-8")
    for line in data:
        words = line.split(" ")
        dic.append(words[0])
    return dic


# 切分效果
sent = ["欢迎新老师生", "使用户满意"]
for s in sent:
    print(forward_segment(s, load_data()))
    print(backward_segment(s, load_data()))
    print(bidirectional_segment(s, load_data()))


# 速度测评
def evaluate_speed(segment, text, dic):
    start_time = time.time()
    for i in range(100):
        segment(text, dic)
    elapsed_time = time.time() - start_time
    print('%.8f 万字每秒' % (len(text) * 100 / 10000 / elapsed_time))

text = "想要了解更多北京相关的内容"
dic = load_data()
evaluate_speed(forward_segment, text, dic)
evaluate_speed(backward_segment, text, dic)
evaluate_speed(bidirectional_segment, text, dic)