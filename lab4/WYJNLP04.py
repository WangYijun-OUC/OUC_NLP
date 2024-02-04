import re
import string
from zhon.hanzi import punctuation
from _overlapped import NULL
from collections import Counter

ans_forward = []  #存放正向匹配的结果
max_length = 5    #分词字典中最大长度字符串的长度

def FowardMatch(s_test, ori_list):
    ans_forward = []
    tmp = []
    len_row = len(s_test)       #len_orw为当前为划分句子的长度
    while len_row > 0:          #当前待划分句子长度为0时，结束划分
        divide = s_test[0:max_length]           #从前向后截取长度为max_length的字符串
        while divide not in ori_list:           #当前截取的字符串不在分词字典中，则进循环
            if len(divide) == 1:                #当前截取的字符串长度为1时，说明分词字典无匹配项
                break                           #直接保留当前的一个字
            divide = divide[0:len(divide) - 1]  #当前截取的字符串长度减一
        ans_forward.append(divide)              #记录下当前截取的字符串
        tmp.append(divide)
        s_test = s_test[len(divide):]           #截取未分词的句子
        len_row = len(s_test)                   #记录未分词的句子的长度
        if(len_row) :                           #将划分的词语用/隔开
            ans_forward.append("/")
    str1 = "".join(ans_forward)
    # str = "".join(tmp)
    print("\'正向最大匹配\'的分词结果为：", str1)
    return tmp

def Modify(s):
    if s[-1] in (r"[%s]+" %punctuation):
        s = s[:-1]

    s_modify1 = re.sub(r"[%s]+" %punctuation, "  ", s)
    return s_modify1

def Partition_Statistics(s, lists):
    format_tmp = "".join(s)
    lists = format_tmp.split()
    #将词按空格分割后依次填入数组              
    return lists

word_dict = {}
stop_words = {}
#self.load()

# 对于最短路径法 读取词库数据，词库为空格分隔的文章等。文件后缀应当为.utf8
def load_data():
    with open('D:/NLP/lab4/metadata.txt', encoding='utf-8') as fr :
        for line in fr:
            words = line.split(" ")
            for word in words:
                if word in stop_words:
                    continue
                word_dict[word] = word_dict.get(word, 0) + 1

# 基于字典 构建有向无环图
def build_dag(sentence):
    dag = {}
    for start in range(len(sentence)):
        tmp = []
        for stop in range(start + 1, len(sentence) + 1):
            fragment = sentence[start:stop]
            # use tf_idf?
            num = word_dict.get(fragment, 0)
            if num > 0 :
                tmp.append((stop, num))

        dag[start] = tmp
    if num == 0 :
        tmp.append((len(sentence), num))
    #print(dag)
    return dag


# 从N个路径中，挑选出最优路径
def predict(sentence):
    wordList = []
    Len = len(sentence)
    route = []
    dag = build_dag(sentence)  # {i: (stop, num)}
    i = 0
    while i < len(sentence):
        end = max(dag[i], key=lambda x: x[0])[0]
        wordList.append(sentence[i:end])
        i = end
    return wordList

# 将生成的最短路径切分
def cut(sentence):
    route = predict(sentence)
    word_list = []
    i = 0
    while i < len(sentence):
        next = route[i]
        word_list.append(sentence[i:next])
        i = next
    return word_list

# 单句测试
def test(s):
    cases = []
    ans = []
    tmp = []
    # cases = [
    #     "党中央必须坚持全心全意为人民服务的宗旨"
    # ]
    
    cases.append(s)
    for case in cases:
        result = predict(case)
        lenth = len(result)
        for word in result:
            ans.append(word)
            tmp.append(word)
            lenth = lenth - 1
            if lenth != 0 :                  
                ans.append("/")
    
    str1 = "".join(ans)                 #list转换为string类型
    print("\'最短路径法\'的分词结果为：", str1)
    return tmp

def  count_list(lists):
    for num in lists:
        sum += 1
    return sum

def acc(s_test, s_test_ac) :
    Test = []
    Test_gold = []

    fr_test = []
    fr_test.append(s_test)

    fr_test_gold = []
    fr_test_gold.append(s_test_ac)


    for x in fr_test:
        # print(x)
        # result = predict(x)
        data = []
        j = 0
        # x = x.split()
        for s in x[:-1:1]:
            word = [j, j + len(s) - 1]
            data.append(word)
            j += len(s)
        # print(data)
        Test.append(data)

    for x in fr_test_gold:
        # print(x)
        # result = predict(x)
        data = []
        j = 0
        x = x.split()
        for s in x[:-1:1]:
            word = [j, j + len(s) - 1]
            data.append(word)
            j += len(s)
        # print(data)
        Test_gold.append(data)

    # print(Test)
    # print(Test_gold)
    Test_num = 0
    Test_gold_num = 0
    right_num = 0

    for i in range(len(Test_gold)):
        Test_num += len(Test[i])
        Test_num += 1
        Test_gold_num += len(Test_gold[i])
        Test_gold_num += 1
        right_num += len([x for x in Test[i] if x in Test_gold[i]])
        right_num += 1

    # print(right_num)
    # print(Test_num)
    # print(Test_gold_num)

    Precision = right_num / Test_num
    Recall = right_num / Test_gold_num
    Fm = ( 2 * Precision * Recall ) / ( Precision + Recall )
    print("正确率", Precision)
    print("召回率", Recall)
    print("F-测度值", Fm)

if __name__ == "__main__":
    with open('D:/NLP/lab4/metadata.txt', encoding='utf-8') as file_obj:
        s = file_obj.read()
        #print(s.rstrip())

    ori_list = []

    #测试句子
    s_test1 = "党中央必须坚持全心全意为人民服务的宗旨"
    s_test1_ac = "党  中央  必须  坚持  全心全意  为  人民  服务  的  宗旨"
    s_test2 = "以经济建设为中心是邓小平理论的基本观点"
    s_test2_ac = "以  经济  建设  为  中心  是  邓小平理论  的  基本  观点"
    s_test3 = "坚定不移建设有中国特色社会主义道路"
    s_test3_ac = "坚定不移  建设  有  中国  特色  社会主义  道路"
    s_test4 = "精神文明建设和民主法制建设"
    s_test4_ac = "精神  文明  建设  和  民主  法制  建设"
    s_test5 = "他只会诊断一般的疾病"
    s_test5_ac = "他  只会  诊断  一般  的  疾病"
    test_list=[]
    count_list=[]

    #分词并将结果存入一个list，词频统计结果存入字典
    s_ori = Modify(s)
    ori_list = Partition_Statistics(s_ori, ori_list)

    #开始对每个测试句子进行分词
    print("Sentece1:")
    print(s_test1)
    print("正确分词")
    print(s_test1_ac)
    Forward_test1 = []
    Shortest_test1 = []
    s_test1 = Modify(s_test1)
    Forward_test1 = FowardMatch(s_test1, ori_list)

    load_data()
    Shortest_test1 = test(s_test1)

    print("正向匹配算法的评价指标值")
    acc(Forward_test1, s_test1_ac)
    print("")
    print("最短路径算法的评价指标值")
    acc(Shortest_test1, s_test1_ac)
    print("")
    #######################
    print("Sentece2:")
    print(s_test2)
    print("正确分词")
    print(s_test2_ac)
    Forward_test2 = []
    Shortest_test2 = []
    s_test2 = Modify(s_test2)
    Forward_test2 = FowardMatch(s_test2, ori_list)

    load_data()
    Shortest_test2 = test(s_test2)

    print("正向匹配算法的评价指标值")
    acc(Forward_test2, s_test2_ac)
    print("")
    print("最短路径算法的评价指标值")
    acc(Shortest_test2, s_test2_ac)
    print("")
    #######################
    print("Sentece3:")
    print(s_test3)
    print("正确分词")
    print(s_test3_ac)
    Forward_test3 = []
    Shortest_test3 = []
    s_test3 = Modify(s_test3)
    Forward_test3 = FowardMatch(s_test3, ori_list)

    load_data()
    Shortest_test3 = test(s_test3)

    print("正向匹配算法的评价指标值")
    acc(Forward_test3, s_test3_ac)
    print("")
    print("最短路径算法的评价指标值")
    acc(Shortest_test3, s_test3_ac)
    print("")
    #######################
    print("Sentece4:")
    print(s_test4)
    print("正确分词")
    print(s_test4_ac)
    Forward_test4 = []
    Shortest_test4 = []
    s_test4 = Modify(s_test4)
    Forward_test4 = FowardMatch(s_test4, ori_list)

    load_data()
    Shortest_test4 = test(s_test4)

    print("正向匹配算法的评价指标值")
    acc(Forward_test4, s_test4_ac)
    print("")
    print("最短路径算法的评价指标值")
    acc(Shortest_test4, s_test4_ac)
    print("")
    #######################
    print("Sentece5:")
    print(s_test5)
    print("正确分词")
    print(s_test5_ac)
    Forward_test5 = []
    Shortest_test5 = []
    s_test5 = Modify(s_test5)
    Forward_test5 = FowardMatch(s_test5, ori_list)

    load_data()
    Shortest_test5 = test(s_test5)

    print("正向匹配算法的评价指标值")
    acc(Forward_test5, s_test5_ac)
    print("")
    print("最短路径算法的评价指标值")
    acc(Shortest_test5, s_test5_ac)
    print("")

