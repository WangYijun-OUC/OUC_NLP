import re
import string
from zhon.hanzi import punctuation
from _overlapped import NULL

def Modify(s):
    if s[-1] in (r"[%s]+" %punctuation):
        s = s[:-1]

    s_modify1 = re.sub(r"[%s]+" %punctuation, "EOS BOS", s)
    s_modify2="BOS  " + s_modify1 + "  EOS"
    return s_modify2

def Partition_Statistics(s, lists, dicts = NULL):
    format_tmp = "".join(s)
    lists = format_tmp.split()
    #将词按空格分割后依次填入数组

    if dicts != NULL:
        for word in lists:
            if word not in dicts:
                dicts[word] = 1
            else:
                dicts[word] += 1               
    return lists


def CompareList(ori_list,test_list):
    count_list=[0]*(len(test_list)-1)
    for i in range(0, len(test_list)-1):
        for j in range(0,len(ori_list)-2):                
            if test_list[i]==ori_list[j] :
                if test_list[i+1]==ori_list[j+1]:
                    count_list[i] += 1
    return count_list


def Probability(test_list,count_list,ori_dict):
    flag=0
    p = 1.0
    tmp = 1.0
    length = len(test_list)
    for i in range(length - 1): 
        tmp = (float(count_list[flag])/float(ori_dict[test_list[i]]))
        print("P(",test_list[flag + 1], "|", test_list[flag], ") = ", tmp)
        p = p * tmp
        flag += 1
    return p


if __name__ == "__main__":
    s = open('D:/NLP/02/metadata.txt',mode='r',encoding='UTF-8');
    
    with open('D:/NLP/02/metadata.txt', encoding='utf-8') as file_obj:
        s = file_obj.read()
        print(s.rstrip())

    ori_list=[]
    ori_dict={}

    #测试句子
    s_test="我们  伟大  祖国  在  新  的  一  年"
    test_list=[]
    count_list=[]

    #分词并将结果存入一个list，词频统计结果存入字典
    s_ori = Modify(s)
    #print(s_ori)
    ori_list = Partition_Statistics(s_ori,ori_list,ori_dict)
    #print(ori_list)

    s_test = Modify(s_test)
    test_list = Partition_Statistics(s_test,test_list)
    #print(test_list)

    count_list = CompareList(ori_list, test_list)
    p = Probability(test_list,count_list,ori_dict)
    print("P(", test_list, ") = ", p)
