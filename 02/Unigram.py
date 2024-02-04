import numpy as np
import string
import heapq
import random
word_list = []#存放所有单词
word_dir = {}#存放无重复字典
word_dir_sl = {}#存放字典索引
word_sentence_list = []#存放句子一维表
word_unRepeat_list = []#存放无重复列表单词

#统计每个单词的出现次数用字典标注,参数返回一个字典
def count_word_hapeen(text):
    for sentence in text:
        notDeal_sentence = sentence.split('|')[2].strip('\n').translate(str.maketrans('', '', string.punctuation))
        word_sentence_list.append(notDeal_sentence)
        for word in notDeal_sentence.split(" "):
            word_list.append(word)
    #开始向字典里面增加数据，并开始计数
    count = 0
    for i in word_list:
        if(word_dir.get(i)!=None):
            word_dir[i]=word_dir[i]+1
        # word_dir.update(i,word_dir.values(i)+1)
        else:
            word_dir_sl[i]=count
            count+=1
           # print(word_dir.items())
           # print('\n')
            word_dir[i]=1
            #通过这里删选的时候来进而存放无重复列表单词
            word_unRepeat_list.append(i)
  #  print(word_dir.items())
#开始对文本里面的字符进行组合判断出现次数
def make_word(word_array):
    #print("\n")
    #print('length',len(word_dir))
    count_number=0
    #还是得看这里
    for k in word_sentence_list:
        sen_had_split = k.split()
        for h in range(0, (len(k.split()) - 2)):
           # print('目前下标',sen_had_split[h],'     '+sen_had_split[h-1],'\n')
           # print('字典里面的坐标',word_dir_sl[sen_had_split[h]], '     ' ,word_dir_sl[sen_had_split[h-1]], '\n')
            word_1 = word_dir_sl[sen_had_split[h]];
            word_2 = word_dir_sl[sen_had_split[h+1]];
            word_array[word_1][word_2]+=1
    return word_array
    '''
      下面的蠢方法,还是别弄了

    for i in range(0,len(word_dir)):
        for j in range(0,len(word_dir)):
            #开始对i j 来进行判断
            for k in word_sentence_list:
                #开始针对第二次方式，
                sen_had_split = k.split()
                for h in range(0,(len(k.split())-1)):
                    #逐行进行判断

                    if((word_unRepeat_list[i]==sen_had_split[h])and( word_unRepeat_list[j]==sen_had_split[h+1])):
                        count_number = count_number+1
            word_array[i][j] = count_number
            if(count_number>0):
                print('\n')
                print('i行=',i,'j列=',j, word_array[i][j])
            count_number = 0
'''
def printData(word_array):
    print("length",len(word_dir))
    for i in range(0,len(word_dir)-1):
        print("\n")
        for j in range(0,len(word_dir)-1):
              print('data  ',word_array[i][j],end='              ')


#开始计算每个出现的概率公式P（i，j）= word_array[i][j]/list[i]
def calculation_Array_Probability(word_array):
    word_problity_array = np.zeros((len(word_dir), len(word_dir)), dtype=float)  # 存储二维数据表
    for i in range(0,len(word_array)):
        print('\n')
        for j in range(0,len(word_array)):
            word_problity_array[i][j] =float((word_array[i][j])/ (word_dir[word_unRepeat_list[i]]))
            print('data  ', word_problity_array[i][j],'        ',word_dir[word_unRepeat_list[i]])
    return word_problity_array
'''
第二部分问题函数
'''
def ge_position_index(x):
    return

def find_five_word_list(word,array):
    five_list = []
    index = word_dir_sl[word]
    test = [0]*len(word_array)
   # word_find_one_array = np.zeros(len(word_dir), dtype=int)
    for i in range(0,len(word_array)):
        test[i] = array[index][i]
    #赋值完毕，建立堆来直接找出最多的五个单词
    index_list  = map(test.index,heapq.nlargest(5, test))
    temper_dir = {}
    for el in index_list:
        if (temper_dir.get(el) != None):
            five_list.append(word_unRepeat_list[random.randint(0,len(word_unRepeat_list))])
        else:
                temper_dir[el] = 1
                five_list.append(word_unRepeat_list[el])
    '''
    for j in range(0,5):
        five_list.append(index_list[j])
        '''
    return five_list
def cycle_input_word(word_array):
    flag = True
    user_input_word = input()
    while(flag):
        input_word = ''
        choice_number = -1
        wait_slect_list = []
        if(user_input_word==-1):
            flag = False
        else:
            if(word_dir.get(input_word)==None):
                wait_slect_list = []
            else:
                wait_slect_list = find_five_word_list(user_input_word,word_array)
                print("点击数字  1、2、3、4、5")
                for i in wait_slect_list:
                    print("  ",i,"   ")
                choice_number = input()
                user_input_word = wait_slect_list[int(choice_number)]
if __name__ == '__main__':
    #开始一些必要的函数，对全局变量的初始化
    #word_array=make_word();
    text = open('D:/NLP/02/metadata.txt',mode='r',encoding='UTF-8');
    # 初始化变量word_list和word_sentence_list
    #splitText(text)
    count_word_hapeen(text)
    word_array = np.zeros((len(word_dir), len(word_dir)), dtype=float)  # 存储二维数据表
    #制作二维表实例
    word_problity_array = np.zeros((len(word_dir),len(word_dir)),dtype=int)#存储二维数据表
    word_array = make_word(word_array)
    #printData(word_array)
    #开始对二维表进行计算其MLE，先计算每个单独的概率P（i，j）= word_array[i][j]/list[i]
    #word_problity_array = calculation_Array_Probability(word_array)
    '''
    开始第二部分问题，由输入数据input()来进行补充，接着开始推荐几个后面跟着的候补单词
    直接从，该单词行来找出，前五个最大的数->从word_array取直接放入list。输出
    由于循环输出，直接使用while(),输入截至的标志为-1；
    '''
    print("---------------请输入单词----------")
    cycle_input_word(word_array)
    printData(word_array)
