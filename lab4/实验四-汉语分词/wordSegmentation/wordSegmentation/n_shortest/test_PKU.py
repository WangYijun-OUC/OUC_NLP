import time
from n_shortestRoute_byDict import DAGSegger

# from wordSegmentation.n_shortest.n_shortestRoute_byDict import DAGSegger

fr_test = open("D:/NLP/lab4/pku_test.utf8", 'r', encoding='utf-8')
fr_test_gold = open("D:/NLP/lab4/pku_test_gold.utf8", 'r', encoding='utf-8')

d = DAGSegger()

Test = []
Test_gold = []
# fr_test1 = "党中央必须坚持全心全意为人民服务的宗旨"
# fr_test = "".join(fr_test1)
# fr_test_gold1 = "党中央  必须  坚持  全心全意  为  人民  服务  的  宗旨"
# fr_test_gold = "".join(fr_test_gold1)

for i, x in enumerate(fr_test):
    print(x)
    print("1")
    data = []
    j = 0
    print(d.test(x))
    for s in d.test(x)[:-1:1]:
        word = [j, j + len(s) - 1]
        # print(word)
        data.append(word)
        j += len(s)
    # print(data)
    Test.append(data)
    

for i, x in enumerate(fr_test_gold):
    data = []
    j = 0
    x = x.split()
    print(x)
    for s in x[:-1:1]:
        word = [j, j + len(s) - 1]
        data.append(word)
        j += len(s)
    #print(data)
    Test_gold.append(data)

Test_num = 0
Test_gold_num = 0
right_num = 0

#print(Test_gold)
#print(Test)
for i in range(len(Test_gold)):
    print(Test[i])
    print(Test_gold[i])
    Test_num += len(Test[i])
    Test_gold_num += len(Test_gold[i])
    right_num += len([x for x in Test[i] if x in Test_gold[i]])

print(right_num)
print(Test_num)
print(Test_gold_num)
print("精确率", right_num / Test_num)
print("召回率", right_num / Test_gold_num)

# 精确率 0.7764031562984604
# 召回率 0.8847287850978248
