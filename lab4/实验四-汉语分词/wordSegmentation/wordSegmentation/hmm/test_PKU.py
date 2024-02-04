import time
from wordSegmentation.hmm.hmm_model import HmmModel

fr_test = open("../dataset/pku/pku_test.utf8", 'r', encoding='utf-8')
fr_test_gold = open("../dataset/pku/pku_test_gold.utf8", 'r', encoding='utf-8')

hmm = HmmModel()

Test = []
Test_gold = []
for i, x in enumerate(fr_test):
    print(x)
    data = []
    j = 0
    for s in hmm.cut(x):
        word = [j, j + len(s) - 1]
        data.append(word)
        j += len(s)
    Test.append(data)

for i, x in enumerate(fr_test_gold):
    data = []
    j = 0
    x = x.split()
    for s in x[:-1:1]:
        word = [j, j + len(s) - 1]
        data.append(word)
        j += len(s)
    Test_gold.append(data)

Test_num = 0
Test_gold_num = 0
right_num = 0

for i in range(len(Test_gold)):
    Test_num += len(Test[i])
    Test_gold_num += len(Test_gold[i])
    right_num += len([x for x in Test[i] if x in Test_gold[i]])

print("精确率", right_num / Test_num)
print("召回率", right_num / Test_gold_num)

# 精确率 0.7764031562984604
# 召回率 0.8847287850978248
