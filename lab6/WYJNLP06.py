# coding:utf-8
#import the packages
from sklearn. feature_extraction.text import TfidfTransformer
from sklearn. feature_extraction.text import CountVectorizer

#0pen files
with open("D:\\NLP\\lab6\\metadata.txt", 'r',encoding=' utf-8') as f :
    dataset = list(f. readlines())
    print(len(dataset))
    print(dataset)

# #open stopword text
# stopwords = open("D:\\NLP\\lab6\\stopwords.txt",'r',encoding= "utf-8").read().replace('\n', ' ').split()
with open("D:\\NLP\\lab6\\stopwords.txt",'r',encoding= "utf-8") as f :
    stopwords = list(f.readlines())
    print(stopwords)

vectorizer = CountVectorizer(stop_words = stopwords, min_df = 0 ) #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j]. 表示j词在i类文本下的词频
transformer = TfidfTransformer() #该类会统计每个词语的tf-idf权值
tfidf = transformer.fit_transform(vectorizer.fit_transform(dataset)) #第一 个fit_ transform是计算tf-idf,第二个fit_ transform是将 文本转为词频矩阵
word = vectorizer.get_feature_names_out() #获取词袋模型中的所有词语
print("word:", word)
print(vectorizer.vocabulary_) #查看到所有文本的关键字和其位置

weight = tfidf.toarray () #将tf-idf矩阵抽取出来，0元素a[i][j]表示j词在i类文本中的tf-idf权重
print("weight:",weight)

word_weight=list()
for i in range(len(word)):#print the weight of each word
    print("-------这里输出第%d类文本的词语tf-idf权重-----" % i)
    print("    ",word[i], weight[0][i])

