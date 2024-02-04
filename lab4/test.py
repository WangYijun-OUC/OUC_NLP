# encoding = utf-8
class WordDictModel:
    def __init__(self):
        self.word_dict = {}
        self.stop_words = {}
        #self.load()

    # 读取词库数据，词库为空格分隔的文章等。文件后缀应当为.utf8
    def load_data(self, filename):
        with open(filename, "r", encoding="utf-8") as fr :
            for line in fr:
                words = line.split(" ")
                for word in words:
                    if word in self.stop_words:
                        continue
                    self.word_dict[word] = self.word_dict.get(word,0) + 1

class DAGSegger(WordDictModel):
    # 基于字典 构建有向无环图
    def build_dag(self, sentence):
        dag = {}
        for start in range(len(sentence)):
            tmp = []
            for stop in range(start + 1, len(sentence) + 1):
                fragment = sentence[start:stop]
                # use tf_idf?
                num = self.word_dict.get(fragment, 0)
                if num > 0 :
                    tmp.append((stop, num))

            dag[start] = tmp
        if num == 0 :
            tmp.append((len(sentence), num))
            print(dag)
        return dag


    # 从N个路径中，挑选出最优路径
    def predict(self, sentence):
        wordList = []
        Len = len(sentence)
        route = []
        dag = self.build_dag(sentence)  # {i: (stop, num)}
        i = 0
        while i < len(sentence):
            end = max(dag[i], key=lambda x: x[0])[0]
            wordList.append(sentence[i:end])
            i = end
        return wordList

    # 将生成的最短路径切分
    def cut(self, sentence):
        route = self.predict(sentence)
        word_list = []
        i = 0
        while i < len(sentence):
            next = route[i]
            word_list.append(sentence[i:next])
            i = next
        return word_list

    # 单句测试
    def test(self):
        cases = [
            "有意见分歧"
        ]
        for case in cases:
            result = self.predict(case)
            for word in result:
                print(word)
            print('')

def main():
    dag_segger = DAGSegger()
    dag_segger.load_data("D:/NLP/lab4/metadata.txt")
    print(dag_segger.word_dict)
    dag_segger.test()

if __name__ == '__main__':
    main()
