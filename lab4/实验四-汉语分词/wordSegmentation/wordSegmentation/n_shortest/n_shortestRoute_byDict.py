# encoding=utf-8
from word_dict import WordDictModel

# ############
# N-最短路径法 基于字典
# ############


class DAGSegger(WordDictModel):
    # 基于字典 构建有向无环图
    def build_dag(self, sentence):
        dag = {}
        for start in range(len(sentence)):
            unique = [start + 1]
            tmp = [(start + 1, 1)]
            for stop in range(start + 1, len(sentence) + 1):
                fragment = sentence[start:stop]
                # use tf_idf?
                num = self.word_dict.get(fragment, 0)
                if num > 0 and (stop not in unique):
                    tmp.append((stop, num))
                    unique.append(stop)
            dag[start] = tmp
        return dag

    # 从N个路径中，挑选出最优路径
    def predict(self, sentence):
        Len = len(sentence)
        route = [0] * Len
        dag = self.build_dag(sentence)  # {i: (stop, num)}

        for i in range(0, Len):
            route[i] = max(dag[i], key=lambda x: x[1])[0]
        return route

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
    def test(self, words):
        result = self.cut(words)
        #print('/'.join(result))
        return result
