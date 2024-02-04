# encoding=utf-8
from wordSegmentation.word_dict.base_trie import BaseTrie

# ############
# N-最短路径法 基于trie树
# ############


class DAGSegger(BaseTrie):
    # 基于trie树 构建有向无环图
    def build_dag(self, sentence):
        dag = {}
        for start in range(len(sentence)):
            root = self.children['root']
            unique = [start + 1]
            tmp = [(start + 1, 1)]

            for stop in range(start+1, len(sentence)+1):
                root = root.children.get(sentence[stop - 1])
                if root is not None:
                    if root.isEnd and (stop not in unique):
                        tmp.append((stop, root.f))
                        unique.append(stop)
                else:
                    break
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

    # 讲生成的最短路径切分
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
    def test(self,words):
        result = self.cut(words)
        print('/'.join(result))
        return result