import pickle


class HmmModel:

    def __init__(self):
        # 分词状态
        self.STATE = {'B', 'M', 'E', 'S'}
        # 状态转移矩阵
        self.A_dict = {}
        # 发射矩阵
        self.B_dict = {}
        # 初始矩阵
        self.Pi_dict = {}

    # 加载数据 先加载模型数据，没有就读取语料库重新训练
    def load(self, model_file='../dataset/hmm/model.pkl', train_file='../dataset/hmm/train.txt'):

        # 加载模型数据
        try:
            with open(model_file, 'rb') as f:
                self.A_dict = pickle.load(f)
                self.B_dict = pickle.load(f)
                self.Pi_dict = pickle.load(f)
                return
        except FileNotFoundError:
            pass

        # 统计状态出现次数 方便求发射矩阵
        Count_dict = {}
        # 存放初始语料所有数据
        data = []
        # 存放初始语料中的一个句子
        sentence = []

        # 初始化模型参数
        def init_params():
            for state in self.STATE:
                self.A_dict[state] = {s: 0.0 for s in self.STATE}
                self.Pi_dict[state] = 0.0
                self.B_dict[state] = {}
                Count_dict[state] = 0

        init_params()

        # 读取语料库
        with open(train_file, encoding='utf8') as f:
            # 每句按元组存在data中
            for line in f:
                line = line.strip()
                word_list = [i for i in line if i != '\t']
                if not line:
                    data.append(sentence)
                    sentence = []
                else:
                    sentence.append((word_list[0], word_list[1]))

            # 统计次数
            for s in data:
                for k, v in enumerate(s):
                    Count_dict[v[1]] += 1
                    if k == 0:
                        self.Pi_dict[v[1]] += 1  # 每个句子的第一个字的状态，用于计算初始状态概率
                    else:
                        self.A_dict[s[k - 1][1]][v[1]] += 1  # 计算转移概率
                        self.B_dict[s[k][1]][v[0]] = self.B_dict[s[k][1]].get(v[0], 0) + 1.0  # 计算发射概率

            # 计算频率
            self.Pi_dict = {k: v * 1.0 / len(data) for k, v in self.Pi_dict.items()}
            self.A_dict = {k: {k1: v1 / Count_dict[k] for k1, v1 in v.items()} for k, v in self.A_dict.items()}
            # 加1平滑
            self.B_dict = {k: {k1: (v1 + 1) / Count_dict[k] for k1, v1 in v.items()} for k, v in self.B_dict.items()}

            # 把中间模型数据保存下来
            self.save()

    # 保存中间模型数据
    def save(self, model_file='../dataset/hmm/model.pkl'):
        # 序列化
        import pickle
        with open(model_file, 'wb') as f:
            pickle.dump(self.A_dict, f)
            pickle.dump(self.B_dict, f)
            pickle.dump(self.Pi_dict, f)

    # 维特比算法
    def viterbi(self, text):
        # 加载数据
        self.load()
        # 赋别名
        states, start_p, trans_p, emit_p = self.STATE, self.Pi_dict, self.A_dict, self.B_dict
        # 初始化顶点集、路径集
        V = [{}]
        path = {}
        # 初始化第一个状态
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]

        # 遍历剩下的状态
        for t in range(1, len(text)):
            V.append({})
            newpath = {}

            # 检验训练的发射概率矩阵中是否有该字
            neverSeen = text[t] not in emit_p['S'].keys() and \
                        text[t] not in emit_p['M'].keys() and \
                        text[t] not in emit_p['E'].keys() and \
                        text[t] not in emit_p['B'].keys()

            for y in states:
                # 生词值为1，发射矩阵一行内词找不到为0(发射矩阵有4行)
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0  # 设置未知字单独成词

                # 在当前状态为y下，计算前一个时刻的四种状态的代价乘积，取max
                (prob, state) = max(
                    [(V[t - 1][y0] * trans_p[y0].get(y, 0) *
                      emitP, y0)
                     for y0 in states if V[t - 1][y0] > 0])

                V[t][y] = prob

                newpath[y] = path[state] + [y]
            path = newpath

        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E', 'M')])
        else:
            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])

        return (prob, path[state])

    def cut(self, text):
        prob, pos_list = self.viterbi(text)
        begin, next = 0, 0
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin: i + 1]
                next = i + 1
            elif pos == 'S':
                yield char
                next = i + 1
        if next < len(text):
            yield text[next:]



# hmm = HmmModel()
# text = '（二○○○年十二月三十一日）（附图片1张）'
# res = hmm.cut(text)
# print(str(list(res)))

