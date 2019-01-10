import re
import numpy as np  
from collections import defaultdict


class TextRank:
    def __init__(self, coll_sw=set([]), d=0.85, window=5):
        self.sw = coll_sw  # collection
        self.d = d
        self.span = window
        self.graph = defaultdict(list)
        self.token2id = {}  # word: num
        self.id2token = {}  # num: word

    def _add_edge(self, start, end, weight):
        if start != end:
            self.graph[start].append((start, end, weight))
            self.graph[end].append((end, start, weight))
        else:
            self.graph[start].append((start, end, 2*weight))

    def _get_weight_matrix(self):
        dimension = len(self.graph)
        # 明确矩阵, 先构建二维列表，再整体转化成矩阵
        graph_list = [[0 for __ in range(dimension)] for _ in range(dimension)]
        sorted_keys = sorted(list(self.graph.keys()))  # a list
        for i, key in enumerate(sorted_keys):
            self.token2id[key] = i
            self.id2token[i] = key
        # 矩阵赋值
        for key in self.graph:
            for elem in self.graph[key]:
                graph_list[self.token2id.get(elem[1])][self.token2id[key]] = elem[2]  # assert len(elem) == 3

        self.matrix = np.array(graph_list, dtype='float32')
        # 权值计算
        self.matrix /= np.sum(self.matrix, axis=0, keepdims=True)

    def _calc_values(self, iter_num=100):  # 默认迭代100次
        self.value_vector = np.zeros((self.matrix.shape[0], 1), dtype='float32')
        for _ in range(iter_num):
            self.value_vector = self.matrix.dot(self.value_vector)
            self.value_vector = 1 - self.d + self.d * self.value_vector

    def _rank(self):
        tmp_list = self.value_vector.reshape(-1).tolist()
        new_list = [(self.id2token[i], round(value, 3)) for i, value in enumerate(tmp_list)]
        rank_list = sorted(new_list, key=lambda e: e[1], reverse=True)

        return rank_list

    def text_rank(self, seg_input, top=-1, sw_flag=True):
        """
        run TextRank task and get the result

        :param seg_input: string, segment text
        :param top: int, the number of key words
        :param sw_flag: bool, decides whether to use stopwords or not
        :return: 2d list, consists of words and theirs scores by TextRank
        """

        assert isinstance(seg_input, str)
        assert isinstance(top, int)

        if sw_flag and self.sw:  # 说明需要去停用词
            self.seg_list = [word for word in re.split('[ ]+', seg_input) if word not in self.sw]
        else:
            self.seg_list = re.split('[ ]+', seg_input)

        self.cm = defaultdict(int)
        length = len(self.seg_list)
        # 获取词图统计信息
        for i in range(length-1):
            start = self.seg_list[i]
            for j in range(i+1, i+self.span):
                if j >= length:
                    break

                end = self.seg_list[j]
                pair = tuple(sorted([start, end]))  # 消除排列顺序，避免重复
                self.cm[pair] += 1

        # 格式转换
        for pair, w in self.cm.items():
            self._add_edge(pair[0], pair[1], w)
        # 权值矩阵
        self._get_weight_matrix()
        # 计算得分
        self._calc_values()
        # 排序
        rank_list = self._rank()

        max_num = len(rank_list) if top == -1 else top
        return rank_list[:max_num]  # rank_list 总数不超过top也没关系


class TfIdf:
    def __init__(self, coll_sw=set([]), smooth=None, normalization=None):
        self.sw = coll_sw  # collection
        self.sot = smooth  # 暂不使用
        self.norm = normalization  # 暂不使用
        self.token2id = {}  # word: num
        self.id2token = {}  # num: word

    def _create_mapping(self):
        num = 0
        for sentence_list in self.input_list:
            for word in sentence_list:
                if word not in self.token2id:
                    self.token2id[word] = num
                    self.id2token[num] = word
                    num += 1

    def _get_weight_matrix(self):
        text_size = len(self.input_list)
        dimension = len(self.token2id)
        # TF
        freq_list = []  # scale: text_size, dimension
        for i in range(text_size):
            tmp_list = np.zeros(dimension, dtype='float32').tolist()
            for word in self.input_list[i]:
                tmp_list[self.token2id.get(word)] += 1.0
            freq_list.append(tmp_list)

        self.tf_matrix = np.array(freq_list, dtype='float32')

        # IDF
        idf = np.log(text_size / np.sum(self.tf_matrix > 0, axis=0, keepdims=True))
        self.matrix = self.tf_matrix * (idf + 0.001)  # broadcasting

    def _rank(self, top=-1):
        tf_idf_list = self.matrix.tolist()
        all_rank_list = []
        for tmp in tf_idf_list:
            new_list = [(self.id2token[i], round(value, 3)) for i, value in enumerate(tmp)]
            rank_list = sorted(new_list, key=lambda e: e[1], reverse=True)
            max_num = len(rank_list) if top == -1 else top
            all_rank_list.append(rank_list[:max_num])

        return all_rank_list

    def tf_idf_rank(self, text_list, top=-1, sw_flag=True):
        """
        run TfIdf task and get the result

        :param text_list: 1d list, consists of segment text
        :param top: int, the number of key words
        :param sw_flag: bool, decides whether to use stopwords or not
        :return: 2d list, consists of words and theirs scores by TfIdf
        """

        assert np.array(text_list).ndim == 1
        assert isinstance(top, int)

        if sw_flag and self.sw:  # 说明需要去停用词
            self.input_list = [[word for word in re.split('[ ]+', text) if word not in self.sw] for text in text_list]
        else:
            self.input_list = [re.split('[ ]+', text) for text in text_list]  # 2d list

        # 获得语料字典
        self._create_mapping()
        # 获得TfIdf矩阵
        self._get_weight_matrix()
        # 排序
        all_rank_list = self._rank(top)

        return all_rank_list


def init():
    # 读取停用词
    stopwords_list = []
    with open('./stopwords/zh_stop.txt', 'r', encoding='utf-8') as f:
        text = f.read().strip()
    zh_stop = re.split('[\n]+', text)
    stopwords_list.extend(zh_stop)

    with open('./stopwords/non_zh_stop.txt', 'r', encoding='utf-8') as f:
        text = f.read().strip()
    non_zh_stop = re.split('[\n]+', text)
    stopwords_list.extend(non_zh_stop)
    coll_stop = set(stopwords_list)

    # 创建类实例
    global ti
    ti = TfIdf(coll_sw=coll_stop)
    global tr
    tr = TextRank(coll_sw=coll_stop, d=0.85, window=5)  # 目前参数设定固定


def keywords_extraction(sequence_inputs, task, top=-1, sw_flag=True):
    """
    call such methods

    :param sequence_inputs: 1d list, consists of segment text
    :param task: string, denotes the current method for keywords extraction, including TfIdf and TextRank
    :param top: int, the number of key words
    :param sw_flag: bool, decides whether to use stopwords or not
    :return: 3d list, [[(word, score),], ]
    """

    assert np.array(sequence_inputs).ndim == 1
    assert isinstance(task, str)
    assert isinstance(top, int)

    all_rank_list = []
    if task == 'TfIdf':
        all_rank_list = ti.tf_idf_rank(sequence_inputs, top, sw_flag)

    elif task == 'TextRank':
        for seg_input in sequence_inputs:
            all_rank_list.append(tr.text_rank(seg_input, top, sw_flag))

    else:
        print('warning: invalid task')

    return all_rank_list


if __name__ == '__main__':
    init()
    seg_list = ['习近平  总书记  的  目标  今年  目标  小', '习近平  在  上海  发表  了  讲话', '习近平  总书记  来  上海  视察  。']
    result = keywords_extraction(seg_list, 'TextRank', 10, sw_flag=False)


