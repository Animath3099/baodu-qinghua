# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
# 结果按类内平均距离排序后输出
# 类内平均距离计算采用欧式距离
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


# 计算两点间距离
def __distance(p1, p2):
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)
    return pow(tmp, 0.5)


# 计算类内平均距离
def __avgdis(vecotrs, center):
    sum = 0
    for i in range(len(vecotrs)):
        sum += __distance(vecotrs[i], center)
    return sum / len(vecotrs)


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    sentence_label_avgdis_dict = defaultdict(tuple)
    for label, sentences in sentence_label_dict.items():
        vectors = sentences_to_vectors(sentences, model)  # 将当前label下的标题向量化
        center = kmeans.cluster_centers_[label]  # 获取当前label的质心
        avgdis = __avgdis(vectors, center)  # 计算类内平均距离
        sentence_label_avgdis_dict[label] = (sentences, avgdis)  # 同标签的标题和类内平均距离放到一起

    # 按类内平均距离排序后打印标签及对应标题（前10项）
    for label, (sentences, avgdis) in sorted(sentence_label_avgdis_dict.items(), key=lambda x: x[1][1]):
        print("cluster %s 类内平均距离:%f :" % (label, avgdis))
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
