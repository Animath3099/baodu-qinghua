import json
import copy
from ngram_language_model import NgramLanguageModel
"""
文本纠错demo
加载同音字字典
加载语言模型
基本原理：
对于文本中每一个字，判断在其同音字中是否有其他字，在替换掉该字时，能使得语言模型计算的成句概率提高
"""

class Corrector:
    def __init__(self, language_model):
        #语言模型
        self.language_model = language_model
        #候选字字典
        self.sub_dict = self.load_tongyinzi("tongyin.txt")
        #成句概率的提升超过阈值则保留修改
        self.threshold = 13

    #实际上不光是同音字，同形字等也可以加入，本质上是常用的错字
    def load_tongyinzi(self, path):
        tongyinzi_dict = {}
        with open(path, encoding="utf8") as f:
            for line in f:
                char, tongyin_chars = line.split()
                tongyinzi_dict[char] = list(tongyin_chars)
        return tongyinzi_dict

    #纠错逻辑
    def correction(self, string):
        # 纠错逻辑：通过将每个词替换为其同义词来实现成句概率的提升
        prob_old = self.language_model.predict(string)
        string = list(string)
        # new_string = string[:]
        best_prob = prob_old
        for i in range(len(string)):
            if self.sub_dict[string[i]]:
                tongyici = self.sub_dict[string[i]]
                best_word = string[i]
                for j in range(len(tongyici)):
                    cur_string = string[:]
                    cur_string[i] = tongyici[j]
                    cur_string = "".join(cur_string)
                    prob_cur = self.language_model.predict(cur_string)
                    # print(prob_new)
                    if prob_cur > best_prob:
                        best_word = tongyici[j]
                        best_prob = prob_cur
                string[i] = best_word
                if best_prob-prob_old >= self.threshold:
                    return "".join(string)
        return "".join(string)


corpus = open("财经.txt", encoding="utf8").readlines()
lm = NgramLanguageModel(corpus, 3)

cr = Corrector(lm)
string = "每国货币政册空间不大"  #美国货币政策空间不大
fix_string = cr.correction(string)
print("修改前：", string)
print("修改后：", fix_string)