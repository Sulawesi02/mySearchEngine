import jieba
import re
from string import punctuation
import pickle
import os
import math
import numpy as np
from sklearn.preprocessing import normalize
import xml.etree.ElementTree as ET

# 加载停用词表
stopwords_list = [line.strip() for line in open(r'data/hit_stopwords.txt', encoding='UTF-8').readlines()]


class Invert_term:
	# 单词出现过的文档id
	id = None
	# 单词出现次数
	count = None
	# 单词出现位置列表，每个元素为一个元组 (起始位置, 终止位置)
	positions = []
	# 词频 (TF)
	tf = None
	# 逆文档频率 (IDF)
	idf = None


	def __init__(self, id, max_term_count=1):
		self.id = id
		self.count = 0
		self.positions = []
		self.max_term_count = max_term_count
		self.tf = 0.0
		self.idf = 0.0


	def update_tf(self, term_count):
		# 使用归一化后的词频
		if self.max_term_count > 0:
			self.tf = self.count / self.max_term_count
		else:
			self.tf = 0.0

	def __repr__(self):
		return f'[id:{self.id}, count:{self.count}, positions:{self.positions}, tf:{self.tf}, idf:{self.idf}]'


# 计算逆文档频率（IDF）
def calculate_idf(inverted_index_dic, file_num):
	# 遍历每个词项，计算其 IDF
	for term, terms_list in inverted_index_dic.items():
		df = len(terms_list)  # 包含该词项的文档数量
		# 计算 IDF, 使用平滑处理
		idf = math.log((file_num / (df + 1)))
		for term_item in terms_list:
			term_item.idf = idf


# 构建 TF-IDF 矩阵
def build_tfidf_matrix(inverted_index_dic, file_num, vocab_dic):
	vocab_num = len(vocab_dic)
	
	tfidf_matrix = np.zeros((file_num, vocab_num))

	for term, terms_list in inverted_index_dic.items():
		term_index = vocab_dic[term]
		for term_item in terms_list:
			doc_id = term_item.id
			tfidf_matrix[doc_id, term_index] = term_item.tf * term_item.idf

	# 对文档向量（即 TF-IDF 矩阵的每一行）进行 L2 归一化处理
	tfidf_matrix = normalize(tfidf_matrix, norm='l2')

	return tfidf_matrix, vocab_dic


def build_inverted_index():
	# 倒排索引字典
	inverted_index_dic = dict()
	# 最大分词次数
	cut_num = 400
	doc_dir_path = 'data/file_set/'

	with open("data/index_set/id_title_dic.pkl", "rb") as f:
		id_title_dic = pickle.load(f)
	file_num = len(id_title_dic)

	# 遍历所有文档
	for doc_id in range(file_num):
		doc_file_path = os.path.join(doc_dir_path, f'{doc_id}.xml')
		try:
			with open(doc_file_path, 'r', encoding='utf-8') as f:
				content = f.read()
	
			# 解析 XML 字符串
			root = ET.fromstring(content)
			# 提取 <content> 标签的文本
			content_text = root.findtext('content', default='')
	
			# 检查 content_text 是否为空
			if not content_text:
				print(f"文件 {doc_file_path} 中 <content> 标签为空或不存在")
				continue
	
			# 去掉标点，小写，分词
			content_text = content_text.replace(' ', '')
			content_text = re.sub(r"[{}、，。！？·【】）》；;《“”（-：——]+".format(punctuation), "", content_text)
			content_text = content_text.lower()
			terms = jieba.lcut_for_search(content_text)[0:cut_num]
	
			# 去除停用词
			terms = [term for term in terms if term not in stopwords_list]
	
			# 文档总词数
			term_count = len(terms)
			# 文档中词项的最大出现次数
			max_term_count = max([terms.count(term) for term in set(terms)], default=1)
	
			# 遍历文档中的每个单词
			for term in terms:
				# 初始化词项
				curr_term = Invert_term(doc_id, max_term_count = max_term_count)
	
				# 查找当前单词在文档中的所有出现位置
				start = 0
				while True:
					start = content_text.find(term, start)
					if start == -1:
						break
					end = start + len(term)
					curr_term.positions.append((start, end))
					curr_term.count += 1
					start = end
	
				# 计算词频 (TF)
				curr_term.tf = curr_term.count / term_count
	
				# 如果单词不在倒排索引字典里，就将该单词加到倒排索引字典里
				if term not in inverted_index_dic.keys():
					inverted_index_dic[term] = [curr_term]
				# 如果单词在倒排字典里，进一步检查这个单词在当前文档（编号为 txt_id）的情况是否已经被添加到倒排字典中这个单词的txt_id列表里
				# 比如说当前文档单词列表 terms = [a,b,a],a之前已经加到倒排索引字典里，现在检查当前文档中的a是否被加到倒排索引字典里
				# 第一个a没有加到倒排字典中单词a的txt_id列表里，需要加入；第二个a已经加到倒排字典中单词a的txt_id列表里了，不需要加入
				else:
					# 倒排字典中这个单词所有已记录的txt_id列表
					file_id_list = [item.id for item in inverted_index_dic[term]]
					if doc_id not in file_id_list:
						inverted_index_dic[term].append(curr_term)
		except Exception as e:
			print(f"处理文件 {doc_file_path} 时出错: {e}")
			continue
			
	# 计算 IDF
	calculate_idf(inverted_index_dic, file_num)
	
	# 词汇表
	vocab = list(inverted_index_dic.keys())
	# 构建词汇表字典（词汇到索引的映射）
	vocab_dic = {term: i for i, term in enumerate(vocab)}

	# 构建 TF-IDF 矩阵
	tfidf_matrix, vocab_dic = build_tfidf_matrix(inverted_index_dic, file_num, vocab_dic)

	# print("inverted_index_dic", inverted_index_dic)
	# print("tfidf_matrix", tfidf_matrix)
	# print("vocab_dic", vocab_dic)

	# 倒排字典存到本地
	with open("data/index_set/inverted_index_dic.pkl", "wb") as tf:
		pickle.dump(inverted_index_dic, tf)
	print('store inverted_index_dic end')

	with open("data/index_set/tfidf_matrix.npy", "wb") as tf:
		np.save(tf, tfidf_matrix)
	print('store tfidf_matrix end')

	with open("data/index_set/vocab_dic.pkl", "wb") as tf:
		pickle.dump(vocab_dic, tf)
	print('store vocab_dic end')

if __name__ == '__main__':
	build_inverted_index()