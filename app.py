import io
import os
import re
import pickle
import jieba
import json
import sqlite3
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, render_template_string
from string import punctuation
from fnmatch import fnmatch
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from content_inverted_index import Invert_term
from datetime import datetime, time
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import math
from datetime import datetime


app = Flask(__name__, static_folder='web/static', template_folder='web/templates')

# 设置secret_key，使用os.urandom生成一个随机的24字节字符串
app.secret_key = os.urandom(24)

# 数据库文件
DB_FILE = "data/users.db"

# 数据集和索引集
doc_dir_path = 'data/file_set/'
inverted_index_path = "data/index_set/inverted_index_dic.pkl"
id_url_path = "data/index_set/id_url_dic.pkl"
id_title_path = "data/index_set/id_title_dic.pkl"
tfidf_matrix_path = "data/index_set/tfidf_matrix.npy"
vocab_path = "data/index_set/vocab_dic.pkl"
id_pagerank_path = "data/index_set/id_pagerank_dic.pkl"

def create_tables():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            age INTEGER,
            sex TEXT,
            academy TEXT
        )
    ''')
    
    # 创建查询日志表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            query_str TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()
    

@app.route('/')
def home():
    return render_template('signin.html')

# 登录
@app.route('/signin', methods=['GET','POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # 连接数据库
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # 查询用户是否存在且密码是否正确
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()

        # 检查是否找到了用户
        if user is not None:
            session['user_id'] = user[0]
            session['user_age'] = user[4]
            session['user_academy'] = user[6]

            conn.close()
            return render_template('query.html')

        else:
            flash('用户名或密码无效，请重试！', 'error')
            conn.close()
            return redirect(url_for('signin'))
        
    return render_template('signin.html')

# 注册
@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        age = int(request.form['age'])
        sex = request.form['sex']
        academy = request.form['academy']
        
        # 连接数据库
        conn = sqlite3.connect(DB_FILE)
        cursor =conn.cursor()
        
        # 检查用户名是否已存在
        cursor.execute('SELECT * FROM users WHERE username=?', (username,))
        if cursor.fetchone() is not None:
            flash('用户名已被注册，请换一个！', 'error')
            return redirect(url_for('signup'))
            
        # 插入新用户
        cursor.execute("INSERT INTO users (username, email, password, age, sex, academy) VALUES (?,?,?,?,?,?)",
                       (username, email, password, age, sex, academy))
        conn.commit()
        conn.close()
        flash('注册成功!', 'success')
    
        return redirect(url_for('signin'))
    
    return render_template('signup.html')


# 更新数据库的 query_logs 表
def update_query_logs(query_str):
    # 获取当前用户的ID
    user_id = session.get('user_id')
    
    # 获取当前时间
    now_time = datetime.now().isoformat()
    
    if not user_id:
        return jsonify({
            'success': False,
            'message': '用户未登录'
        }), 401
    
    # 连接数据库
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 检查是否已存在记录
    cursor.execute('''
        SELECT COUNT(*) FROM query_logs WHERE user_id = ? AND query_str = ?
    ''', (user_id, query_str))
    
    exists = cursor.fetchone()[0] > 0
    
    if exists:
        # 如果存在，更新 timestamp
        cursor.execute('''
            UPDATE query_logs
            SET timestamp = ?
            WHERE user_id = ? AND query_str = ?
        ''', (now_time, user_id, query_str))
    else:
        # 如果不存在，插入新纪录
        cursor.execute('''
            INSERT INTO query_logs (user_id, query_str, timestamp)
            VALUES (?, ?, ?)
        ''', (user_id, query_str, now_time))
    
    conn.commit()
    conn.close()


# 加载倒排索引、URL 映射、标题映射、TF-IDF 矩阵、词汇映射和PageRank映射
def load_data():
    with open(inverted_index_path, "rb") as f:
        inverted_index_dic = pickle.load(f)
    with open(id_url_path, "rb") as f:
        id_url_dic = pickle.load(f)
    with open(id_title_path, "rb") as f:
        id_title_dic = pickle.load(f)
    tfidf_matrix = csr_matrix(np.load(tfidf_matrix_path))
    with open(vocab_path, "rb") as f:
        vocab_dic = pickle.load(f)
    with open(id_pagerank_path, "rb") as f:
        id_pagerank_dic = pickle.load(f)
    return inverted_index_dic, id_url_dic, id_title_dic, tfidf_matrix, vocab_dic, id_pagerank_dic


# 预处理
def pretreatment(query_str):
    # 去掉标点，小写，分词
    query_str = query_str.replace(' ', '')
    content = re.sub(r"[{}、，。！？·【】）》；;《“”（-]+".format(punctuation), "", query_str)
    content = content.lower()
    query_terms = jieba.lcut_for_search(content)
    
    return query_terms

# 从数据库中获取用户的查询日志
def fetch_query_history(user_id):
    try:
        if not user_id:
            return []

        # 连接数据库
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row  # 设置行工厂为 sqlite3.Row
        cursor = conn.cursor()
        
        # 查询该用户的最近10条查询历史（包含时间戳）
        cursor.execute('''
            SELECT query_str, timestamp
            FROM query_logs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (user_id,))
        query_logs = cursor.fetchall()
        
        conn.close()
        
        # 将查询历史转换为列表
        history = []
        for log in query_logs:
            if not log['timestamp']:  # 如果时间戳为空，跳过该条日志
                continue
            
            try:
                # 解析时间戳，使用 ISO 8601 格式
                timestamp = datetime.strptime(log['timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
                history.append({
                    'query_str': log['query_str'],
                    'timestamp': timestamp  # 将解析后的时间戳存储为 datetime 对象
                })
            except ValueError as e:
                print(f"Error parsing timestamp '{log['timestamp']}': {e}")
                continue  # 如果解析失败，跳过该条日志
        
        return history
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []

# 将查询字符串转换为查询向量
def query_to_vector(query_terms, vocab_dic, inverted_index_dic, user_age=None, user_academy=None, query_logs=None,
                    alpha=0.9, beta=0.05, gamma=0.03, delta=0.02):
    """
    用原始查询词项、用户信息（年龄、学院）和查询日志构建最终的查询向量，按照给定权重融合各部分向量，
    其中年龄按学历/身份分类、查询日志采用时间衰减原则处理。
    """
    
    # 初始化查询向量 query_vector 和词频字典 query_term_freq
    query_vector = np.zeros(len(vocab_dic))
    query_term_freq = {}  # 存储查询字符串词项的词频
    
    # 统计查询词项列表中每个词项的频次
    for term in query_terms:
        if term in vocab_dic:
            query_term_freq[term] = query_term_freq.get(term, 0) + 1
        

    # 用户年龄转化为向量（词频设为1）
    age_vector = np.zeros(len(vocab_dic))
    if user_age:
        age_group = ""
        if 18 <= user_age <= 22:
            age_group = "本科生"
        elif 22 < user_age <= 25:
            age_group = "研究生"
        elif 25 < user_age <= 30:
            age_group = "博士生"
        elif user_age >= 30:
            age_group = "老师"
        
        age_term = f"age_{age_group}"
        if age_term in vocab_dic:
                    age_vector[vocab_dic[age_term]] = 1
    
    # 对年龄向量进行 L2 归一化处理
    age_vector = normalize(age_vector.reshape(1, -1), norm='l2').flatten()

    # 用户学院转化为向量（词频设为1）
    academy_vector = np.zeros(len(vocab_dic))
    if user_academy:
        academy_term = f"academy_{user_academy}"
        if academy_term in vocab_dic:
                    academy_vector[vocab_dic[academy_term]] = 1
    
    # 对学院向量进行 L2 归一化处理
    academy_vector = normalize(academy_vector.reshape(1, -1), norm='l2').flatten()
    
    # 查询日志转化为向量（取最近10条，采用时间衰减原则）
    log_vector = np.zeros(len(vocab_dic))
    if query_logs:
        current_time = datetime.now()  # 获取当前时间
        decay_rate = 0.1  # 时间衰减率
        for log in query_logs[:10]:  # 只取最近10条查询日志
            log_query_str = log['query_str']
            log_terms = pretreatment(log_query_str)
            time_diff = (current_time - log['timestamp']).total_seconds() / (60 * 60 * 24)  # 时间差以天为单位
            weight = math.exp(-decay_rate * time_diff)  # 指数衰减
            
            for term in log_terms:
                if term in vocab_dic:
                    log_vector[vocab_dic[term]] += weight
    
    # 对查询日志向量进行 L2 归一化处理
    log_vector = normalize(log_vector.reshape(1, -1), norm='l2').flatten()
    
    # 使用 TF * IDF 构建查询向量
    for term, freq in query_term_freq.items():
        term_index = vocab_dic[term]
        if term in inverted_index_dic:
            # 使用查询词项列表中的词频作为 TF
            tf = freq / len(query_terms)
            # 获取词项的 idf 值
            # 理论上不同文档中的同一个词项，它们的 idf 值都是相同的
            # 所以选择了第一个包含该词项的文档项的 idf 值
            idf = inverted_index_dic[term][0].idf
            query_vector[term_index] = tf * idf
        else:
            print(f"查询词项 '{term}' 不在词汇表中")
            query_vector[term_index] = 0

    # 对查询向量进行 L2 归一化处理
    query_vector = normalize(query_vector.reshape(1, -1), norm='l2').flatten()
	
    # 构建最终查询向量，按照给定权重融合各部分向量
    final_query_vector = (
        alpha * query_vector + 
        beta * age_vector + 
        gamma * academy_vector + 
        delta * log_vector
    )

	# 对最终查询向量进行 L2 归一化处理
    final_query_vector = normalize(final_query_vector.reshape(1, -1), norm='l2').flatten()
	
    return final_query_vector

# 使用倒排索引筛选相关文档
def get_relevant_documents(query_terms, inverted_index_dic):
    relevant_doc_ids = set()
    
    for item in query_terms:
        if item in inverted_index_dic.keys():
            for entry in inverted_index_dic[item]:
                relevant_doc_ids.add(entry.id)
                
    return list(relevant_doc_ids)

# 计算相似度并进行相关性排序
def get_similar_documents(query_vector, tfidf_matrix, id_pagerank_dic, relevant_doc_ids,
                          relevance_threshold=0.02, alpha=0.8):
    # 筛选相关文档的 TF - IDF 矩阵
    relevant_tfidf_matrix = tfidf_matrix[relevant_doc_ids]
    
    # 计算查询向量与相关文档之间的余弦相似度
    cosine_similarity_scores = cosine_similarity([query_vector], relevant_tfidf_matrix).flatten()
    
    # 获取相关文档的 PageRank 分数
    pagerank_scores = np.array([id_pagerank_dic.get(doc_id, 0) for doc_id in relevant_doc_ids])
    
    # 计算加权组合得分
    final_scores = alpha * cosine_similarity_scores + (1 - alpha) * pagerank_scores

    # 按得分从高到低排序
    sorted_indexes = np.argsort(final_scores)[::-1]

    # 返回对应的文档 ID
    similar_docs = []
    for i in sorted_indexes:
        if final_scores[i] > relevance_threshold:
            doc_id = relevant_doc_ids[i]
            similar_docs.append(doc_id)
        else:
            break

    return similar_docs

# 普通查询
@app.route('/common_query', methods=['POST'])
def common_query():
    try:
        # 解析前端发送的JSON数据
        data = json.loads(request.data.decode('utf-8'))
        query_str = data.get('query_str')
        
        # 更新数据库的 query_logs 表
        update_query_logs(query_str)
        
        # 加载相关数据
        inverted_index_dic, id_url_dic, id_title_dic, tfidf_matrix, vocab_dic, id_pagerank_dic = load_data()
        
        # 去掉标点，小写，分词
        query_terms = pretreatment(query_str)
        
        # 获取当前用户信息
        user_id = session.get('user_id') # 用户ID
        user_age = session.get('user_age') # 用户年龄
        user_academy = session.get('user_academy') # 用户学院

        # 获取当前用户的查询日志
        query_logs = fetch_query_history(user_id)

        # 将查询字符串转换为查询向量
        query_vector = query_to_vector(query_terms, vocab_dic, inverted_index_dic, user_age=user_age, user_academy=user_academy,
                                       query_logs=query_logs)

        # 使用倒排索引筛选相关文档
        relevant_doc_ids = get_relevant_documents(query_terms, inverted_index_dic)
        if not relevant_doc_ids:
            return jsonify({
                'success': True,
                'results': []
            })

        # 计算相似度并获取最相关的文档
        similar_docs = get_similar_documents(query_vector, tfidf_matrix, id_pagerank_dic, relevant_doc_ids)

        # 构建最终结果
        results = []
        for doc_id in similar_docs:
            doc_file_path = os.path.join(doc_dir_path, f'{doc_id}.xml')
            
            snap_shot_path = url_for('static', filename=f'snap_shot/{doc_id}.png', _external=True)
            
            try:
                with open(doc_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析 XML 字符串
                root = ET.fromstring(content)
                # 提取 <content> 标签的文本
                content_text = root.findtext('content', default='')

                if content_text is not None:
                    content_text = content_text.strip()[:120] + "..."
                else:
                    content_text = "未找到 <context> 标签"
            except FileNotFoundError:
                content_text = "未找到对应文档内容"
            
            result_item = {
                'doc_id': doc_id,
                'title': id_title_dic[doc_id],
                'content': content_text,
                'url': id_url_dic[doc_id],
                'snap_shot_url':snap_shot_path
            }
            results.append(result_item)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        # 捕获所有异常并返回错误信息
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# 站内查询
@app.route('/instation_query', methods=['POST'])
def instation_query():
    try:
        # 解析前端发送的JSON数据
        data = json.loads(request.data.decode('utf-8'))
        query_str = data.get('query_str')
        query_url = data.get('query_url')

        # 更新数据库的 query_logs 表
        update_query_logs(query_str)
        
        # 加载相关数据
        inverted_index_dic, id_url_dic, id_title_dic, tfidf_matrix, vocab_dic, id_pagerank_dic = load_data()
        
        # 去掉标点，小写，分词
        query_terms = pretreatment(query_str)
        
        # 将查询字符串转换为查询向量
        query_vector = query_to_vector(query_terms, vocab_dic, inverted_index_dic)
        
        # 使用倒排索引筛选相关文档
        relevant_doc_ids = get_relevant_documents(query_terms, inverted_index_dic)
        if not relevant_doc_ids:
            return jsonify({
                'success': True,
                'results': []
            })

        # 计算相似度并获取最相关的文档
        similar_docs = get_similar_documents(query_vector, tfidf_matrix, id_pagerank_dic, relevant_doc_ids)
        print(similar_docs)
        # 构建最终结果
        results = []
        for doc_id in similar_docs:
            doc_file_path = os.path.join(doc_dir_path, f'{doc_id}.xml')
            
            snap_shot_path = url_for('static', filename=f'snap_shot/{doc_id}.png', _external=True)
            
            try:
                with open(doc_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析 XML 字符串
                root = ET.fromstring(content)
                # 提取 <content> 标签的文本
                content_text = root.findtext('content', default='')

                if content_text is not None:
                    content_text = content_text.strip()[:120] + "..."
                else:
                    content_text = "未找到 <context> 标签"
            except FileNotFoundError:
                content_text = "未找到对应文档内容"
                
            # 解析 query_url 和 id_url_dic[doc_id]
            parsed_query_url = urlparse(query_url) if query_url else None
            parsed_doc_url = urlparse(id_url_dic[doc_id])

            # 检查 URL 是否匹配
            if (parsed_query_url and (parsed_query_url.netloc == parsed_doc_url.netloc) and
                                 (parsed_doc_url.path.startswith(parsed_query_url.path))):
                result_item = {
                    'doc_id': doc_id,
                    'title': id_title_dic[doc_id],
                    'content': content_text,
                    'url': id_url_dic[doc_id],
                    'snap_shot_url':snap_shot_path
                }
                results.append(result_item)

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        # 捕获所有异常并返回错误信息
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# 文档查询
@app.route('/document_query', methods=['POST'])
def document_query():
    try:
        # 解析前端发送的JSON数据
        data = json.loads(request.data.decode('utf-8'))
        query_str = data.get('query_str')
        
        # 更新数据库的 query_logs 表
        update_query_logs(query_str)
        
        # 加载相关数据
        inverted_index_dic, id_url_dic, id_title_dic, tfidf_matrix, vocab_dic, id_pagerank_dic = load_data()
        
        # 去掉标点，小写，分词
        query_terms = pretreatment(query_str)
        
        # 将查询字符串转换为查询向量
        query_vector = query_to_vector(query_terms, vocab_dic, inverted_index_dic)
        
        # 使用倒排索引筛选相关文档
        relevant_doc_ids = get_relevant_documents(query_terms, inverted_index_dic)
        if not relevant_doc_ids:
            return jsonify({
                'success': True,
                'results': []
            })

        # 计算相似度并获取最相关的文档
        similar_docs = get_similar_documents(query_vector, tfidf_matrix, id_pagerank_dic, relevant_doc_ids)

        # 构建最终结果
        results = []
        for doc_id in similar_docs:
            doc_file_path = os.path.join(doc_dir_path, f'{doc_id}.xml')
            
            snap_shot_path = url_for('static', filename=f'snap_shot/{doc_id}.png', _external=True)
            
            try:
                with open(doc_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析 XML 字符串
                root = ET.fromstring(content)
                # 提取 <content> 标签的文本
                content_text = root.findtext('content', default='')

                if content_text is not None:
                    content_text = content_text.strip()[:120] + "..."
                else:
                    content_text = "未找到 <context> 标签"
            except FileNotFoundError:
                content_text = "未找到对应文档内容"
            
            result_item = {
                'doc_id': doc_id,
                'title': id_title_dic[doc_id],
                'content': content_text,
                'url': id_url_dic[doc_id],
                'snap_shot_url':snap_shot_path
            }
            results.append(result_item)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        # 捕获所有异常并返回错误信息
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# 使用动态规划筛选短语连续的文档
def get_relevant_documents_dp(query_terms, inverted_index_dic, id_title_dic):
    # 维护一个二维数组 dp[i][j]，其中 i 表示查询的词项的索引，j 表示文档的 ID。
    # dp[i][j] 的值为 True，表示查询的第 i 个词项可以与前一个词项在文档 j 中连续匹配
    file_num = len(id_title_dic)
    dp = [[False] * file_num for _ in range(len(query_terms))]
    
    # 处理第一个分词
    for term_entry in inverted_index_dic.get(query_terms[0], []):
        doc_id = term_entry.id
        positions = term_entry.positions
        
        # 将所有包含该词项的文档 dp[0][id] 设置为 True
        for start_pos, end_pos in positions:
            dp[0][doc_id] = True
    
    # 遍历后续分词
    for i in range(1, len(query_terms)):
        for term_entry in inverted_index_dic.get(query_terms[i], []):
            doc_id = term_entry.id
            positions = term_entry.positions
            
            # 检查当前分词结果的起始位置是否不晚于前一个分词结果的结束位置
            for start_pos, end_pos in positions:
                if dp[i - 1][doc_id]:
                    prev_positions = [pos for pos in inverted_index_dic[query_terms[i - 1]] if pos.id == doc_id]
                    for prev_start_pos, prev_end_pos in prev_positions[0].positions:
                        if start_pos <= prev_end_pos:
                            dp[i][doc_id] = True
                            break
    # 匹配成功的文档
    matched_doc_ids = [i for i in range(file_num) if dp[len(query_terms) - 1][i]]
    
    return matched_doc_ids

# 短语查询
@app.route('/phrase_query', methods=['POST'])
def phrase_query():
    try:
        # 解析前端发送的JSON数据
        data = json.loads(request.data.decode('utf-8'))
        query_str = data.get('query_str')
        
        # 更新数据库的 query_logs 表
        update_query_logs(query_str)
        
        # 加载相关数据
        inverted_index_dic, id_url_dic, id_title_dic, tfidf_matrix, vocab_dic, id_pagerank_dic = load_data()
        
        # 去掉标点，小写，分词
        query_terms = pretreatment(query_str)
        
        # 将查询字符串转换为查询向量
        query_vector = query_to_vector(query_terms, vocab_dic, inverted_index_dic)
        
        # 使用动态规划筛选短语连续的文档
        relevant_doc_ids = get_relevant_documents_dp(query_terms, inverted_index_dic, id_title_dic)
        if not relevant_doc_ids:
            return jsonify({
                'success': True,
                'results': []
            })

        # 计算相似度并获取最相关的文档
        similar_docs = get_similar_documents(query_vector, tfidf_matrix, id_pagerank_dic, relevant_doc_ids)

        # 构建最终结果
        results = []
        for doc_id in similar_docs:
            doc_file_path = os.path.join(doc_dir_path, f'{doc_id}.xml')
            
            snap_shot_path = url_for('static', filename=f'snap_shot/{doc_id}.png', _external=True)
            
            try:
                with open(doc_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析 XML 字符串
                root = ET.fromstring(content)
                # 提取 <content> 标签的文本
                content_text = root.findtext('content', default='')

                if content_text is not None:
                    content_text = content_text.strip()[:120] + "..."
                else:
                    content_text = "未找到 <context> 标签"
            except FileNotFoundError:
                content_text = "未找到对应文档内容"
            
            result_item = {
                'doc_id': doc_id,
                'title': id_title_dic[doc_id],
                'content': content_text,
                'url': id_url_dic[doc_id],
                'snap_shot_url':snap_shot_path
            }
            results.append(result_item)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        # 捕获所有异常并返回错误信息
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# 通配查询
@app.route('/wildcard_query', methods=['POST'])
def wildcard_query():
    try:
        # 解析前端发送的JSON数据
        data = json.loads(request.data.decode('utf-8'))
        query_str = data.get('query_str')
        
        # 更新数据库的 query_logs 表
        update_query_logs(query_str)
        
        # 加载相关数据
        inverted_index_dic, id_url_dic, id_title_dic, tfidf_matrix, vocab_dic, id_pagerank_dic = load_data()
        
        # 去掉标点，小写，分词
        query_terms = pretreatment(query_str)
        print(query_terms)
        # 通配符查询
        for item in list(inverted_index_dic.keys()):
            if fnmatch(item, query_str):
                query_terms.append(item)
        
        # 如果没有匹配到任何词项，直接返回空结果
        if not query_terms:
            return jsonify({
                'success': True,
                'results': []
            })
            
        # 将查询字符串转换为查询向量
        query_vector = query_to_vector(query_terms, vocab_dic, inverted_index_dic)

        # 使用倒排索引筛选相关文档
        relevant_doc_ids = get_relevant_documents(query_terms, inverted_index_dic)
        if not relevant_doc_ids:
            return jsonify({
                'success': True,
                'results': []
            })

        # 计算相似度并获取最相关的文档
        similar_docs = get_similar_documents(query_vector, tfidf_matrix, id_pagerank_dic, relevant_doc_ids)

        # 构建最终结果
        results = []
        for doc_id in similar_docs:
            doc_file_path = os.path.join(doc_dir_path, f'{doc_id}.xml')
            
            snap_shot_path = url_for('static', filename=f'snap_shot/{doc_id}.png', _external=True)
            
            try:
                with open(doc_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析 XML 字符串
                root = ET.fromstring(content)
                # 提取 <content> 标签的文本
                content_text = root.findtext('content', default='')

                if content_text is not None:
                    content_text = content_text.strip()[:120] + "..."
                else:
                    content_text = "未找到 <context> 标签"
            except FileNotFoundError:
                content_text = "未找到对应文档内容"
            
            result_item = {
                'doc_id': doc_id,
                'title': id_title_dic[doc_id],
                'content': content_text,
                'url': id_url_dic[doc_id],
                'snap_shot_url':snap_shot_path
            }
            results.append(result_item)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        # 捕获所有异常并返回错误信息
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/get_html/<int:doc_id>', methods=['GET'])
def get_html(doc_id):
    try:
        # 加载相关数据
        inverted_index_dic, id_url_dic, id_title_dic, tfidf_matrix, vocab_dic, id_pagerank_dic = load_data()
        
        # 获取当前文档的标题
        current_page_title = id_title_dic.get(doc_id, "未知文档")

        # 获取当前用户信息
        user_id = session.get('user_id')  # 用户ID
        user_age = session.get('user_age')  # 用户年龄
        user_academy = session.get('user_academy')  # 用户学院

        # 获取当前用户的查询日志
        query_logs = fetch_query_history(user_id)

        # 将当前网页的标题转换为向量
        current_page_terms = pretreatment(current_page_title)
        current_page_vector = query_to_vector(current_page_terms, vocab_dic, inverted_index_dic, user_age=user_age, user_academy=user_academy,
                                              query_logs=query_logs)

        # 使用倒排索引筛选相关文档
        relevant_doc_ids = get_relevant_documents(current_page_terms, inverted_index_dic)
        if not relevant_doc_ids:
            return []

        # 计算相似度并获取最相关的文档
        similar_docs = get_similar_documents(current_page_vector, tfidf_matrix, id_pagerank_dic, relevant_doc_ids)

        # 构建推荐文档列表
        recommend_docs = []
        for ddoc_id in similar_docs[:6]:  # 取前6个，以确保排除当前文档后仍有5个
            if ddoc_id == doc_id:  # 排除当前文档
                continue
            recommend_docs.append({
                'title': id_title_dic[ddoc_id],
                'doc_id': ddoc_id
            })

        # 构建 XML 文件路径
        doc_file_path = os.path.join(doc_dir_path, f'{doc_id}.xml')

        # 检查文件是否存在
        if not os.path.exists(doc_file_path):
            return jsonify({
                'success': False,
                'message': 'XML 文件不存在'
            }), 404

        # 读取 XML 文件内容
        with open(doc_file_path, 'r', encoding='utf-8') as f:
            doc_content = f.read()

        # 解析 XML
        root = ET.fromstring(doc_content)

        # 提取 <title> 和 <content> 标签的内容
        title = root.findtext('title', default='无标题')
        content = root.findtext('content', default='无内容')
        
        # 获取原文链接
        original_url = id_url_dic.get(doc_id, '#')
        
        # 生成 HTML 内容
        html_content = render_template_string(
            """
			<!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{{ title }}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333; }
                    p { line-height: 1.6; }
                    a { color: #007BFF; text-decoration: none; }
                    a:hover { text-decoration: underline; }
                    ul { list-style-type: disc; padding-left: 20px; }
                    li { margin-bottom: 10px; }
                </style>
            </head>
            <body>
                <h1>{{ title }}</h1>
                <p><a href="{{ original_url }}" target="_blank">原文链接</a></p>
                <p>{{ content }}</p>
                <h2>相关推荐</h2>
                <ul>
                    {% for rec in recommend_docs %}
                        <li>
                            <a href="{{ url_for('get_html', doc_id=rec['doc_id']) }}" target="_blank">{{ rec['title'] }}</a>
                        </li>
                    {% endfor %}
                </ul>
            </body>
            </html>
			""",
            title=title,
            original_url=original_url,
            content=content,
            recommend_docs=recommend_docs,
            url_for=url_for
        )
        
        # 返回生成的 HTML 内容
        return html_content

    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# 获取查询日志
@app.route('/get_query_history', methods=['GET'])
def get_query_history():
    try:
        # 获取当前用户的ID
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'message': '用户未登录'
            }), 401
    
        # 连接数据库
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row  # 设置行工厂为 sqlite3.Row
        cursor = conn.cursor()
        
        # 查询该用户的查询历史
        cursor.execute('''
            SELECT id, query_str, timestamp
            FROM query_logs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 10  -- 显示最近10条查询历史
        ''', (user_id,))
        query_logs = cursor.fetchall()
        
        conn.commit()
        conn.close()
        
        # 将查询历史转换为JSON格式
        history = []
        for log in query_logs:
            history.append({
                'id': log['id'],
                'query_str': log['query_str']
            })
        
        return jsonify({
            'success': True,
            'history': history
        })
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return jsonify({
            'success': False,
            'message': '数据库查询失败'
        }), 500
    

# 删除查询日志项
@app.route('/delete_query_history/<int:history_id>', methods=['DELETE'])
def delete_query_history(history_id):
    try:
        # 获取当前用户的ID
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'message': '用户未登录'
            }), 401
    
        # 连接数据库
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # 删除该用户的查询历史
        cursor.execute('''
                DELETE FROM query_logs
                WHERE id = ? and user_id = ?
            ''', (history_id,user_id,))
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'message': '删除成功'
        })

        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': '删除失败'
        }), 500

if __name__ == '__main__':
    create_tables()
    # app.run(debug = True)
    app.run(host="127.0.0.1", port=5001, debug = True)