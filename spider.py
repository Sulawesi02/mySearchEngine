import io
import re
import os
import requests
from bs4 import BeautifulSoup
from string import punctuation
from time import sleep
import pickle
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import uuid
from PIL import Image
import xml.etree.ElementTree as ET


# 配置ChromeDriver的路径
service = Service('C:/Users/XU/AppData/Local/Google/Chrome/Application/chromedriver.exe')

# 设置Chrome选项
chrome_options = Options()
chrome_options.add_argument('--headless')  # 无头模式
chrome_options.add_argument('--no-sandbox')  # 解决DevToolsActivePort文件不存在的报错
chrome_options.add_argument('--disable-dev-shm-usage')  # 解决资源限制问题

# 创建WebDriver实例
driver = webdriver.Chrome(service=service, options=chrome_options)


def content_to_xml(title, content):
	root = ET.Element("webpage")
	title_element = ET.SubElement(root, "title")
	title_element.text = title
	content_element = ET.SubElement(root, "content")
	content_element.text = content
	tree = ET.ElementTree(root)
	return ET.tostring(root, encoding='unicode')  # 返回 XML 字符串


def get_url_data(url, count, base_url, used_url_set, to_use_url_list, url_id_dic,
                 id_url_dic, id_title_dic, id_jumpUrls_dic, id_jumpAnchorText_dic
                 ):
	print("@ start crawl the web page", url)
	# 创建会话对象
	session = requests.Session()
	
	# 返回爬取到的网页
	try:
		html = session.get(url, timeout=5)
	except:
		to_use_url_list.remove(url)
		used_url_set.add(url)
		print('请求超时 无法爬取该页面//$^$//')
		return count
	
	html.encoding = 'utf-8'
	# 从网页抓取数据。
	soup = BeautifulSoup(html.text, 'lxml')
	# 网页标题
	curr_html_title = soup.find('title')
	if curr_html_title is None:
		to_use_url_list.remove(url)
		used_url_set.add(url)
		print('title为空 没有内容 无法爬取该页面//$^$//')
		return count
	else:
		html_title = curr_html_title.text
	
	# 如果标题为404
	if html_title == '404NotFound':
		to_use_url_list.remove(url)
		used_url_set.add(url)
		print('>>>404 Not Found, return')
		return count
	
	html_title = re.sub(r'[^\w\-_\. ]', '_', html_title)
	
	# 使用Selenium打开网页并截图
	try:
		driver.get(url)
		# 使用显式等待，等待页面的<body>元素加载完成
		wait = WebDriverWait(driver, 10)  # 设置最长等待时间为10秒
		wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

		# 获取整个页面的高度和宽度
		total_width = driver.execute_script("return document.documentElement.scrollWidth")
		total_height = driver.execute_script("return document.documentElement.scrollHeight")

		# 设置浏览器窗口大小以适应整个页面
		driver.set_window_size(total_width, total_height)

		# 捕获整个页面的截图
		screenshot_as_png = driver.get_screenshot_as_png()
		image_stream = io.BytesIO(screenshot_as_png)
		stitched_image = Image.open(image_stream)

		# 保存最终截图
		screenshot_filename = str(count) + '.png'
		screenshot_path = os.path.join('web/static/snap_shot', screenshot_filename)
		os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
		stitched_image.save(screenshot_path)
	except:
		print('网页截图失败')
	
	# 暂存网页内容
	content = ""
	
	# 网页的文字
	data = soup.select('body')
	for item in data:
		text = re.sub('[\r \n\t]', '', item.get_text())
		if text is None or text == '':
			continue
		content += text
	
	# 网页的超链接
	data = soup.select('a')
	# 字典，记录每个url页面可以跳转到的其他url，用 url 字符串标识
	id_jumpUrls_dic[count] = []
	# 字典，记录每个url页面可以跳转到其他url的锚文本，键是 id，值是对应的 url的锚文本
	id_jumpAnchorText_dic[count] = []
	for item in data:
		# 锚文本（去掉空格、换行符、制表符）
		text = re.sub("[\r \n\t]", '', item.get_text())
		# 如果text为空，就跳过当前循环
		if text is None or text == '':
			continue
		# 超链接的目标 url
		target_url = item.get('href')
		# 如果目标 url为空或不符合要求，就跳过当前循环
		if target_url is None or target_url == '' or re.search('java|void', target_url) != None:
			continue
		
		# 完善目标 url。加上前缀 http://cc.nankai.edu.cn/
		# 如果目标 url 不以 http、https 或者 www. 开头
		if re.match(r'^http|https|www\.', target_url) is None:
			# 如果 url 不以 / 开头
			if re.match(r'^/', target_url) is None:
				target_url = '/' + target_url  # 先给它加上 /
			target_url = base_url + target_url  # 再加上base_url
		
		# 将目标 url 加入接下来将要爬取的网页 url 的列表
		if target_url not in used_url_set and target_url not in to_use_url_list:
			to_use_url_list.append(target_url)
			id_jumpUrls_dic[count].append(target_url)  # 添加到url跳转列表
		
		# 将锚文本添加到锚文本索引字典
		if text:
			id_jumpAnchorText_dic[count].append(text)
	
	# 将内容转换为XML格式
	xml_str = content_to_xml(html_title, content)
	
	# 处理当前 url
	to_use_url_list.remove(url)
	used_url_set.add(url)
	url_id_dic[url] = count
	id_url_dic[count] = url
	id_title_dic[count] = html_title
	count = count + 1
	print("@ end crawl the web page", url)
	
	# 最后关闭会话
	session.close()
	return count, xml_str


def spider():
	# 南开大学新闻网的基础 url
	base_url = 'http://news.nankai.edu.cn'
	# 已经爬取过的网页 url 的集合
	used_url_set = set()
	# 接下来将要爬取的网页 url 的列表
	to_use_url_list = [
		'http://news.nankai.edu.cn',
	]
	# url 到 id 的映射
	url_id_dic = dict()
	# id 到 url 的映射
	id_url_dic = dict()
	# id 到 title 的映射
	id_title_dic = dict()
	# id 到可以跳转到的其他url的映射
	id_jumpUrls_dic = dict()
	# id 到可以跳转到其他url的锚文本的映射
	id_jumpAnchorText_dic = dict()
	
	# 循环从 to_use_url_list 中取出第一个 url，调用 get_url_data 函数进行爬取
	count = 0
	batch_size = 10  # 定义批量写入的数量
	content_buffer = {}  # 用于暂存一批网页内容的字典，键为count，值为对应的xml_file
	while count < 10000 and len(to_use_url_list) > 0:
		print("爬取网页个数：", count)
		url = to_use_url_list[0]
		result = get_url_data(url, count, base_url, used_url_set, to_use_url_list, url_id_dic,
		                      id_url_dic, id_title_dic, id_jumpUrls_dic, id_jumpAnchorText_dic)
		
		if isinstance(result, tuple):
			xml_file = result[1]
			content_buffer[count] = xml_file
			if len(content_buffer) >= batch_size:
				for doc_id, dodoc_content in content_buffer.items():
					try:
						doc_file_path = os.path.join('data/file_set/', f'{doc_id}.xml')
						
						# 使用 'w' 模式写入文本数据
						with open(doc_file_path, 'w', encoding='utf-8') as doc_context:
							doc_context.write(dodoc_content)  # 写入字符串形式的 XML
						
						print(f"成功保存文件: {doc_file_path}")
					except:
						print(f'保存文件 {doc_id} 失败')
				content_buffer = {}
			count = result[0]  # 将count更新放在这里，确保存储内容后再更新计数
		else:
			count = result
		
		# 爬虫礼仪(随机睡眠一个介于0到1秒之间的小数秒)
		sleep_time = random.uniform(0, 1)
		sleep(sleep_time)
	
	# 处理剩余不足batch_size数量的网页内容写入文件
	for doc_id, doc_content in content_buffer.items():
		try:
			doc_context = open(os.path.join('data/file_set/', f'{doc_id}.xml'),
			                   'w', encoding='utf-8')
			doc_context.write(doc_content)
			doc_context.close()
		except:
			print(f'保存文件 {doc_id} 失败')
	
	# 保存字典文件url_id_dic
	with open("data/index_set/url_id_dic.pkl", "wb") as tf:
		pickle.dump(url_id_dic, tf)
	print('save url_id_dic end')
	
	# 保存字典文件id_url_dic
	with open("data/index_set/id_url_dic.pkl", "wb") as tf:
		pickle.dump(id_url_dic, tf)
	print('save id_url_dic end')
	
	# 保存字典文件id_title_dic
	with open("data/index_set/id_title_dic.pkl", "wb") as tf:
		pickle.dump(id_title_dic, tf)
	print('save id_title_dic end')
	
	# 保存字典文件id_jumpUrls_dic
	with open("data/index_set/id_jumpUrls_dic.pkl", "wb") as tf:
		pickle.dump(id_jumpUrls_dic, tf)
	print('save id_jumpUrls_dic end')
	
	# 保存字典文件id_jumpAnchorText_dic
	with open("data/index_set/id_jumpAnchorText_dic.pkl", "wb") as tf:
		pickle.dump(id_jumpAnchorText_dic, tf)
	print('save id_jumpAnchorText_dic end')


if __name__ == '__main__':
	spider()
