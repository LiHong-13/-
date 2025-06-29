import re
import os
import string
import matplotlib.pyplot as plt
import requests
import json
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis.gensim
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns


# ------------------- 中文字体设置 -------------------
def setup_font():
    """设置macOS系统中文字体"""
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 尝试系统字体
    system_fonts = ["Heiti TC", "PingFang TC", "STHeiti"]
    for font in system_fonts:
        if font in plt.rcParams["font.family"]:
            plt.rcParams["font.family"] = font
            print(f"使用系统字体: {font}")
            return

    # 尝试字体文件路径
    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",  # 苹方字体
        "/System/Library/Fonts/STHeiti Light.ttc"  # 黑体-简
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                from matplotlib.font_manager import FontProperties
                font = FontProperties(fname=font_path)
                plt.rcParams["font.family"] = [font.get_name()]
                print(f"使用字体文件: {font_path}, 字体名称: {font.get_name()}")
                return
            except Exception as e:
                print(f"加载字体文件 {font_path} 失败: {e}")

    print("无法设置中文字体，使用默认字体")


setup_font()
print(f"当前字体设置: {plt.rcParams['font.family']}")

# ------------------- 停用词列表 -------------------
CUSTOM_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'could',
    'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
    'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his',
    'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'it', 'it\'s', 'its', 'itself',
    'let\'s', 'me', 'more', 'most', 'my', 'myself', 'nor', 'of', 'on', 'once', 'only', 'or', 'other', 'ought', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'so', 'some',
    'such',
    'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these',
    'they',
    'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
    'very', 'was',
    'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s',
    'which', 'while',
    'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'would', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your',
    'yours',
    'yourself', 'yourselves'
}


# ------------------- 工具初始化 -------------------
def init_tools():
    """初始化NLTK工具"""
    stemmer = PorterStemmer()
    return stemmer


# ------------------- 数据处理 -------------------
def read_data(file_path):
    """读取数据文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # 跳过表头
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    data.append(parts[1])
    except Exception as e:
        print(f"读取数据出错: {e}")
        data = ["示例文本1", "示例文本2", "示例文本3"]
    print(f"已读取 {len(data)} 条文本")
    return data


def preprocess_text(texts, stemmer):
    """数据预处理"""
    processed_texts = []
    translator = str.maketrans('', '', string.punctuation)
    empty_count = 0

    print("数据预处理中...")
    for i, text in enumerate(texts):
        # 转换为小写并去除标点
        text = text.lower().translate(translator)

        # 分词
        try:
            words = word_tokenize(text)
        except:
            words = text.split()

        # 过滤停用词和短词
        words = [word for word in words if word not in CUSTOM_STOPWORDS and len(word) > 1]

        # 词干提取
        words = [stemmer.stem(word) for word in words]

        processed_texts.append(words)

        # 打印前5条调试信息
        if i < 5:
            print(f"原始文本 {i + 1} (前50字符): {text[:50]}...")
            print(f"处理后词汇: {words}")

        # 统计空文本
        if not words:
            empty_count += 1

    print(f"预处理完成，空文本数量: {empty_count}/{len(texts)}")
    return processed_texts


# ------------------- LDA模型训练 -------------------
def train_lda(processed_texts, num_topics=5):
    """训练LDA模型"""
    # 过滤空文档
    non_empty = [text for text in processed_texts if text]
    print(f"过滤后有效文档: {len(non_empty)}/{len(processed_texts)}")

    if not non_empty:
        raise ValueError("所有文档预处理后均为空，无法训练模型")

    # 构建词典
    dictionary = Dictionary(non_empty)
    dictionary.filter_extremes(no_below=2, no_above=0.7)

    # 生成语料库
    corpus = [dictionary.doc2bow(text) for text in non_empty]
    if not corpus or sum(1 for doc in corpus if doc) == 0:
        raise ValueError("语料库中没有有效词汇")

    # 训练模型
    print(f"开始训练LDA模型，主题数: {num_topics}")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    print("LDA模型训练完成")
    return lda_model, corpus, dictionary, non_empty


# ------------------- 可视化分析 -------------------
def visualize(lda_model, corpus, dictionary, texts):
    """生成可视化结果"""
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    # LDA交互可视化
    try:
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis, 'visualizations/lda_visualization.html')
        print("LDA交互可视化已保存至 visualizations/lda_visualization.html")
    except Exception as e:
        print(f"生成LDA可视化失败: {e}")

    # 词云图
    for i in range(lda_model.num_topics):
        top_words = lda_model.show_topic(i, topn=20)
        wordcloud_text = " ".join([word for word, _ in top_words])

        # 指定词云字体
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            font_path='/System/Library/Fonts/PingFang.ttc'  # macOs苹方字体
        ).generate(wordcloud_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"主题 {i} 的词云")
        plt.axis('off')
        plt.savefig(f'visualizations/topic_{i}_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 热力图
    doc_topic_matrix = []
    for doc in corpus:
        topic_dist = lda_model.get_document_topics(doc, minimum_probability=0)
        doc_topic_matrix.append([prob for _, prob in topic_dist])

    if doc_topic_matrix:
        doc_topic_df = pd.DataFrame(doc_topic_matrix)
        plt.figure(figsize=(10, 8))
        sns.heatmap(doc_topic_df, cmap='viridis')
        plt.title('文档-主题概率分布热力图')
        plt.xlabel('主题')
        plt.ylabel('文档')
        plt.savefig('visualizations/doc_topic_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("可视化结果已保存至 visualizations 文件夹")


# ------------------- 本地大模型API请求 (Ollama 1.x) -------------------
def analyze_with_deepseek(lda_model, api_url="http://localhost:11434/api/chat"):
    """通过Ollama 1.x API调用本地DeepSeek模型"""
    topic_analyses = []
    print(f"开始通过Ollama API分析主题...")
    print(f"(确保模型服务运行在 {api_url})")

    for i in range(lda_model.num_topics):
        top_words = lda_model.show_topic(i, topn=20)
        topic_words = [word for word, _ in top_words]

        prompt = f"""你是社交媒体分析专家，请分析以下主题关键词: {', '.join(topic_words)}
        1. 推断主题涉及的领域或事件
        2. 总结用户主要观点或情绪
        3. 给出5字以内的主题名称

        请按以下格式回答:
        领域/事件: 
        观点/情绪: 
        主题名称:"""

        try:
            # Ollama 1.x API请求格式
            payload = {
                "model": "deepseek-r1:1.5b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False  # 不使用流式输出
            }
            response = requests.post(api_url, json=payload, timeout=60)
            response.raise_for_status()

            # 解析JSON响应
            result = response.json()
            # Ollama 1.x 返回格式: {"message": {"content": "回复内容"}}
            analysis = result.get("message", {}).get("content", "")
            topic_analyses.append(analysis.strip())
            print(f"\n主题 {i} 分析结果:\n{analysis}")
        except Exception as e:
            print(f"分析主题 {i} 失败: {e}")
            topic_analyses.append(f"分析失败: {e}")

    return topic_analyses


# ------------------- 结果保存 -------------------
def save_results(lda_model, corpus, dictionary, texts, topic_analyses):
    """保存分析结果"""
    if not os.path.exists('results'):
        os.makedirs('results')

    # 保存主题-关键词
    with open('results/topic_keywords.txt', 'w', encoding='utf-8') as f:
        for i in range(lda_model.num_topics):
            f.write(f"主题 {i}:\n")
            top_words = lda_model.show_topic(i, topn=20)
            for word, prob in top_words:
                f.write(f"  {word}: {prob:.4f}\n")
            f.write("\n")
        print("主题-关键词已保存至 results/topic_keywords.txt")

    # 保存主题分析
    if topic_analyses:
        with open('results/topic_analyses.txt', 'w', encoding='utf-8') as f:
            for i, analysis in enumerate(topic_analyses):
                f.write(f"主题 {i} 分析结果:\n{analysis}\n\n")
        print("主题分析已保存至 results/topic_analyses.txt")

    # 保存文档-主题分布
    with open('results/doc_topic_distribution.txt', 'w', encoding='utf-8') as f:
        f.write("文档ID\t主要主题\t主题分布\n")
        for doc_id, doc in enumerate(corpus):
            topic_dist = lda_model.get_document_topics(doc, minimum_probability=0.1)
            if not topic_dist:
                topic_dist = lda_model.get_document_topics(doc, minimum_probability=0)
                topic_dist = sorted(topic_dist, key=lambda x: x[1], reverse=True)[:1]

            main_topic = topic_dist[0][0] if topic_dist else -1
            f.write(f"{doc_id}\t{main_topic}\t{topic_dist}\n")
        print("文档-主题分布已保存至 results/doc_topic_distribution.txt")


# ------------------- 主函数 -------------------
def main():
    """主函数"""
    file_path = 'newsdata.txt'  # 数据文件路径
    num_topics = 5  # 主题数量
    api_url = "http://localhost:11434/api/chat"  # Ollama 1.x API地址

    # 初始化
    print("=" * 50)
    print("Twitter主题分析程序启动 (使用本地Ollama模型)")
    print("=" * 50)

    # 字体设置
    setup_font()

    # 工具初始化
    stemmer = init_tools()

    # 读取数据
    texts = read_data(file_path)

    # 数据预处理
    processed_texts = preprocess_text(texts, stemmer)

    # 训练LDA模型
    try:
        lda_model, corpus, dictionary, non_empty_texts = train_lda(processed_texts, num_topics)
    except ValueError as e:
        print(f"模型训练失败: {e}")
        return

    # 打印主题-关键词
    print("\n主题-关键词分布:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"主题 {idx}:\n  {topic}")

    # 可视化分析
    visualize(lda_model, corpus, dictionary, non_empty_texts)

    # 本地大模型API分析
    topic_analyses = analyze_with_deepseek(lda_model, api_url)

    # 保存结果
    save_results(lda_model, corpus, dictionary, non_empty_texts, topic_analyses)

    print("\n=" * 50)
    print("分析完成！所有结果已保存")
    print("=" * 50)


if __name__ == "__main__":
    main()