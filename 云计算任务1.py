import os
import requests
import json
import re
import time
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


class NewsAnalyzer:
    """新闻分析器，用于处理新闻数据的真伪判别和情感分析"""

    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get("model_name", "deepseek-r1:1.5b")
        self.api_url = config.get("api_url", "http://localhost:11434/api/chat")
        self.timeout = config.get("timeout", 90)
        self.retries = config.get("retries", 5)
        self.news_file = config.get("news_file", "news.txt")
        self.labels_file = config.get("labels_file", "label.txt")
        self.output_file = config.get("output_file", "news_analysis_results.json")
        self.log_file = config.get("log_file", "model_inference.log")

        # 初始化日志文件
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=== 新闻分析日志 ===\n")
            f.write(f"使用模型: {self.model_name}\n")
            f.write(f"新闻文件: {self.news_file}\n")  # 对应日志输出
            f.write(f"标签文件: {self.labels_file}\n")
            f.write(f"API地址: {self.api_url}\n\n")

    def log(self, message: str, level: str = "INFO"):
        """带时间戳的日志记录"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

    def _call_api(self, payload: Dict) -> Optional[Dict]:
        """调用Ollama API的底层方法"""
        for attempt in range(self.retries):
            try:
                start_time = time.time()
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout
                )
                duration = time.time() - start_time

                if response.status_code == 200:
                    self.log(f"API调用成功 (耗时: {duration:.2f}s, 尝试: {attempt + 1})", "DEBUG")
                    return response.json()

                self.log(f"API错误 (状态码 {response.status_code}): {response.text}", "ERROR")

            except requests.Timeout:
                self.log(f"API请求超时 (尝试 {attempt + 1}/{self.retries})", "WARNING")
            except Exception as e:
                self.log(f"API异常: {type(e).__name__} - {str(e)}", "ERROR")

            time.sleep(3)

        return None

    def call_model(self, prompt: str) -> str:
        """调用大模型获取响应"""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }

        response = self._call_api(payload)

        if response:
            content = response.get("message", {}).get("content", "")
            self.log(f"模型响应内容 (长度: {len(content)}字符)", "DEBUG")
            return content

        self.log("模型调用失败，返回空内容", "ERROR")
        return ""

    def read_data_file(self, file_path: str) -> List[str]:
        """读取数据文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            self.log(f"读取文件失败: {file_path} - {str(e)}", "CRITICAL")
            raise

    def read_news(self) -> List[str]:
        """读取新闻数据"""
        self.log(f"读取新闻文件: {self.news_file}")  # 读取news.txt
        news_list = self.read_data_file(self.news_file)

        if not news_list:
            self.log("新闻文件为空", "ERROR")
            raise ValueError("新闻文件内容为空")

        self.log(f"成功读取 {len(news_list)} 条新闻")
        return news_list

    def read_labels(self) -> List[int]:
        """读取标签数据"""
        self.log(f"读取标签文件: {self.labels_file}")
        labels = [int(label) for label in self.read_data_file(self.labels_file) if label.isdigit()]

        if not labels:
            self.log("标签文件无效", "ERROR")
            raise ValueError("标签文件内容无效")

        self.log(f"成功读取 {len(labels)} 个标签")
        return labels

    def analyze_news(self, news_list: List[str], task: str) -> List:
        """分析新闻数据"""
        results = []
        task_name = "真假判别" if task == "veracity" else "情感分析"

        self.log(f"开始{task_name}任务")

        # 为小模型优化提示词
        if task == "veracity":
            base_prompt = ("你是一个新闻真伪判别专家。请仔细阅读以下新闻内容，"
                           "根据事实逻辑判断其真假。如果是假新闻输出0，真新闻输出1。"
                           "不要解释，直接输出数字。\n新闻：")
        else:
            base_prompt = ("你是一个情感分析专家。请分析以下新闻的情感倾向。"
                           "如果是正面情感输出'积极'，负面情感输出'消极'，中性情感输出'中性'。"
                           "不要解释，直接输出结果。\n新闻：")

        for i, news in enumerate(tqdm(news_list, desc=task_name)):
            # 构建提示词，添加序号便于追踪
            prompt = f"#{i + 1}: {base_prompt}{news}"

            # 为小模型添加更明确的指令
            if task == "veracity":
                prompt += "\n\n指令：只输出数字0或1，不要其他内容！"
            else:
                prompt += "\n\n指令：只输出'积极'、'消极'或'中性'，不要其他内容！"

            response = self.call_model(prompt)

            if task == "veracity":
                result = self._parse_veracity(response, i + 1)
            else:
                result = self._parse_sentiment(response, i + 1)

            results.append(result)

            # 每处理5条新闻记录一次进度
            if (i + 1) % 5 == 0:
                self.log(f"已处理 {i + 1}/{len(news_list)} 条新闻")

        return results

    def _parse_veracity(self, response: str, index: int) -> int:
        """解析真伪判别结果"""
        # 更宽松的匹配模式
        match = re.search(r'\b(0|1|假|false|伪)\b', response, re.IGNORECASE)
        if match:
            # 处理中文数字
            if match.group(1) in ["0", "假", "false", "伪"]:
                result = 0
            else:
                result = 1
            self.log(f"新闻 #{index} 预测结果: {result}", "DEBUG")
            return result

        # 尝试从响应中提取数字
        digits = re.findall(r'\d', response)
        if digits:
            result = int(digits[0])
            self.log(f"新闻 #{index} 从响应中提取到数字: {result}", "DEBUG")
            return result

        self.log(f"新闻 #{index} 无法解析预测结果: {response[:100]}...", "WARNING")
        return 0  # 默认返回假新闻

    def _parse_sentiment(self, response: str, index: int) -> str:
        """解析情感分析结果"""
        # 更宽松的匹配模式
        if re.search(r'积极|正面|好|支持|positive|optimistic', response, re.IGNORECASE):
            sentiment = "积极"
        elif re.search(r'消极|负面|坏|反对|negative|pessimistic', response, re.IGNORECASE):
            sentiment = "消极"
        else:
            sentiment = "中性"

        self.log(f"新闻 #{index} 情感分析: {sentiment}", "DEBUG")
        return sentiment

    def calculate_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """计算评估指标"""
        if len(predictions) != len(labels):
            self.log("预测结果与标签数量不匹配", "ERROR")
            raise ValueError("预测结果与标签数量不匹配")

        total = len(labels)
        correct = sum(p == t for p, t in zip(predictions, labels))

        # 计算假新闻指标
        fake_indices = [i for i, t in enumerate(labels) if t == 0]
        fake_count = len(fake_indices)
        correct_fake = sum(1 for i in fake_indices if predictions[i] == 0)

        # 计算真新闻指标
        true_indices = [i for i, t in enumerate(labels) if t == 1]
        true_count = len(true_indices)
        correct_true = sum(1 for i in true_indices if predictions[i] == 1)

        return {
            "Accuracy": correct / total if total > 0 else 0,
            "Accuracy_fake": correct_fake / fake_count if fake_count > 0 else 0,
            "Accuracy_true": correct_true / true_count if true_count > 0 else 0,
            "Total_news": total,
            "Correct_predictions": correct,
            "Fake_news_count": fake_count,
            "True_news_count": true_count
        }

    def save_results(self, news: List[str], predictions: List[int],
                     sentiments: List[str], metrics: Dict[str, float]):
        """保存分析结果"""
        results = {
            "model": self.model_name,
            "news": news,
            "veracity_predictions": predictions,
            "sentiment_analysis": sentiments,
            "evaluation_metrics": metrics,
            "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        self.log(f"结果已保存至: {self.output_file}")

    def check_api_connection(self) -> bool:
        """检查API连接是否可用"""
        try:
            self.log("检查API连接和模型可用性...")
            response = requests.get(f"{self.api_url}/../tags", timeout=15)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                if self.model_name in models:
                    self.log(f"API连接正常，模型可用: {self.model_name}")
                    return True

                self.log(f"模型 '{self.model_name}' 未找到。可用模型: {', '.join(models)}", "ERROR")
                return False

            self.log(f"API连接失败 (状态码 {response.status_code})", "ERROR")
            return False

        except Exception as e:
            self.log(f"API连接检查异常: {str(e)}", "ERROR")
            return False


def main():
    # 配置参数 - 适配1.5b模型和新的文件名
    config = {
        "model_name": "deepseek-r1:1.5b",
        "api_url": "http://localhost:11434/api/chat",
        "timeout": 90,
        "retries": 5,
        "news_file": "news.txt",  # 改为news.txt
        "labels_file": "label.txt",
        "output_file": "analysis_results.json",
        "log_file": "news_analysis.log"
    }

    analyzer = NewsAnalyzer(config)

    print("\n" + "=" * 50)
    print(f"新闻分析系统 - 使用模型: {config['model_name']}")
    print(f"新闻文件: {config['news_file']}")  # 对应输出news.txt
    print(f"标签文件: {config['labels_file']}")
    print("=" * 50)

    # 检查API连接
    if not analyzer.check_api_connection():
        print("无法连接到模型API，请检查服务是否运行")
        return

    try:
        # 读取数据
        news_list = analyzer.read_news()
        labels = analyzer.read_labels()

        # 验证数据量匹配
        if len(news_list) != len(labels):
            analyzer.log(f"新闻数量({len(news_list)})与标签数量({len(labels)})不匹配", "ERROR")
            raise ValueError("数据不匹配")

        # 任务1: 真伪判别
        print("\n=== 执行任务1: 新闻真伪判别 ===")
        predictions = analyzer.analyze_news(news_list, "veracity")

        # 计算指标
        metrics = analyzer.calculate_metrics(predictions, labels)
        print("\n=== 评估指标 ===")
        print(f"整体准确率: {metrics['Accuracy']:.4f} ({metrics['Correct_predictions']}/{metrics['Total_news']})")
        print(f"假新闻识别准确率: {metrics['Accuracy_fake']:.4f} ({metrics['Fake_news_count']}条假新闻)")
        print(f"真新闻识别准确率: {metrics['Accuracy_true']:.4f} ({metrics['True_news_count']}条真新闻)")

        # 任务2: 情感分析
        print("\n=== 执行任务2: 情感分析 ===")
        sentiments = analyzer.analyze_news(news_list, "sentiment")

        # 保存结果
        analyzer.save_results(news_list, predictions, sentiments, metrics)

        print("\n分析完成!")
        print(f"详细日志见: {config['log_file']}")
        print(f"完整结果见: {config['output_file']}")

    except Exception as e:
        analyzer.log(f"程序异常终止: {str(e)}", "CRITICAL")
        print(f"程序出错: {str(e)}")


if __name__ == "__main__":
    main()