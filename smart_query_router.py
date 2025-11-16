"""
Smart Query Router with LoRA-based SLM Selection
实现基于embedding相似度的智能查询路由系统
"""

import os
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import openai


class SmartQueryRouter:
    """
    智能查询路由器：根据query的embedding选择最合适的领域专家SLM
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.6,
        device: str = None
    ):
        """
        初始化路由器
        
        Args:
            embedding_model_name: 用于生成embedding的模型名称
            similarity_threshold: 相似度阈值，低于此值将使用Web搜索
            device: 计算设备 (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载embedding模型
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model.to(self.device)
        
        # 存储SLM配置
        self.slm_configs = {}
        self.slm_embeddings = {}
        self.loaded_models = {}
        
        # 相似度阈值
        self.similarity_threshold = similarity_threshold
        
    def register_slm(
        self,
        domain_name: str,
        base_model_id: str,
        lora_weights_path: str,
        domain_description: str
    ):
        """
        注册一个领域专家SLM
        
        Args:
            domain_name: 领域名称 (e.g., "medical", "finance", "legal", "tech")
            base_model_id: 基础模型的HuggingFace ID
            lora_weights_path: LoRA权重路径
            domain_description: 领域描述，用于生成domain embedding
        """
        print(f"Registering SLM for domain: {domain_name}")
        
        self.slm_configs[domain_name] = {
            "base_model_id": base_model_id,
            "lora_weights_path": lora_weights_path,
            "domain_description": domain_description
        }
        
        # 生成domain embedding (基于领域描述)
        domain_embedding = self.embedding_model.encode(
            domain_description,
            convert_to_tensor=True,
            device=self.device
        )
        self.slm_embeddings[domain_name] = domain_embedding
        
        print(f"Domain '{domain_name}' registered successfully")
        
    def load_slm(self, domain_name: str):
        """
        按需加载SLM模型（包含LoRA权重）
        
        Args:
            domain_name: 要加载的领域名称
        """
        if domain_name in self.loaded_models:
            return self.loaded_models[domain_name]
        
        if domain_name not in self.slm_configs:
            raise ValueError(f"Domain '{domain_name}' not registered")
        
        config = self.slm_configs[domain_name]
        print(f"Loading SLM for domain: {domain_name}")
        
        try:
            # 加载base model
            base_model = AutoModelForCausalLM.from_pretrained(
                config["base_model_id"],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config["base_model_id"],
                trust_remote_code=True
            )
            
            # 加载LoRA权重
            if os.path.exists(config["lora_weights_path"]):
                model = PeftModel.from_pretrained(
                    base_model,
                    config["lora_weights_path"]
                )
                print(f"LoRA weights loaded from: {config['lora_weights_path']}")
            else:
                print(f"Warning: LoRA weights not found at {config['lora_weights_path']}, using base model only")
                model = base_model
            
            # 设置为评估模式
            model.eval()
            
            self.loaded_models[domain_name] = {
                "model": model,
                "tokenizer": tokenizer
            }
            
            print(f"SLM for '{domain_name}' loaded successfully")
            return self.loaded_models[domain_name]
            
        except Exception as e:
            print(f"Error loading SLM for '{domain_name}': {e}")
            raise
    
    def get_query_embedding(self, query: str) -> torch.Tensor:
        """
        将query转换为embedding
        
        Args:
            query: 用户输入的查询
            
        Returns:
            query的embedding tensor
        """
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            device=self.device
        )
        return query_embedding
    
    def calculate_similarity(
        self,
        query_embedding: torch.Tensor,
        domain_embedding: torch.Tensor
    ) -> float:
        """
        计算query embedding和domain embedding的余弦相似度
        
        Args:
            query_embedding: query的embedding
            domain_embedding: domain的embedding
            
        Returns:
            相似度分数 (0-1)
        """
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            domain_embedding.unsqueeze(0)
        ).item()
        
        return similarity
    
    def select_best_slm(self, query: str) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        选择最适合处理query的SLM
        
        Args:
            query: 用户输入的查询
            
        Returns:
            (选中的domain名称, 最高相似度分数, 所有domain的相似度分数字典)
        """
        # 步骤1: 获取query embedding
        print(f"\n[Step 1] Encoding query to embedding...")
        query_embedding = self.get_query_embedding(query)
        
        # 步骤2 & 3: 计算与所有domain的相似度
        print(f"[Step 2&3] Calculating similarity with all registered SLMs...")
        similarities = {}
        
        for domain_name, domain_embedding in self.slm_embeddings.items():
            similarity = self.calculate_similarity(query_embedding, domain_embedding)
            similarities[domain_name] = similarity
            print(f"  - {domain_name}: {similarity:.4f}")
        
        # 找出最高相似度
        if not similarities:
            return None, 0.0, {}
        
        best_domain = max(similarities, key=similarities.get)
        best_similarity = similarities[best_domain]
        
        # 判断是否超过阈值
        if best_similarity < self.similarity_threshold:
            print(f"\n[Selection] Best similarity ({best_similarity:.4f}) below threshold ({self.similarity_threshold})")
            print(f"[Selection] Will use Web Search instead")
            return None, best_similarity, similarities
        
        print(f"\n[Selection] Selected domain: {best_domain} (similarity: {best_similarity:.4f})")
        return best_domain, best_similarity, similarities
    
    def generate_background_info(
        self,
        domain_name: str,
        query: str,
        max_length: int = 300
    ) -> str:
        """
        使用选定的SLM生成背景信息
        
        Args:
            domain_name: 选中的领域名称
            query: 原始查询
            max_length: 生成的最大长度
            
        Returns:
            生成的背景信息
        """
        print(f"\n[Step 4] Generating background information using {domain_name} SLM...")
        
        # 加载模型
        slm = self.load_slm(domain_name)
        model = slm["model"]
        tokenizer = slm["tokenizer"]
        
        # 构造prompt
        prompt = f"""Based on the following query, provide relevant background information and context to help understand the topic better. Do NOT answer the question directly.

Query: {query}

Background Information:"""
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的背景信息（去掉prompt部分）
        background_info = generated_text[len(prompt):].strip()
        
        print(f"Background info generated ({len(background_info)} chars)")
        return background_info
    
    def web_search_fallback(
        self,
        query: str,
        model_name: str = "qwen-plus",
        max_length: int = 300
    ) -> str:
        """
        使用Web搜索获取背景信息（当没有匹配的SLM时）
        
        Args:
            query: 原始查询
            model_name: 使用的模型名称
            max_length: 最大长度
            
        Returns:
            搜索得到的背景信息
        """
        print(f"\n[Step 5] Using Web Search to retrieve information...")
        
        result = self.Qianwen_search_result(
            query=query,
            model_name=model_name,
            Web_search=True,
            max_length=max_length
        )
        
        return result
    
    @staticmethod
    def Qianwen_search_result(query, model_name, Web_search=True, entity_search=None, max_length=300):
        """
        Perform web search using the Qianwen API to get relevant information for a query.
        Optimized for OpenAI version 0.27.6.
        """
        if entity_search is None:
            entity_search = []
        
        try:
            # Set up for OpenAI version 0.27.6
            import openai
            
            # Configure API credentials
            api_key = os.getenv("QIANWEN_API_KEY", "xxx")
            openai.api_key = api_key
            openai.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            
            # Create messages for the API
            messages = [
                {'role': 'system', 'content': "You are a research assistant. ONLY provide relevant background information, concepts, and definitions to understand the topic. Do NOT solve the problem or give direct answers. Always respond in English."},
                {'role': 'user', 'content': f"Find ONLY background information in English about this topic (DO NOT solve it): {query}"}
            ]
            
            # Make the API call using the 0.27.6 format
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                max_tokens=max_length,
                # For 0.27.6, use custom parameters
                enable_search=Web_search,
                search_options={
                    "forced_search": True
                }
            )
            
            # Extract the assistant's response for 0.27.6 format
            if 'choices' in completion and len(completion['choices']) > 0:
                search_result = completion['choices'][0]['message']['content']
                return f"{search_result}"
            
            # Fallback if the expected response format is different
            return f"{str(completion)}"
                
        except Exception as e:
            print(f"Web search error: {e}")
            # Provide more detailed error handling
            import traceback
            traceback.print_exc()
            return ""  # Return empty string on error
    
    def process_query(
        self,
        query: str,
        max_background_length: int = 300
    ) -> Dict[str, any]:
        """
        完整的查询处理流程
        
        Args:
            query: 用户输入的查询
            max_background_length: 背景信息的最大长度
            
        Returns:
            包含处理结果的字典
        """
        print("="*80)
        print(f"Processing Query: {query}")
        print("="*80)
        
        # 选择最佳SLM
        selected_domain, similarity, all_similarities = self.select_best_slm(query)
        
        # 生成背景信息
        if selected_domain is not None:
            # 使用选中的SLM生成背景信息
            background_info = self.generate_background_info(
                selected_domain,
                query,
                max_background_length
            )
            method = "domain_slm"
        else:
            # 使用Web搜索
            background_info = self.web_search_fallback(
                query,
                max_length=max_background_length
            )
            method = "web_search"
        
        # 组合增强后的query
        enhanced_query = f"""Original Query: {query}

Background Information:
{background_info}

Please answer the original query considering the background information provided above."""
        
        # 返回结果
        result = {
            "original_query": query,
            "selected_domain": selected_domain,
            "similarity_score": similarity,
            "all_similarities": all_similarities,
            "method_used": method,
            "background_info": background_info,
            "enhanced_query": enhanced_query
        }
        
        print("\n" + "="*80)
        print("Processing Complete!")
        print(f"Method: {method}")
        if selected_domain:
            print(f"Domain: {selected_domain}")
        print("="*80 + "\n")
        
        return result
    
    def unload_slm(self, domain_name: str):
        """
        卸载指定的SLM以释放内存
        """
        if domain_name in self.loaded_models:
            del self.loaded_models[domain_name]
            torch.cuda.empty_cache()
            print(f"SLM for '{domain_name}' unloaded")
    
    def unload_all_slms(self):
        """
        卸载所有已加载的SLM
        """
        for domain_name in list(self.loaded_models.keys()):
            self.unload_slm(domain_name)
        print("All SLMs unloaded")


def main():
    """
    示例使用
    """
    # 初始化路由器
    router = SmartQueryRouter(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.6
    )
    
    # 注册4个领域专家SLM
    # 注意：您需要替换为实际的模型ID和LoRA权重路径
    
    # 1. 医疗领域
    router.register_slm(
        domain_name="medical",
        base_model_id = "Qwen/Qwen2.5-1.5B-Instruct",  # 替换为您的模型ID
        lora_weights_path = "Arthur-77/QWEN2.5-1.5B-medical-finetuned",  # 替换为您的LoRA权重路径
        domain_description="Medical and healthcare domain, including diseases, treatments, medications, medical procedures, anatomy, physiology, and clinical practices"
    )
    
    # 2. 金融领域
    router.register_slm(
        domain_name="finance",
        base_model_id = "WiroAI/WiroAI-Finance-Qwen-1.5B",
        lora_weights_path = "",
        domain_description="Finance and economics domain, including stock market, investments, banking, financial analysis, cryptocurrency, trading, and economic theories"
    )
    
    # 3. 法律领域
    router.register_slm(
        domain_name="math",
        base_model_id="Qwen/Qwen2.5-Math-1.5B-Instruct",
        lora_weights_path="",
        domain_description="Mathematics domain, including algebra, calculus, geometry, statistics, mathematical reasoning, problem solving, equations, and quantitative analysis"
    )
    
    # 4. 技术领域
    router.register_slm(
        domain_name="technology",
        base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        lora_weights_path = "",
        domain_description="Technology domain, including programming, software development, artificial intelligence, machine learning, computer science, algorithms, and technical systems"
    )
    
    # 测试不同类型的查询
    test_queries = [
        "What are the symptoms and treatment options for type 2 diabetes?",
        "How does the Federal Reserve's interest rate decision affect the stock market?",
        "What are the legal implications of breaking a non-compete agreement?",
        "Explain the difference between supervised and unsupervised learning in machine learning",
        "What is the best way to cook pasta?"  # 不属于任何领域，应该使用Web搜索
    ]
    
    for query in test_queries:
        result = router.process_query(query)
        
        # 打印结果摘要
        print(f"\nQuery: {query}")
        print(f"Selected Domain: {result['selected_domain']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Method Used: {result['method_used']}")
        print(f"Background Info Preview: {result['background_info'][:200]}...")
        print("\n" + "-"*80 + "\n")
    
    # 清理
    router.unload_all_slms()


if __name__ == "__main__":
    main()
