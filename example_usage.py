"""
简化使用示例 - Quick Start Example
展示如何快速使用Smart Query Router系统
"""

from smart_query_router import SmartQueryRouter
import config

def quick_start_example():
    """
    快速开始示例
    """
    print("=" * 80)
    print("Smart Query Router - Quick Start Example")
    print("=" * 80)
    
    # 步骤1: 初始化路由器
    print("\n[1] Initializing router...")
    router = SmartQueryRouter(
        embedding_model_name=config.EMBEDDING_MODEL,
        similarity_threshold=config.SIMILARITY_THRESHOLD,
        device=config.DEVICE
    )
    
    # 步骤2: 注册领域专家SLM
    print("\n[2] Registering domain expert SLMs...")
    for domain_name, domain_config in config.SLM_DOMAINS.items():
        router.register_slm(
            domain_name=domain_name,
            base_model_id=domain_config["base_model_id"],
            lora_weights_path=domain_config["lora_weights_path"],
            domain_description=domain_config["domain_description"]
        )
    
    # 步骤3: 处理查询
    print("\n[3] Processing queries...\n")
    
    # 示例查询
    queries = [
        "What are the symptoms of COVID-19?",  # 医疗领域
        "How to invest in the stock market?",  # 金融领域
        "What is a patent?",  # 法律领域
        "Explain neural networks",  # 技术领域
        "Best recipe for chocolate cake"  # 不属于任何领域 -> Web搜索
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print('=' * 80)
        
        # 处理查询
        result = router.process_query(
            query=query,
            max_background_length=config.MAX_BACKGROUND_LENGTH
        )
        
        # 显示结果
        print(f"\n[Result Summary]")
        print(f"Selected Domain: {result['selected_domain'] or 'None (Web Search)'}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Method Used: {result['method_used']}")
        
        print(f"\n[Similarity Scores for All Domains]")
        for domain, score in result['all_similarities'].items():
            print(f"  {domain}: {score:.4f}")
        
        print(f"\n[Background Information Preview]")
        preview_length = 200
        background = result['background_info']
        if len(background) > preview_length:
            print(f"{background[:preview_length]}...")
        else:
            print(background)
        
        print(f"\n[Enhanced Query Available]")
        print(f"Length: {len(result['enhanced_query'])} characters")
        print("(This enhanced query can now be sent to your main LLM)")
        
        # 可以将enhanced_query发送给主LLM进行最终回答
        # main_llm_response = your_main_llm.generate(result['enhanced_query'])
        
    # 步骤4: 清理资源
    print("\n[4] Cleaning up...")
    router.unload_all_slms()
    print("Done!")


def single_query_example():
    """
    单个查询的简化示例
    """
    # 初始化
    router = SmartQueryRouter(
        similarity_threshold=0.6
    )
    
    # 只注册一个医疗领域的SLM作为示例
    router.register_slm(
        domain_name="medical",
        base_model_id="meta-llama/Llama-2-7b-hf",
        lora_weights_path="./lora_weights/medical_lora",
        domain_description="Medical and healthcare domain"
    )
    
    # 处理查询
    result = router.process_query("What is diabetes?")
    
    # 使用结果
    enhanced_query = result['enhanced_query']
    print(f"\nEnhanced Query ready for main LLM:")
    print(enhanced_query)
    
    # 清理
    router.unload_all_slms()


def custom_threshold_example():
    """
    自定义阈值示例 - 展示如何调整相似度阈值
    """
    query = "Tell me about machine learning"
    
    # 测试不同的阈值
    thresholds = [0.4, 0.6, 0.8]
    
    for threshold in thresholds:
        print(f"\n{'=' * 80}")
        print(f"Testing with threshold: {threshold}")
        print('=' * 80)
        
        router = SmartQueryRouter(similarity_threshold=threshold)
        
        # 注册技术领域
        router.register_slm(
            domain_name="technology",
            base_model_id="meta-llama/Llama-2-7b-hf",
            lora_weights_path="./lora_weights/tech_lora",
            domain_description="Technology and computer science domain"
        )
        
        result = router.process_query(query)
        
        print(f"Similarity: {result['similarity_score']:.4f}")
        print(f"Method: {result['method_used']}")
        print(f"Domain: {result['selected_domain'] or 'Web Search'}")
        
        router.unload_all_slms()


def batch_processing_example():
    """
    批处理示例 - 展示如何高效处理多个查询
    """
    # 初始化一次
    router = SmartQueryRouter()
    
    # 注册所有领域
    for domain_name, domain_config in config.SLM_DOMAINS.items():
        router.register_slm(
            domain_name=domain_name,
            base_model_id=domain_config["base_model_id"],
            lora_weights_path=domain_config["lora_weights_path"],
            domain_description=domain_config["domain_description"]
        )
    
    # 批量查询
    queries = [
        "What is hypertension?",
        "How to calculate ROI?",
        "What is a trademark?",
        "Explain gradient descent"
    ]
    
    results = []
    for query in queries:
        result = router.process_query(query)
        results.append(result)
    
    # 统计结果
    print("\n[Batch Processing Summary]")
    domain_counts = {}
    for result in results:
        method = result['method_used']
        domain_counts[method] = domain_counts.get(method, 0) + 1
    
    print(f"Total queries processed: {len(results)}")
    for method, count in domain_counts.items():
        print(f"  {method}: {count}")
    
    # 清理
    router.unload_all_slms()


if __name__ == "__main__":
    # 运行快速开始示例
    quick_start_example()
    
    # 或者运行其他示例:
    # single_query_example()
    # custom_threshold_example()
    # batch_processing_example()
