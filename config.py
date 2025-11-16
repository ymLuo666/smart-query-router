"""
配置文件示例 - Smart Query Router Configuration
"""

# ============================================================================
# 基础配置
# ============================================================================

# Embedding模型配置
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# 其他可选的embedding模型:
# - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 支持多语言
# - "sentence-transformers/all-mpnet-base-v2"  # 更高质量
# - "BAAI/bge-small-en-v1.5"  # 中文友好

# 相似度阈值 (0-1)
# 低于此阈值将使用Web搜索而不是SLM
SIMILARITY_THRESHOLD = 0.6

# 计算设备
DEVICE = "cuda"  # 或 "cpu"

# ============================================================================
# SLM领域配置
# ============================================================================

SLM_DOMAINS = {
    "medical": {
        "base_model_id": "meta-llama/Llama-2-7b-hf",
        "lora_weights_path": "./lora_weights/medical_lora",
        "domain_description": (
            "Medical and healthcare domain, including diseases, treatments, "
            "medications, medical procedures, anatomy, physiology, diagnostics, "
            "patient care, and clinical practices"
        )
    },
    
    "finance": {
        "base_model_id": "meta-llama/Llama-2-7b-hf",
        "lora_weights_path": "./lora_weights/finance_lora",
        "domain_description": (
            "Finance and economics domain, including stock market, investments, "
            "banking, financial analysis, cryptocurrency, trading, risk management, "
            "portfolio optimization, and economic theories"
        )
    },
    
    "legal": {
        "base_model_id": "meta-llama/Llama-2-7b-hf",
        "lora_weights_path": "./lora_weights/legal_lora",
        "domain_description": (
            "Legal domain, including laws, regulations, contracts, court cases, "
            "legal procedures, constitutional law, intellectual property, "
            "litigation, and legal rights"
        )
    },
    
    "technology": {
        "base_model_id": "meta-llama/Llama-2-7b-hf",
        "lora_weights_path": "./lora_weights/tech_lora",
        "domain_description": (
            "Technology domain, including programming, software development, "
            "artificial intelligence, machine learning, deep learning, computer science, "
            "algorithms, data structures, system design, and technical systems"
        )
    }
}

# ============================================================================
# Web搜索配置 (Qianwen API)
# ============================================================================

# Qianwen API配置
QIANWEN_API_KEY = "xxx"  # 从环境变量或这里读取
QIANWEN_MODEL = "qwen-plus"  # 可选: "qwen-plus", "qwen-turbo", "qwen-max"

# Web搜索参数
WEB_SEARCH_MAX_LENGTH = 300

# ============================================================================
# 生成配置
# ============================================================================

# 背景信息生成的最大长度
MAX_BACKGROUND_LENGTH = 300

# 生成参数
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "max_new_tokens": 300
}

# ============================================================================
# 日志配置
# ============================================================================

# 是否打印详细日志
VERBOSE = True

# 日志级别
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
