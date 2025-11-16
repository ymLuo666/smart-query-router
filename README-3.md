# Smart Query Router with LoRA-based SLM Selection

## 📋 项目简介

这是一个智能查询路由系统，能够根据用户输入的query自动选择最合适的领域专家小语言模型(SLM)来提供背景信息。系统使用embedding相似度匹配来实现智能路由，并在没有合适领域时自动fallback到Web搜索。

### 核心功能

1. **Query Embedding转换**: 将用户查询转换为向量表示
2. **多领域SLM支持**: 支持注册多个领域专家模型(使用LoRA部署)
3. **智能路由选择**: 基于embedding相似度自动选择最佳SLM
4. **背景信息生成**: 使用选定的SLM生成领域相关背景信息
5. **Web搜索Fallback**: 当query与所有领域都不匹配时，使用Web搜索获取信息

## 🏗️ 系统架构

```
用户Query
    ↓
[1] Query → Embedding 转换
    ↓
[2] 与4个LoRA-SLM的Domain Embedding计算相似度
    ↓
[3] 选择相似度最高的SLM
    ↓
    ├─→ [4a] 相似度 ≥ 阈值 → 使用选中的SLM生成背景信息
    └─→ [4b] 相似度 < 阈值 → 使用Web搜索获取信息
    ↓
[5] 组合增强的Query (Original Query + Background Info)
    ↓
发送到主LLM进行最终回答
```

## 📦 依赖项

```bash
# 核心依赖
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
sentence-transformers>=2.2.0
numpy>=1.24.0

# Web搜索支持
openai==0.27.6

# 可选依赖
accelerate>=0.20.0  # 用于模型加载优化
bitsandbytes>=0.39.0  # 用于量化支持
```

## 🚀 安装

```bash
# 1. 克隆或下载项目文件
# 2. 安装依赖
pip install torch transformers peft sentence-transformers numpy openai==0.27.6

# 3. (可选) 安装加速库
pip install accelerate bitsandbytes
```

## 📝 配置

### 1. 环境变量设置

```bash
# 设置Qianwen API密钥 (用于Web搜索)
export QIANWEN_API_KEY="your-api-key-here"
```

### 2. 修改config.py

```python
# 配置你的SLM模型
SLM_DOMAINS = {
    "medical": {
        "base_model_id": "your-base-model-id",  # 替换为你的模型ID
        "lora_weights_path": "./lora_weights/medical_lora",  # LoRA权重路径
        "domain_description": "Medical and healthcare domain..."
    },
    # ... 添加更多领域
}
```

## 💻 使用方法

### 基础使用

```python
from smart_query_router import SmartQueryRouter

# 1. 初始化路由器
router = SmartQueryRouter(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold=0.6
)

# 2. 注册领域专家SLM
router.register_slm(
    domain_name="medical",
    base_model_id="meta-llama/Llama-2-7b-hf",
    lora_weights_path="./lora_weights/medical_lora",
    domain_description="Medical and healthcare domain, including diseases, treatments..."
)

# 3. 处理查询
result = router.process_query("What are the symptoms of diabetes?")

# 4. 获取增强的query
enhanced_query = result['enhanced_query']

# 5. 将enhanced_query发送给主LLM
# main_llm_response = your_main_llm.generate(enhanced_query)

# 6. 清理资源
router.unload_all_slms()
```

### 完整示例

查看 `example_usage.py` 文件获取更多示例：
- 快速开始示例
- 单个查询处理
- 自定义阈值调整
- 批量查询处理

## 🔧 核心参数说明

### SmartQueryRouter 初始化参数

- `embedding_model_name`: Embedding模型名称 (默认: "sentence-transformers/all-MiniLM-L6-v2")
- `similarity_threshold`: 相似度阈值，范围[0, 1] (默认: 0.6)
  - 高于阈值: 使用匹配的SLM
  - 低于阈值: 使用Web搜索
- `device`: 计算设备 "cuda" 或 "cpu" (默认: 自动检测)

### register_slm 参数

- `domain_name`: 领域名称，唯一标识符
- `base_model_id`: HuggingFace模型ID
- `lora_weights_path`: LoRA权重文件路径
- `domain_description`: 领域描述，用于生成domain embedding

### process_query 返回结果

```python
{
    "original_query": str,           # 原始查询
    "selected_domain": str or None,  # 选中的领域名称
    "similarity_score": float,       # 最高相似度分数
    "all_similarities": dict,        # 所有领域的相似度
    "method_used": str,              # "domain_slm" 或 "web_search"
    "background_info": str,          # 生成的背景信息
    "enhanced_query": str            # 增强后的查询(可直接发送给主LLM)
}
```

## 📊 工作流程详解

### 步骤1: Query Embedding转换

```python
query = "What are the symptoms of diabetes?"
query_embedding = router.get_query_embedding(query)
# 输出: torch.Tensor of shape [embedding_dim]
```

### 步骤2-3: 计算相似度并选择SLM

```python
# 计算与每个domain的余弦相似度
similarities = {
    "medical": 0.85,
    "finance": 0.32,
    "legal": 0.28,
    "technology": 0.35
}

# 选择相似度最高的domain
selected_domain = "medical"  # 0.85 > threshold (0.6)
```

### 步骤4: 生成背景信息

如果选中了SLM:
```python
background_info = router.generate_background_info(
    domain_name="medical",
    query=query
)
# 输出: "Diabetes is a chronic metabolic disorder..."
```

如果使用Web搜索:
```python
background_info = router.web_search_fallback(query)
# 输出: Web搜索获取的背景信息
```

### 步骤5: 组合增强Query

```python
enhanced_query = f"""Original Query: {query}

Background Information:
{background_info}

Please answer the original query considering the background information provided above."""
```

## 🎯 最佳实践

### 1. Domain Description优化

好的domain description应该:
- 包含领域关键词和概念
- 描述清晰且具体
- 长度适中 (50-150词)

```python
# ✅ 好的描述
domain_description = "Medical and healthcare domain, including diseases, symptoms, treatments, medications, medical procedures, diagnostics, anatomy, physiology, patient care, and clinical practices"

# ❌ 不好的描述
domain_description = "Medical stuff"
```

### 2. 阈值调整

根据实际应用调整阈值:
- **严格匹配** (threshold=0.8): 只有非常相关的query才使用SLM
- **平衡** (threshold=0.6): 默认推荐
- **宽松匹配** (threshold=0.4): 更多query使用SLM

### 3. 内存管理

处理大量query时:
```python
# 批量处理前加载一次
for query in batch_queries:
    result = router.process_query(query)
    # 处理result...

# 处理完后统一卸载
router.unload_all_slms()
```

### 4. 错误处理

```python
try:
    result = router.process_query(query)
except Exception as e:
    print(f"Error processing query: {e}")
    # 使用fallback方案
    result = router.web_search_fallback(query)
```

## 🔍 调试和监控

### 启用详细日志

```python
router = SmartQueryRouter(
    embedding_model_name="...",
    similarity_threshold=0.6
)

# 系统会自动打印详细的处理步骤
```

### 检查相似度分数

```python
result = router.process_query(query)

print("Similarity scores:")
for domain, score in result['all_similarities'].items():
    print(f"  {domain}: {score:.4f}")
```

## ⚠️ 常见问题

### Q1: LoRA权重文件找不到

**A**: 确保LoRA权重路径正确，系统会自动fallback到base model

```python
# 检查路径是否存在
import os
if not os.path.exists(lora_path):
    print(f"Warning: LoRA weights not found at {lora_path}")
```

### Q2: CUDA内存不足

**A**: 考虑以下优化方案:
```python
# 1. 使用CPU
router = SmartQueryRouter(device="cpu")

# 2. 按需加载/卸载模型
router.unload_slm("medical")  # 卸载不需要的模型

# 3. 使用量化
# 在模型加载时添加量化配置
```

### Q3: Web搜索失败

**A**: 检查API配置:
```python
# 确保设置了正确的API key
export QIANWEN_API_KEY="your-key"

# 或在代码中设置
os.environ["QIANWEN_API_KEY"] = "your-key"
```

### Q4: 相似度总是很低

**A**: 优化domain description:
```python
# 使用更详细、更相关的描述
domain_description = "详细描述领域内的关键概念、术语、应用场景..."
```

## 🛠️ 高级用法

### 自定义Embedding模型

```python
# 使用多语言模型
router = SmartQueryRouter(
    embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 使用更高质量的模型
router = SmartQueryRouter(
    embedding_model_name="sentence-transformers/all-mpnet-base-v2"
)
```

### 动态添加/移除领域

```python
# 添加新领域
router.register_slm(
    domain_name="science",
    base_model_id="...",
    lora_weights_path="...",
    domain_description="..."
)

# 移除领域
if "science" in router.slm_configs:
    del router.slm_configs["science"]
    del router.slm_embeddings["science"]
```

### 自定义生成参数

修改 `smart_query_router.py` 中的生成配置:
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=500,      # 增加生成长度
    temperature=0.5,         # 降低随机性
    top_p=0.95,             # 调整采样
    repetition_penalty=1.2  # 避免重复
)
```

## 📈 性能优化

1. **使用GPU**: 显著提升处理速度
2. **批量处理**: 一次初始化，处理多个query
3. **模型缓存**: 避免重复加载模型
4. **Embedding缓存**: 对domain embedding进行缓存

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request!

## 📧 联系方式

如有问题，请通过以下方式联系：
- 提交GitHub Issue
- Email: [your-email]

---

**注意**: 本系统需要有效的HuggingFace模型和LoRA权重。请确保你有相应的访问权限和模型文件。
