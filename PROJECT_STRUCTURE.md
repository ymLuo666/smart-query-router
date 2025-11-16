# 项目文件结构说明

## 📁 文件清单

```
smart-query-router/
│
├── smart_query_router.py      # 核心实现文件 - 主要的路由器类
├── config.py                   # 配置文件 - 所有可配置参数
├── example_usage.py            # 使用示例 - 展示如何使用系统
├── test_router.py              # 测试套件 - 验证系统功能
├── diagnostic.py               # 诊断工具 - 检查环境配置
├── requirements.txt            # 依赖列表 - Python包依赖
└── README.md                   # 详细文档 - 完整使用指南
```

## 📄 各文件说明

### 1. smart_query_router.py (核心文件) ⭐
**用途**: 系统的核心实现
**包含**:
- `SmartQueryRouter` 类: 主要的路由器实现
- Query embedding转换
- SLM注册和加载
- 相似度计算
- 领域选择逻辑
- 背景信息生成
- Web搜索fallback

**何时修改**:
- 需要调整核心算法时
- 添加新功能时
- 优化性能时

---

### 2. config.py (配置文件)
**用途**: 集中管理所有配置参数
**包含**:
- Embedding模型配置
- 相似度阈值设置
- SLM领域配置(模型ID、LoRA路径、领域描述)
- Web搜索配置(API密钥、模型选择)
- 生成参数配置

**何时修改**:
- 更换embedding模型时
- 添加/删除领域时
- 调整阈值时
- 配置API密钥时

---

### 3. example_usage.py (使用示例)
**用途**: 展示如何使用系统
**包含**:
- `quick_start_example()`: 快速开始示例
- `single_query_example()`: 单个查询处理
- `custom_threshold_example()`: 自定义阈值调整
- `batch_processing_example()`: 批量查询处理

**如何使用**:
```bash
python example_usage.py
```

---

### 4. test_router.py (测试套件)
**用途**: 验证系统各组件是否正常工作
**包含的测试**:
1. Embedding模型测试
2. 领域注册测试
3. 相似度计算测试
4. 领域选择测试
5. 阈值行为测试
6. Web搜索结构测试
7. 完整工作流测试

**如何使用**:
```bash
python test_router.py
```

---

### 5. diagnostic.py (诊断工具)
**用途**: 检查环境配置和依赖项
**检查项目**:
1. Python版本
2. 依赖包安装情况
3. CUDA可用性
4. Embedding模型加载
5. 环境变量设置
6. 模型路径有效性
7. 磁盘空间
8. 基本功能测试

**如何使用**:
```bash
python diagnostic.py
```

---

### 6. requirements.txt (依赖列表)
**用途**: 列出所有Python包依赖
**如何使用**:
```bash
pip install -r requirements.txt
```

---

### 7. README.md (完整文档)
**用途**: 项目的详细文档
**包含**:
- 项目简介
- 系统架构
- 安装指南
- 配置说明
- 使用方法
- 参数说明
- 最佳实践
- 常见问题
- 高级用法

---

## 🚀 快速开始流程

### 第一步: 安装依赖
```bash
pip install -r requirements.txt
```

### 第二步: 运行诊断
```bash
python diagnostic.py
```
确保所有检查项都通过。

### 第三步: 配置系统
编辑 `config.py`:
1. 设置你的模型ID
2. 配置LoRA权重路径
3. 设置API密钥

### 第四步: 运行测试
```bash
python test_router.py
```
验证系统功能正常。

### 第五步: 运行示例
```bash
python example_usage.py
```
查看系统如何工作。

### 第六步: 集成到你的项目
参考 `example_usage.py` 中的代码，将系统集成到你的项目中。

---

## 🔧 典型使用流程

### 场景1: 基础使用
```python
from smart_query_router import SmartQueryRouter
import config

# 初始化
router = SmartQueryRouter()

# 注册领域
for domain_name, domain_config in config.SLM_DOMAINS.items():
    router.register_slm(
        domain_name=domain_name,
        base_model_id=domain_config["base_model_id"],
        lora_weights_path=domain_config["lora_weights_path"],
        domain_description=domain_config["domain_description"]
    )

# 处理查询
result = router.process_query("Your query here")

# 使用增强的query
enhanced_query = result['enhanced_query']
# 发送给主LLM...
```

### 场景2: 自定义配置
```python
# 使用自定义参数
router = SmartQueryRouter(
    embedding_model_name="custom-model",
    similarity_threshold=0.7,
    device="cuda"
)

# 动态注册领域
router.register_slm(
    domain_name="custom_domain",
    base_model_id="your-model-id",
    lora_weights_path="./your_lora",
    domain_description="Your domain description"
)
```

### 场景3: 批量处理
```python
# 初始化一次
router = SmartQueryRouter()
# 注册领域...

# 批量处理
queries = ["query1", "query2", "query3"]
results = []

for query in queries:
    result = router.process_query(query)
    results.append(result)

# 清理
router.unload_all_slms()
```

---

## 📊 文件依赖关系

```
requirements.txt
      ↓ (安装依赖)
diagnostic.py
      ↓ (验证环境)
config.py
      ↓ (配置参数)
smart_query_router.py (核心实现)
      ↓ (被调用)
example_usage.py / test_router.py
      ↓ (参考)
你的项目
```

---

## 🎯 文件优先级

### 必须文件 (⭐⭐⭐)
1. `smart_query_router.py` - 核心功能
2. `requirements.txt` - 依赖安装
3. `config.py` - 基础配置

### 推荐文件 (⭐⭐)
4. `diagnostic.py` - 环境检查
5. `example_usage.py` - 使用参考
6. `README.md` - 详细文档

### 可选文件 (⭐)
7. `test_router.py` - 功能测试

---

## 💡 修改建议

### 如果你想...

**添加新的领域**:
→ 修改 `config.py` 中的 `SLM_DOMAINS`

**更换embedding模型**:
→ 修改 `config.py` 中的 `EMBEDDING_MODEL`

**调整相似度阈值**:
→ 修改 `config.py` 中的 `SIMILARITY_THRESHOLD`

**修改生成参数**:
→ 修改 `config.py` 中的 `GENERATION_CONFIG`

**添加新功能**:
→ 修改 `smart_query_router.py`

**自定义Web搜索**:
→ 修改 `smart_query_router.py` 中的 `web_search_fallback` 方法

---

## 📝 注意事项

1. **config.py 是关键**: 大部分配置都在这里，先配置好再运行
2. **先运行诊断**: 使用 `diagnostic.py` 确保环境正确
3. **查看示例**: `example_usage.py` 提供了完整的使用范例
4. **阅读文档**: `README.md` 包含详细的说明和最佳实践
5. **测试验证**: 使用 `test_router.py` 验证系统功能

---

## 🆘 需要帮助?

1. 运行 `python diagnostic.py` 诊断问题
2. 查看 `README.md` 的常见问题部分
3. 查看 `example_usage.py` 了解正确用法
4. 检查 `config.py` 确保配置正确
