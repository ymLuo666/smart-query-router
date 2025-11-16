"""
诊断脚本 - Diagnostic Script
用于检查环境配置和依赖项是否正确安装
"""

import sys
import os


def check_python_version():
    """检查Python版本"""
    print("\n" + "="*80)
    print("1. Checking Python Version")
    print("="*80)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible (>=3.8)")
        return True
    else:
        print("❌ Python version should be 3.8 or higher")
        return False


def check_dependencies():
    """检查必需的依赖包"""
    print("\n" + "="*80)
    print("2. Checking Dependencies")
    print("="*80)
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'sentence_transformers': 'Sentence Transformers',
        'numpy': 'NumPy',
        'openai': 'OpenAI'
    }
    
    all_installed = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} is installed")
        except ImportError:
            print(f"❌ {name} is NOT installed")
            all_installed = False
    
    if not all_installed:
        print("\n⚠ Some dependencies are missing. Install them with:")
        print("pip install -r requirements.txt")
    
    return all_installed


def check_cuda_availability():
    """检查CUDA是否可用"""
    print("\n" + "="*80)
    print("3. Checking CUDA Availability")
    print("="*80)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✅ CUDA is available")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            
            return True
        else:
            print("⚠ CUDA is NOT available (will use CPU)")
            print("   This is fine but inference will be slower")
            return True  # Not a failure, just a warning
            
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False


def check_embedding_model():
    """检查Embedding模型是否可以加载"""
    print("\n" + "="*80)
    print("4. Checking Embedding Model")
    print("="*80)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("Attempting to load embedding model (this may take a moment)...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # 测试编码
        test_text = "This is a test sentence"
        embedding = model.encode(test_text)
        
        print(f"✅ Embedding model loaded successfully")
        print(f"   Embedding dimension: {len(embedding)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        return False


def check_environment_variables():
    """检查环境变量"""
    print("\n" + "="*80)
    print("5. Checking Environment Variables")
    print("="*80)
    
    qianwen_key = os.getenv("QIANWEN_API_KEY")
    
    if qianwen_key:
        # 隐藏大部分key内容
        masked_key = qianwen_key[:10] + "..." + qianwen_key[-4:] if len(qianwen_key) > 14 else "***"
        print(f"✅ QIANWEN_API_KEY is set: {masked_key}")
    else:
        print("⚠ QIANWEN_API_KEY is not set")
        print("   Web search functionality may not work")
        print("   Set it with: export QIANWEN_API_KEY='your-key-here'")
    
    return True  # Not critical


def check_model_paths():
    """检查配置文件中的模型路径"""
    print("\n" + "="*80)
    print("6. Checking Model Paths")
    print("="*80)
    
    try:
        import config
        
        print("Checking LoRA weight paths from config...")
        
        all_exist = True
        for domain_name, domain_config in config.SLM_DOMAINS.items():
            lora_path = domain_config['lora_weights_path']
            
            if os.path.exists(lora_path):
                print(f"✅ {domain_name}: {lora_path}")
            else:
                print(f"⚠ {domain_name}: {lora_path} (not found)")
                all_exist = False
        
        if not all_exist:
            print("\n⚠ Some LoRA weight paths don't exist")
            print("   The system will fallback to base models")
            print("   Update paths in config.py to point to your LoRA weights")
        
        return True
        
    except ImportError:
        print("⚠ config.py not found or cannot be imported")
        return True
    except Exception as e:
        print(f"⚠ Error checking model paths: {e}")
        return True


def check_disk_space():
    """检查磁盘空间"""
    print("\n" + "="*80)
    print("7. Checking Disk Space")
    print("="*80)
    
    try:
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        
        total_gb = total // (2**30)
        used_gb = used // (2**30)
        free_gb = free // (2**30)
        
        print(f"Total: {total_gb} GB")
        print(f"Used: {used_gb} GB")
        print(f"Free: {free_gb} GB")
        
        if free_gb < 10:
            print("⚠ Low disk space! Models may fail to download")
            return False
        else:
            print("✅ Sufficient disk space available")
            return True
            
    except Exception as e:
        print(f"⚠ Could not check disk space: {e}")
        return True


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "="*80)
    print("8. Testing Basic Functionality")
    print("="*80)
    
    try:
        from smart_query_router import SmartQueryRouter
        
        print("Creating SmartQueryRouter instance...")
        router = SmartQueryRouter()
        
        print("✅ SmartQueryRouter initialized successfully")
        
        print("\nRegistering a test domain...")
        router.register_slm(
            domain_name="test",
            base_model_id="test-model",
            lora_weights_path="./test",
            domain_description="Test domain"
        )
        
        print("✅ Domain registration works")
        
        print("\nTesting query embedding...")
        query = "What is machine learning?"
        embedding = router.get_query_embedding(query)
        
        print(f"✅ Query embedding works (shape: {embedding.shape})")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_diagnostics():
    """运行所有诊断检查"""
    print("\n" + "="*80)
    print("SMART QUERY ROUTER - DIAGNOSTIC TOOL")
    print("="*80)
    print("This tool will check if your environment is properly configured.")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA Availability", check_cuda_availability),
        ("Embedding Model", check_embedding_model),
        ("Environment Variables", check_environment_variables),
        ("Model Paths", check_model_paths),
        ("Disk Space", check_disk_space),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ {check_name} check crashed: {e}")
            results.append((check_name, False))
    
    # 打印总结
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print("\n" + "-"*80)
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("1. Configure your SLM models in config.py")
        print("2. Run: python example_usage.py")
    else:
        print(f"\n⚠ {total - passed} check(s) failed.")
        print("\nPlease address the issues above before proceeding.")
        print("\nCommon solutions:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Set environment variables: export QIANWEN_API_KEY='your-key'")
        print("- Update model paths in config.py")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    run_diagnostics()
