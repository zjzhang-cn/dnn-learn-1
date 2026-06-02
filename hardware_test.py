import torch
import platform

def check_pytorch_hardware():
    """
    检测并显示当前PyTorch使用的硬件信息
    """
    print("="*50)
    print("PyTorch 硬件检测程序")
    print("="*50)
    
    # 显示PyTorch版本
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"Python 版本: {platform.python_version()}")
    print()
    
    # 检查CUDA是否可用
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
            print(f"    GPU 内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"    计算能力: {torch.cuda.get_device_capability(i)}")
    else:
        print("CUDA 不可用，使用CPU进行计算")
    print()
    
    # 检查MPS (Metal Performance Shaders) - 适用于Apple Silicon
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) 可用: True")
    else:
        print("MPS (Metal Performance Shaders) 可用: False")
    print()
    
    # 检查CPU相关信息
    print(f"CPU 数量: {torch.get_num_threads()}")
    print(f"CPU 可用核心数: {torch.get_num_interop_threads()}")
    print()
    
    # 显示当前默认设备
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"当前默认CUDA设备: {current_device} ({torch.cuda.get_device_name(current_device)})")
    else:
        print("当前默认设备: CPU")
    print()
    
    # 简单的硬件性能测试
    print("硬件性能测试:")
    try:
        # 创建两个大张量并执行矩阵乘法
        size = 1000
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        import time
        start_time = time.time()
        c = torch.mm(a, b)
        end_time = time.time()
        
        print(f"  {size}x{size} 矩阵乘法耗时: {end_time - start_time:.4f} 秒")
        print(f"  使用设备: {device}")
        
        # 清理GPU内存
        del a, b, c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"  性能测试出现错误: {e}")
    
    print()
    print("="*50)
    print("硬件检测完成")

if __name__ == "__main__":
    check_pytorch_hardware()