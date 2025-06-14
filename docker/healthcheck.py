#!/usr/bin/env python3
"""
STRATO-PEFT Docker健康检查脚本

用于Docker容器的健康状态监控，检查关键组件和服务的可用性。
"""

import sys
import os
import time
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class HealthChecker:
    """Docker容器健康检查器"""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        
    def log(self, message: str, level: str = "INFO"):
        """记录日志消息"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def check_python_environment(self) -> bool:
        """检查Python环境和核心包"""
        self.log("Checking Python environment...")
        
        try:
            # 检查Python版本
            python_version = sys.version_info
            if python_version.major != 3 or python_version.minor < 8:
                self.log(f"Python version {python_version.major}.{python_version.minor} may not be compatible", "WARN")
                self.warnings.append("Python version compatibility")
            
            # 检查核心包
            required_packages = [
                'torch', 'transformers', 'peft', 'omegaconf', 
                'numpy', 'pandas', 'tqdm', 'rich'
            ]
            
            missing_packages = []
            for package in required_packages:
                if importlib.util.find_spec(package) is None:
                    missing_packages.append(package)
            
            if missing_packages:
                self.log(f"Missing packages: {', '.join(missing_packages)}", "ERROR")
                return False
                
            self.log("Python environment check passed")
            return True
            
        except Exception as e:
            self.log(f"Python environment check failed: {e}", "ERROR")
            return False
    
    def check_pytorch_gpu(self) -> bool:
        """检查PyTorch和GPU可用性"""
        self.log("Checking PyTorch and GPU availability...")
        
        try:
            import torch
            
            # 检查PyTorch版本
            torch_version = torch.__version__
            self.log(f"PyTorch version: {torch_version}")
            
            # 检查CUDA
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                self.log(f"CUDA available: {gpu_count} GPU(s)")
                self.log(f"Current device: {current_device} ({device_name})")
                
                # 简单的GPU测试
                try:
                    test_tensor = torch.randn(10, 10).cuda()
                    result = torch.matmul(test_tensor, test_tensor.T)
                    self.log("GPU computation test passed")
                except Exception as e:
                    self.log(f"GPU computation test failed: {e}", "WARN")
                    self.warnings.append("GPU computation issue")
            else:
                # 检查MPS (Apple Silicon)
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.log("MPS (Apple Silicon) available")
                else:
                    self.log("No GPU available, using CPU", "WARN")
                    self.warnings.append("No GPU acceleration")
            
            return True
            
        except Exception as e:
            self.log(f"PyTorch/GPU check failed: {e}", "ERROR")
            return False
    
    def check_file_system(self) -> bool:
        """检查文件系统和重要文件"""
        self.log("Checking file system...")
        
        try:
            # 检查工作目录
            work_dir = Path("/app")
            if not work_dir.exists():
                self.log("Working directory /app not found", "ERROR")
                return False
            
            # 检查关键文件
            critical_files = [
                "/app/main.py",
                "/app/src",
                "/app/configs"
            ]
            
            missing_files = []
            for file_path in critical_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                self.log(f"Missing critical files: {', '.join(missing_files)}", "ERROR")
                return False
            
            # 检查写入权限
            test_dirs = ["/app/results", "/app/logs", "/app/cache", "/app/tmp"]
            for test_dir in test_dirs:
                dir_path = Path(test_dir)
                if not dir_path.exists():
                    try:
                        dir_path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        self.log(f"Cannot create directory {test_dir}: {e}", "WARN")
                        self.warnings.append(f"Directory creation issue: {test_dir}")
                        continue
                
                # 测试写入权限
                test_file = dir_path / "health_check_test.tmp"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                except Exception as e:
                    self.log(f"Cannot write to {test_dir}: {e}", "WARN")
                    self.warnings.append(f"Write permission issue: {test_dir}")
            
            self.log("File system check passed")
            return True
            
        except Exception as e:
            self.log(f"File system check failed: {e}", "ERROR")
            return False
    
    def check_network_connectivity(self) -> bool:
        """检查网络连接（可选）"""
        self.log("Checking network connectivity...")
        
        try:
            # 尝试导入requests并测试连接
            import requests
            
            # 测试到Hugging Face的连接
            test_urls = [
                "https://huggingface.co",
                "https://pypi.org",
                "https://github.com"
            ]
            
            connection_failures = 0
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        self.log(f"Connection to {url}: OK")
                    else:
                        self.log(f"Connection to {url}: HTTP {response.status_code}", "WARN")
                        connection_failures += 1
                except Exception as e:
                    self.log(f"Connection to {url}: Failed ({e})", "WARN")
                    connection_failures += 1
            
            if connection_failures == len(test_urls):
                self.log("All network connectivity tests failed", "WARN")
                self.warnings.append("Network connectivity issues")
            
            return True
            
        except ImportError:
            self.log("requests module not available, skipping network check", "WARN")
            return True
        except Exception as e:
            self.log(f"Network check failed: {e}", "WARN")
            self.warnings.append("Network check error")
            return True  # 网络检查失败不应该导致健康检查整体失败
    
    def check_gpu_drivers(self) -> bool:
        """检查GPU驱动状态"""
        self.log("Checking GPU drivers...")
        
        try:
            platform = os.getenv('STRATO_PLATFORM', 'unknown')
            
            if platform == 'cuda':
                # 检查nvidia-smi
                try:
                    result = subprocess.run(['nvidia-smi'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        self.log("nvidia-smi check passed")
                        # 解析GPU信息
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if 'MiB' in line and 'GPU' in line:
                                self.log(f"GPU Memory: {line.strip()}")
                                break
                    else:
                        self.log("nvidia-smi failed", "WARN")
                        self.warnings.append("nvidia-smi execution failed")
                except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                    self.log(f"nvidia-smi not available: {e}", "WARN")
                    self.warnings.append("nvidia-smi not found")
                    
            elif platform == 'rocm':
                # 检查rocm-smi
                try:
                    result = subprocess.run(['rocm-smi'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        self.log("rocm-smi check passed")
                    else:
                        self.log("rocm-smi failed", "WARN")
                        self.warnings.append("rocm-smi execution failed")
                except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                    self.log(f"rocm-smi not available: {e}", "WARN")
                    self.warnings.append("rocm-smi not found")
            else:
                self.log(f"Platform: {platform} (no specific driver check)")
            
            return True
            
        except Exception as e:
            self.log(f"GPU driver check failed: {e}", "WARN")
            self.warnings.append("GPU driver check error")
            return True  # GPU驱动检查失败不应该导致健康检查整体失败
    
    def check_memory_usage(self) -> bool:
        """检查内存使用情况"""
        self.log("Checking memory usage...")
        
        try:
            import psutil
            
            # 系统内存
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            self.log(f"System memory usage: {memory_percent:.1f}%")
            self.log(f"Available memory: {memory_available_gb:.1f} GB")
            
            if memory_percent > 90:
                self.log("High memory usage detected", "WARN")
                self.warnings.append("High system memory usage")
            
            # GPU内存（如果可用）
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        self.log(f"GPU {i} memory - Allocated: {memory_allocated:.1f} GB, Reserved: {memory_reserved:.1f} GB")
            except Exception:
                pass
            
            return True
            
        except ImportError:
            self.log("psutil not available, skipping memory check", "WARN")
            return True
        except Exception as e:
            self.log(f"Memory check failed: {e}", "WARN")
            self.warnings.append("Memory check error")
            return True
    
    def run_all_checks(self) -> bool:
        """运行所有健康检查"""
        self.log("Starting STRATO-PEFT health check...")
        
        checks = [
            ("Python Environment", self.check_python_environment),
            ("PyTorch & GPU", self.check_pytorch_gpu),
            ("File System", self.check_file_system),
            ("GPU Drivers", self.check_gpu_drivers),
            ("Memory Usage", self.check_memory_usage),
            ("Network Connectivity", self.check_network_connectivity),
        ]
        
        for check_name, check_func in checks:
            try:
                if check_func():
                    self.checks_passed += 1
                    self.log(f"✓ {check_name}")
                else:
                    self.checks_failed += 1
                    self.log(f"✗ {check_name}")
            except Exception as e:
                self.checks_failed += 1
                self.log(f"✗ {check_name}: {e}")
        
        # 生成总结
        total_checks = self.checks_passed + self.checks_failed
        self.log(f"\nHealth Check Summary:")
        self.log(f"Passed: {self.checks_passed}/{total_checks}")
        self.log(f"Failed: {self.checks_failed}")
        self.log(f"Warnings: {len(self.warnings)}")
        
        if self.warnings:
            self.log("Warnings:")
            for warning in self.warnings:
                self.log(f"  - {warning}")
        
        # 判断整体健康状态
        if self.checks_failed == 0:
            self.log("✓ Container is healthy", "INFO")
            return True
        elif self.checks_failed <= 2:  # 允许少量非关键检查失败
            self.log("⚠ Container is functional with warnings", "WARN")
            return True
        else:
            self.log("✗ Container has serious issues", "ERROR")
            return False


def main():
    """主函数"""
    try:
        checker = HealthChecker()
        is_healthy = checker.run_all_checks()
        
        # 返回适当的退出码
        sys.exit(0 if is_healthy else 1)
        
    except Exception as e:
        print(f"Health check failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()