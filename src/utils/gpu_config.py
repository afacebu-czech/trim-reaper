"""
GPU Configuration for Reels Generator
Provides GPU detection and configuration utilities
"""

import os
from typing import Dict, Any, Optional, Tuple
from loguru import logger

class GPUConfig:
    """GPU configuration and detection"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if GPUConfig._initialized:
            return
        
        self.cuda_available = False
        self.gpu_name = None
        self.gpu_memory = 0
        self.cuda_version = None
        self.pytorch_version = None
        self.device = "cpu"
        
        self._detect_gpu()
        GPUConfig._initialized = True
    
    def _detect_gpu(self):
        """Detect GPU and CUDA availability"""
        
        # Check PyTorch CUDA
        try:
            import torch
            self.pytorch_version = torch.__version__
            
            if torch.cuda.is_available():
                self.cuda_available = True
                self.device = "cuda"
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.cuda_version = torch.version.cuda
                
                logger.info(f"GPU detected: {self.gpu_name} ({self.gpu_memory:.1f} GB)")
                logger.info(f"CUDA version: {self.cuda_version}")
                logger.info(f"PyTorch version: {self.pytorch_version}")
            else:
                logger.info("CUDA not available, using CPU")
                
        except ImportError:
            logger.warning("PyTorch not installed, GPU detection unavailable")
    
    def get_device(self) -> str:
        """Get the best available device"""
        return self.device
    
    def get_torch_device(self):
        """Get PyTorch device object"""
        try:
            import torch
            return torch.device(self.device)
        except:
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        return {
            "cuda_available": self.cuda_available,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": round(self.gpu_memory, 1) if self.gpu_memory else 0,
            "cuda_version": self.cuda_version,
            "pytorch_version": self.pytorch_version
        }
    
    def optimize_for_gpu(self):
        """Apply GPU optimizations"""
        if not self.cuda_available:
            return
        
        try:
            import torch
            
            # Enable cuDNN benchmark for faster inference
            torch.backends.cudnn.benchmark = True
            
            # Enable TF32 for faster training on Ampere+ GPUs
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            
            logger.info("GPU optimizations applied: cuDNN benchmark, TF32 enabled")
            
        except Exception as e:
            logger.warning(f"Could not apply GPU optimizations: {e}")
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.cuda_available:
            try:
                import torch
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleared")
            except:
                pass
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get GPU memory usage (used, total in GB)"""
        if not self.cuda_available:
            return (0, 0)
        
        try:
            import torch
            used = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return (round(used, 2), round(total, 2))
        except:
            return (0, 0)
    
    def print_status(self):
        """Print GPU status"""
        info = self.get_info()
        print("\n" + "=" * 50)
        print("GPU Status")
        print("=" * 50)
        
        if info["cuda_available"]:
            print(f"✅ CUDA Available: Yes")
            print(f"🎮 GPU: {info['gpu_name']}")
            print(f"💾 GPU Memory: {info['gpu_memory_gb']} GB")
            print(f"🔧 CUDA Version: {info['cuda_version']}")
            print(f"📦 PyTorch Version: {info['pytorch_version']}")
            print(f"⚡ Device: {info['device']}")
            
            used, total = self.get_memory_usage()
            print(f"📊 Memory Usage: {used:.2f} / {total:.2f} GB")
        else:
            print(f"❌ CUDA Available: No")
            print(f"⚡ Device: CPU")
            print(f"📦 PyTorch Version: {info['pytorch_version'] or 'Not installed'}")
        
        print("=" * 50 + "\n")


def check_opencv_cuda() -> bool:
    """Check if OpenCV is built with CUDA support"""
    try:
        import cv2
        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count > 0
    except:
        return False


def get_recommended_settings() -> Dict[str, Any]:
    """Get recommended settings based on available hardware"""
    gpu = GPUConfig()
    
    settings = {
        "whisper_model": "small",
        "detection_model": "yolov8s",
        "detection_sample_rate": 5,
        "batch_size": 1,
        "num_workers": 4
    }
    
    if gpu.cuda_available:
        # GPU-accelerated settings
        if gpu.gpu_memory >= 16:
            settings["whisper_model"] = "large"
            settings["detection_model"] = "yolov8x"
            settings["batch_size"] = 8
        elif gpu.gpu_memory >= 8:
            settings["whisper_model"] = "medium"
            settings["detection_model"] = "yolov8l"
            settings["batch_size"] = 4
        elif gpu.gpu_memory >= 4:
            settings["whisper_model"] = "small"
            settings["detection_model"] = "yolov8m"
            settings["batch_size"] = 2
        else:
            settings["whisper_model"] = "tiny"
            settings["detection_model"] = "yolov8n"
            settings["batch_size"] = 1
    
    return settings


# Global GPU config instance
gpu_config = GPUConfig()


if __name__ == "__main__":
    gpu_config.print_status()
    
    print("Recommended Settings:")
    import json
    print(json.dumps(get_recommended_settings(), indent=2))