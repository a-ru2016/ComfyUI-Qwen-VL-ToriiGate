from __future__ import annotations
import torch
import os
import tempfile
# import io
import json
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from huggingface_hub import snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as modelscope_snapshot_download
from PIL import Image
from pathlib import Path
import folder_paths
from qwen_vl_utils import process_vision_info
import numpy as np
import requests
import time
import torchvision.io
from transformers import BitsAndBytesConfig
# 尝试导入opencv作为备选视频处理库
# Try importing OpenCV as an alternative video processing library
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("警告: OpenCV不可用，视频处理功能可能受限" + " | " + "Warning: OpenCV is not available, video processing functions may be limited")

# 模型注册表JSON文件路径
# Model registry JSON file path
MODEL_REGISTRY_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_registry.json")

def load_model_registry():
    """从JSON文件加载模型注册表"""
    """Load the model registry from a JSON file"""
    try:
        with open(MODEL_REGISTRY_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 模型注册表文件 {MODEL_REGISTRY_JSON} 不存在" + " | " + f"Error: Model registry file {MODEL_REGISTRY_JSON} does not exist")
        return {}
    except json.JSONDecodeError as e:
        print(f"错误: 解析模型注册表JSON文件时出错: {e}" + " | " + f"Error: Failed to parse model registry JSON file: {e}")
        return {}

# 加载模型注册表
# Load the model registry
MODEL_REGISTRY = load_model_registry()

def get_gpu_info():
    """获取GPU信息，包括显存使用情况"""
    """Get GPU information, including memory usage"""
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            total_memory = props.total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
            free_memory = total_memory - allocated_memory
            
            return {
                "available": True,
                "count": gpu_count,
                "name": props.name,
                "total_memory": total_memory,
                "allocated_memory": allocated_memory,
                "free_memory": free_memory
            }
        else:
            return {
                "available": False,
                "count": 0,
                "name": "None",
                "total_memory": 0,
                "allocated_memory": 0,
                "free_memory": 0
            }
    except Exception as e:
        print(f"获取GPU信息时出错: {e}" + " | " + f"Error getting GPU information: {e}")
        return {
            "available": False,
            "count": 0,
            "name": "None",
            "total_memory": 0,
            "allocated_memory": 0,
            "free_memory": 0
        }

def get_system_memory_info():
    """获取系统内存信息，包括总内存和可用内存"""
    """Get system memory information, including total and available memory"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total": mem.total / 1024**3,  # GB
            "available": mem.available / 1024**3,  # GB
            "used": mem.used / 1024**3,  # GB
            "percent": mem.percent
        }
    except ImportError:
        print("警告: 无法导入psutil库，系统内存检测功能将不可用" + " | " + "Warning: Failed to import psutil library, system memory detection will be unavailable")
        return {
            "total": 0,
            "available": 0,
            "used": 0,
            "percent": 0
        }

def get_device_info():
    """获取设备信息，包括GPU和CPU，并分析最佳运行设备"""
    """Get device information, including GPU and CPU, and analyze the optimal running device"""
    device_info = {
        "device_type": "unknown",
        "gpu": get_gpu_info(),
        "system_memory": get_system_memory_info(),
        "recommended_device": "cpu",  # 默认推荐CPU | Default recommended device: CPU
        "memory_sufficient": True,
        "warning_message": None
    }
    
    # 检查是否为Apple Silicon
    # Check if it's Apple Silicon
    try:
        import platform
        if platform.system() == "Darwin" and platform.processor() == "arm":
            device_info["device_type"] = "apple_silicon"
            # M1/M2芯片有统一内存，检查总内存是否充足
            # M1/M2 chips have unified memory, check if total memory is sufficient
            if device_info["system_memory"]["total"] >= 16:  # 至少16GB内存 | At least 16GB of memory
                device_info["recommended_device"] = "mps"
            else:
                device_info["memory_sufficient"] = False
                device_info["warning_message"] = "Apple Silicon芯片内存不足，建议使用至少16GB内存的设备" + " | " + "Insufficient memory for Apple Silicon chip, it is recommended to use a device with at least 16GB of memory"
            return device_info
    except:
        pass
    
    # 检查是否有NVIDIA GPU
    # Check if there's an NVIDIA GPU
    if device_info["gpu"]["available"]:
        device_info["device_type"] = "nvidia_gpu"
        # 检查GPU内存是否充足
        # Check if GPU memory is sufficient
        if device_info["gpu"]["total_memory"] >= 8:  # 至少8GB显存 | At least 8GB of VRAM
            device_info["recommended_device"] = "cuda"
        else:
            # 显存不足，但仍可使用，只是性能会受影响
            # Insufficient VRAM, but can still be used with performance impact
            device_info["memory_sufficient"] = False
            device_info["warning_message"] = "NVIDIA GPU显存不足，可能会使用系统内存，性能会下降" + " | " + "Insufficient VRAM for NVIDIA GPU, system memory may be used, performance will degrade"
            device_info["recommended_device"] = "cuda"  # 仍推荐使用GPU，但会启用内存优化 | Still recommended to use GPU with memory optimization enabled
        return device_info
    
    # 检查是否有AMD GPU (ROCm)
    # Check if there's an AMD GPU (ROCm)
    try:
        import torch
        if hasattr(torch, 'device') and torch.device('cuda' if torch.cuda.is_available() else 'cpu').type == 'cuda':
            device_info["device_type"] = "amd_gpu"
            # AMD GPU内存检查
            # AMD GPU memory check
            if device_info["gpu"]["total_memory"] >= 8:
                device_info["recommended_device"] = "cuda"
            else:
                device_info["memory_sufficient"] = False
                device_info["warning_message"] = "AMD GPU显存不足，可能会使用系统内存，性能会下降" + " | " + "Insufficient VRAM for AMD GPU, system memory may be used, performance will degrade"
                device_info["recommended_device"] = "cuda"
            return device_info
    except:
        pass
    
    # 默认为CPU
    # Default to CPU
    device_info["device_type"] = "cpu"
    # 检查系统内存是否充足
    # Check if system memory is sufficient
    if device_info["system_memory"]["total"] < 8:
        device_info["memory_sufficient"] = False
        device_info["warning_message"] = "系统内存不足，模型运行可能会非常缓慢" + " | " + "Insufficient system memory, model operation may be very slow"
    
    return device_info

def calculate_required_memory(model_name, quantization, use_cpu=False, use_mps=False):
    """根据模型名称、量化方式和设备类型计算所需内存"""
    """Calculate the required memory based on model name, quantization method, and device type"""
    model_info = MODEL_REGISTRY.get(model_name, {})
    vram_config = model_info.get("vram_requirement", {})
    
    # 检查模型是否已经量化
    # Check if the model is already quantized
    is_quantized_model = model_info.get("quantized", False)
    
    # 基础内存需求计算
    # Basic memory requirement calculation
    if is_quantized_model:
        base_memory = vram_config.get("full", 0)
    else:
        if quantization == "👍 4-bit (VRAM-friendly)":
            base_memory = vram_config.get("4bit", 0)
        elif quantization == "⚖️ 8-bit (Balanced Precision)":
            base_memory = vram_config.get("8bit", 0)
        else:
            base_memory = vram_config.get("full", 0)
    
    # 调整内存需求（CPU和MPS通常需要更多内存）
    # Adjust memory requirements (CPU and MPS usually require more memory)
    if use_cpu or use_mps:
        # CPU和MPS通常需要更多内存用于内存交换
        # CPU and MPS usually require more memory for memory swapping
        memory_factor = 1.5 if use_cpu else 1.2
        return base_memory * memory_factor
    
    return base_memory

def check_flash_attention():
    """检测Flash Attention 2支持（需Ampere架构及以上）"""
    """Check Flash Attention 2 support (requires Ampere architecture or higher)"""
    try:
        from flash_attn import flash_attn_func
        major, _ = torch.cuda.get_device_capability()
        return major >= 8  # 仅支持计算能力8.0+的GPU | Only supports GPUs with compute capability 8.0+
    except ImportError:
        return False


FLASH_ATTENTION_AVAILABLE = check_flash_attention()


def init_qwen_paths(model_name):
    """初始化模型路径，支持动态生成不同模型版本的路径"""
    """Initialize model paths, supporting dynamic generation of paths for different model versions"""
    base_dir = Path(folder_paths.models_dir).resolve()
    qwen_dir = base_dir / "Qwen" / "Qwen-VL"  # 添加VLM子目录 | Add VLM subdirectory
    model_dir = qwen_dir / model_name  # 使用模型名称作为子目录 | Use model name as subdirectory
    
    # 创建目录
    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 注册到ComfyUI
    # Register to ComfyUI
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path("Qwen", str(model_dir))
    else:
        folder_paths.folder_names_and_paths["Qwen"] = ([str(model_dir)], {'.safetensors', '.bin'})
    
    print(f"模型路径已初始化: {model_dir}" + " | " + f"Model path initialized: {model_dir}")
    return str(model_dir)


def test_download_speed(url):
    """测试下载速度，下载 5 秒"""
    """Test download speed by downloading for 5 seconds"""
    try:
        start_time = time.time()
        response = requests.get(url, stream=True, timeout=10)
        downloaded_size = 0
        for data in response.iter_content(chunk_size=1024):
            if time.time() - start_time > 5:
                break
            downloaded_size += len(data)
        end_time = time.time()
        speed = downloaded_size / (end_time - start_time) / 1024  # KB/s
        return speed
    except Exception as e:
        print(f"测试下载速度时出现错误: {e}" + " | " + f"Error testing download speed: {e}")
        return 0


def validate_model_path(model_path, model_name):
    """验证模型路径的有效性和模型文件是否齐全"""
    """Validate the effectiveness of the model path and check if model files are complete"""
    path_obj = Path(model_path)
    
    # 基本路径检查
    # Basic path check
    if not path_obj.is_absolute():
        print(f"错误: {model_path} 不是绝对路径" + " | " + f"Error: {model_path} is not an absolute path")
        return False
    
    if not path_obj.exists():
        print(f"模型目录不存在: {model_path}" + " | " + f"Model directory does not exist: {model_path}")
        return False
    
    if not path_obj.is_dir():
        print(f"错误: {model_path} 不是目录" + " | " + f"Error: {model_path} is not a directory")
        return False
    
    # 检查模型文件是否齐全
    # Check if model files are complete
    if not check_model_files_exist(model_path, model_name):
        print(f"模型文件不完整: {model_path}" + " | " + f"Model files are incomplete: {model_path}")
        return False
    
    return True


def check_model_files_exist(model_dir, model_name):
    """检查特定模型版本所需的文件是否齐全"""
    """Check if all required files for a specific model version are present"""
    if model_name not in MODEL_REGISTRY:
        print(f"错误: 未知模型版本 {model_name}" + " | " + f"Error: Unknown model version {model_name}")
        return False
    
    required_files = MODEL_REGISTRY[model_name]["required_files"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            return False
    return True


# 视频处理工具类
# Video processing utility class
class VideoProcessor:
    def __init__(self):
        # 尝试导入torchcodec作为首选视频处理库
        # Try importing torchcodec as the preferred video processing library
        self.use_torchcodec = False
        self.use_opencv = False
        
        try:
            import torchcodec
            # 检查VideoDecoder属性是否存在
            # Check if the VideoDecoder attribute exists
            if hasattr(torchcodec, 'VideoDecoder'):
                self.use_torchcodec = True
                print("使用torchcodec进行视频处理" + " | " + "Using torchcodec for video processing")
            else:
                print("torchcodec库中没有VideoDecoder属性" + " | " + "torchcodec library does not have the VideoDecoder attribute")
                raise ImportError
        except ImportError:
            print("torchcodec不可用" + " | " + "torchcodec is not available")
            if OPENCV_AVAILABLE:
                self.use_opencv = True
                print("使用OpenCV作为备选视频处理库" + " | " + "Using OpenCV as an alternative video processing library")
            else:
                print("警告: 没有找到可用的视频处理库，将尝试使用torchvision（可能有弃用警告）" + " | " + "Warning: No available video processing library found, will attempt to use torchvision (may have deprecation warnings)")
                # 抑制torchvision视频API弃用警告
                # Suppress torchvision video API deprecation warnings
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")
    
    def read_video(self, video_path):
        """读取视频文件并返回帧数据"""
        """Read video file and return frame data"""
        start_time = time.time()
        try:
            if self.use_torchcodec:
                # 使用torchcodec读取视频
                # Read video using torchcodec
                import torchcodec
                decoder = torchcodec.VideoDecoder(video_path)
                frames = []
                for frame in decoder:
                    frames.append(frame)
                fps = decoder.get_fps()
                total_frames = len(frames)
                frames = torch.stack(frames) if frames else torch.zeros(0)
                print(f"使用torchcodec成功处理视频: {video_path}" + " | " + f"Successfully processed video with torchcodec: {video_path}")
            elif self.use_opencv:
                # 使用OpenCV读取视频
                # Read video using OpenCV
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"无法打开视频文件: {video_path}" + " | " + f"Failed to open video file: {video_path}")
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # 转换为RGB并转为PyTorch张量
                    # Convert to RGB and convert to PyTorch tensor
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    frames.append(frame)
                
                # 修正：使用release()方法释放资源
                # Fix: Use release() method to release resources
                cap.release()
                frames = torch.stack(frames) if frames else torch.zeros(0)
                print(f"使用OpenCV成功处理视频: {video_path}" + " | " + f"Successfully processed video with OpenCV: {video_path}")
            else:
                # 使用torchvision读取视频（弃用API）
                # Read video using torchvision (deprecated API)
                frames, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
                fps = info["video_fps"]
                total_frames = frames.shape[0]
                frames = frames.permute(0, 3, 1, 2).float() / 255.0  # 转换为[B, C, H, W]格式 | Convert to [B, C, H, W] format
                print(f"使用torchvision成功处理视频: {video_path}" + " | " + f"Successfully processed video with torchvision: {video_path}")
            
            process_time = time.time() - start_time
            print(f"视频处理完成: {video_path}, 总帧数: {total_frames}, FPS: {fps:.2f}, 处理时间: {process_time:.3f}s" + " | " + f"Video processing completed: {video_path}, total frames: {total_frames}, FPS: {fps:.2f}, processing time: {process_time:.3f}s")
            return frames, fps, total_frames
            
        except Exception as e:
            print(f"视频处理错误: {e}" + " | " + f"Video processing error: {e}")
            return None, None, None

class QwenVisionParser:
    def __init__(self):
        # 默认使用注册表中的第一个默认模型
        # Use the first default model in the registry by default
        default_model = next((name for name, info in MODEL_REGISTRY.items() if info.get("default", False)), 
                            list(MODEL_REGISTRY.keys())[0])
        
        # 重置环境变量，避免干扰
        # Reset environment variables to avoid interference
        os.environ.pop("HUGGINGFACE_HUB_CACHE", None)     

        self.current_model_name = default_model
        self.current_quantization = None  # 记录当前的量化配置 | Record the current quantization configuration
        self.model_path = init_qwen_paths(self.current_model_name)
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"模型路径: {self.model_path}" + " | " + f"Model path: {self.model_path}")
        print(f"缓存路径: {self.cache_dir}" + " | " + f"Cache path: {self.cache_dir}")
        
        # 验证并创建缓存目录
        # Validate and create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.video_processor = VideoProcessor()  # 初始化视频处理器 | Initialize video processor
        self.last_generated_text = ""  # 保存上次生成的文本，用于调试 | Save last generated text for debugging
        self.generation_stats = {"count": 0, "total_time": 0}  # 统计生成性能 | Statistics for generation performance
        
        # 初始化设备信息
        # Initialize device information
        self.device_info = get_device_info()
        self.default_device = self.device_info["recommended_device"]
        
        print(f"检测到的设备: {self.device_info['device_type']}" + " | " + f"Detected device: {self.device_info['device_type']}")
        print(f"自动选择的运行设备: {self.default_device}" + " | " + f"Automatically selected running device: {self.default_device}")
        
        if not self.device_info["memory_sufficient"]:
            print(f"警告: {self.device_info['warning_message']}" + " | " + f"Warning: {self.device_info['warning_message']}")
        
        # 初始化内存优化选项
        # Initialize memory optimization options
        self.optimize_for_low_memory = not self.device_info["memory_sufficient"]

    def clear_model_resources(self):
        """释放当前模型占用的资源"""
        """Release resources occupied by the current model"""
        if self.model is not None:
            print("释放当前模型占用的资源..." + " | " + "Releasing resources occupied by the current model...")
            del self.model, self.processor, self.tokenizer
            self.model = None
            self.processor = None
            self.tokenizer = None
            torch.cuda.empty_cache()  # 清理GPU缓存 | Clean GPU cache

        # 更新设备信息（可选，因为初始化时已设置）
        # Update device information (optional, already set during initialization)
        # self.device_info = get_device_info()
        # self.default_device = self.device_info["recommended_device"]
        
        # 初始化内存优化选项
        # Initialize memory optimization options
        self.optimize_for_low_memory = not self.device_info["memory_sufficient"]


    def check_memory_requirements(self, model_name, quantization):
        """检查当前设备内存是否满足模型要求，必要时调整量化级别"""
        """Check if the current device memory meets the model requirements, adjust quantization level if necessary"""
        # 使用自动选择的设备
        # Use the automatically selected device
        device = self.default_device
        use_cpu = device == "cpu"
        use_mps = device == "mps"
        
        # 计算所需内存
        # Calculate required memory
        required_memory = calculate_required_memory(model_name, quantization, use_cpu, use_mps)
        
        if use_cpu or use_mps:
            # 检查系统内存
            # Check system memory
            available_memory = self.device_info["system_memory"]["available"]
            memory_type = "系统内存" + " | " + "System memory"
        else:
            # 检查GPU内存
            # Check GPU memory
            available_memory = self.device_info["gpu"]["free_memory"]
            memory_type = "GPU显存" + " | " + "GPU VRAM"
        
        # 添加20%的安全余量
        # Add a 20% safety margin
        safety_margin = 1.2
        required_memory_with_margin = required_memory * safety_margin
        
        print(f"模型 {model_name} (量化: {quantization}) 需要 {required_memory:.2f} GB {memory_type}" + " | " + f"Model {model_name} (quantization: {quantization}) requires {required_memory:.2f} GB {memory_type}")
        print(f"考虑安全余量后，需要 {required_memory_with_margin:.2f} GB {memory_type}" + " | " + f"After considering safety margin, requires {required_memory_with_margin:.2f} GB {memory_type}")
        print(f"当前可用 {memory_type}: {available_memory:.2f} GB" + " | " + f"Currently available {memory_type}: {available_memory:.2f} GB")
        
        # 如果内存不足，自动调整量化级别
        # Automatically adjust quantization level if memory is insufficient
        if required_memory_with_margin > available_memory:
            print(f"警告: 所选量化级别需要的{memory_type}超过可用内存，自动调整量化级别" + " | " + f"Warning: The selected quantization level requires more {memory_type} than available, automatically adjusting quantization level")
            
            # 降级策略
            # Downgrade strategy
            if quantization == "🚫 None (Original Precision)":
                print("将量化级别从'无量化'调整为'8-bit'" + " | " + "Adjusting quantization level from 'No quantization' to '8-bit'")
                return "⚖️ 8-bit (Balanced Precision)"
            elif quantization == "⚖️ 8-bit (Balanced Precision)":
                print("将量化级别从'8-bit'调整为'4-bit'" + " | " + "Adjusting quantization level from '8-bit' to '4-bit'")
                return "👍 4-bit (VRAM-friendly)"
            else:
                # 已经是4-bit，无法再降级
                # Already at 4-bit, cannot downgrade further
                print(f"错误: 即使使用4-bit量化，模型仍然需要更多{memory_type}" + " | " + f"Error: Even with 4-bit quantization, the model still requires more {memory_type}")
                raise RuntimeError(f"错误: 可用{memory_type}不足，需要至少 {required_memory_with_margin:.2f} GB，但只有 {available_memory:.2f} GB" + " | " + f"Error: Insufficient available {memory_type}, requires at least {required_memory_with_margin:.2f} GB, but only {available_memory:.2f} GB available")
        
        return quantization

    
    def load_model(self, model_name, quantization):
        # 检查内存需求并可能调整量化级别
        # Check memory requirements and possibly adjust quantization level
        adjusted_quantization = self.check_memory_requirements(model_name, quantization)
        
        # 使用自动选择的设备
        # Use the automatically selected device
        device = self.default_device
        print(f"使用设备: {device}" + " | " + f"Using device: {device}")

        # 检查是否需要重新加载模型
        # Check if the model needs to be reloaded
        if (self.model is not None and 
            self.current_model_name == model_name and 
            self.current_quantization == quantization):
            print(f"使用已加载的模型: {model_name}，量化: {quantization}" + " | " + f"Using already loaded model: {model_name}, quantization: {quantization}")
            return
        
        # 需要重新加载，先释放现有资源
        # Need to reload, release existing resources first
        self.clear_model_resources()
        
        # 更新当前模型名称和路径
        # Update current model name and path
        self.current_model_name = model_name
        self.model_path = init_qwen_paths(self.current_model_name)
        self.current_quantization = quantization

        # 检查模型文件是否存在且完整
        # Check if model files exist and are complete
        if not validate_model_path(self.model_path, self.current_model_name):
            print(f"检测到模型文件缺失，正在为你下载 {model_name} 模型，请稍候..." + " | " + f"Model files detected as missing, downloading {model_name} model for you, please wait...")
            print(f"下载将保存在: {self.model_path}" + " | " + f"Download will be saved to: {self.model_path}")
            
            # 开始下载逻辑
            # Start download logic
            try:
                # 从注册表获取模型信息
                # Get model information from registry
                model_info = MODEL_REGISTRY[model_name]
                
                # 测试下载速度
                # Test download speed
                huggingface_test_url = f"https://huggingface.co/{model_info['repo_id']['huggingface']}/resolve/main/{model_info['test_file']}"
                modelscope_test_url = f"https://modelscope.cn/api/v1/models/{model_info['repo_id']['modelscope']}/repo?Revision=master&FilePath={model_info['test_file']}"
                huggingface_speed = test_download_speed(huggingface_test_url)
                modelscope_speed = test_download_speed(modelscope_test_url)

                print(f"Hugging Face下载速度: {huggingface_speed:.2f} KB/s" + " | " + f"Hugging Face download speed: {huggingface_speed:.2f} KB/s")
                print(f"ModelScope下载速度: {modelscope_speed:.2f} KB/s" + " | " + f"ModelScope download speed: {modelscope_speed:.2f} KB/s")

                # 根据下载速度选择优先下载源
                # Select preferred download source based on download speed
                if huggingface_speed > modelscope_speed * 1.5:
                    download_sources = [
                        (snapshot_download, model_info['repo_id']['huggingface'], "Hugging Face"),
                        (modelscope_snapshot_download, model_info['repo_id']['modelscope'], "ModelScope")
                    ]
                    print("基于下载速度分析，优先尝试从Hugging Face下载" + " | " + "Based on download speed analysis, attempting to download from Hugging Face first")
                else:
                    download_sources = [
                        (modelscope_snapshot_download, model_info['repo_id']['modelscope'], "ModelScope"),
                        (snapshot_download, model_info['repo_id']['huggingface'], "Hugging Face")
                    ]
                    print("基于下载速度分析，优先尝试从ModelScope下载" + " | " + "Based on download speed analysis, attempting to download from ModelScope first")

                max_retries = 3
                success = False
                final_error = None
                used_cache_path = None

                for download_func, repo_id, source in download_sources:
                    for retry in range(max_retries):
                        try:
                            print(f"开始从 {source} 下载模型（第 {retry + 1} 次尝试）..." + " | " + f"Starting to download model from {source} (attempt {retry + 1})...")
                            if download_func == snapshot_download:
                                cached_path = download_func(
                                    repo_id,
                                    cache_dir=self.cache_dir,
                                    ignore_patterns=["*.msgpack", "*.h5"],
                                    resume_download=True,
                                    local_files_only=False
                                )
                            else:
                                cached_path = download_func(
                                    repo_id,
                                    cache_dir=self.cache_dir,
                                    revision="master"
                                )

                            used_cache_path = cached_path  # 记录使用的缓存路径 | Record the cache path used
                            
                            # 将下载的模型复制到模型目录
                            # Copy the downloaded model to the model directory
                            self.copy_cached_model_to_local(cached_path, self.model_path)
                            
                            print(f"成功从 {source} 下载模型到 {self.model_path}" + " | " + f"Successfully downloaded model from {source} to {self.model_path}")
                            success = True
                            break

                        except Exception as e:
                            final_error = e  # 保存最后一个错误 | Save the last error
                            if retry < max_retries - 1:
                                print(f"从 {source} 下载模型失败（第 {retry + 1} 次尝试）: {e}，即将进行下一次尝试..." + " | " + f"Failed to download model from {source} (attempt {retry + 1}): {e}, proceeding to next attempt...")
                            else:
                                print(f"从 {source} 下载模型失败（第 {retry + 1} 次尝试）: {e}，尝试其他源..." + " | " + f"Failed to download model from {source} (attempt {retry + 1}): {e}, trying other source...")
                    if success:
                        break
                else:
                    raise RuntimeError("从所有源下载模型均失败。" + " | " + "Failed to download model from all sources.")
                
                # 下载完成后再次验证
                # Verify again after download is complete
                if not validate_model_path(self.model_path, self.current_model_name):
                    raise RuntimeError(f"下载后模型文件仍不完整: {self.model_path}" + " | " + f"Model files still incomplete after download: {self.model_path}")
                
                print(f"模型 {model_name} 已准备就绪" + " | " + f"Model {model_name} is ready")
                
            except Exception as e:
                print(f"下载模型时发生错误: {e}" + " | " + f"Error occurred while downloading model: {e}")
                
                # 下载失败提示
                # Download failure prompt
                if used_cache_path:
                    print("\n⚠️ 注意：下载过程中创建了缓存文件" + " | " + "\n⚠️ Note: Cache files were created during the download")
                    print(f"缓存路径: {used_cache_path}" + " | " + f"Cache path: {used_cache_path}")
                    print("你可以前往此路径删除缓存文件以释放硬盘空间" + " | " + "You can go to this path to delete the cache files to free up disk space")
                
                raise RuntimeError(f"无法下载模型 {model_name}，请手动下载并放置到 {self.model_path}" + " | " + f"Failed to download model {model_name}, please download manually and place in {self.model_path}")

        # 模型文件完整，正常加载
        # Model files are complete, load normally
        print(f"加载模型: {self.model_path}，量化: {quantization}" + " | " + f"Loading model: {self.model_path}, quantization: {quantization}")

        # 检查模型是否已经量化
        # Check if the model is already quantized
        is_quantized_model = MODEL_REGISTRY.get(model_name, {}).get("quantized", False)
        
        # 配置量化参数
        # Configure quantization parameters
        if is_quantized_model:
            print(f"模型 {model_name} 已经是量化模型，将忽略用户的量化设置" + " | " + f"Model {model_name} is already a quantized model, user quantization settings will be ignored")
            # 对于已经量化的模型，使用原始精度加载
            # For already quantized models, load with original precision
            load_dtype = torch.float16
            quant_config = None
        else:
            # 对于非量化模型，应用用户选择的量化设置
            # For non-quantized models, apply user-selected quantization settings
            if quantization == "👍 4-bit (VRAM-friendly)":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_dtype = None  # 让量化配置决定数据类型 | Let quantization configuration determine data type
            elif quantization == "⚖️ 8-bit (Balanced Precision)":
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                load_dtype = None  # 让量化配置决定数据类型 | Let quantization configuration determine data type
            else:
                # 不使用量化，使用原始精度
                # No quantization, use original precision
                load_dtype = torch.float16
                quant_config = None

        # 配置device_map
        # Configure device_map
        if device == "cuda":
            if torch.cuda.device_count() > 0:
                device_map = {"": 0}  # 使用第一个GPU | Use first GPU
                print(f"使用GPU: {torch.cuda.get_device_name(0)}" + " | " + f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device_map = "auto"
                print("未检测到可用GPU，将尝试使用auto设备映射" + " | " + "No available GPU detected, will attempt to use auto device mapping")
        elif device == "mps":
            device_map = "auto"  # MPS不支持device_map，加载后需手动移到设备 | MPS does not support device_map, need to manually move to device after loading
        else:
            device_map = "auto"  # CPU加载 | CPU loading

        # 准备加载参数
        # Prepare loading parameters
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": load_dtype,
            "attn_implementation": "flash_attention_2" if FLASH_ATTENTION_AVAILABLE and device == "cuda" else "sdpa",
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }

        # 如果有量化配置，添加到加载参数中
        # If there's a quantization configuration, add it to the loading parameters
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        # 加载模型
        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            **load_kwargs
        ).eval()

        # 对于MPS，需要手动将模型移到设备
        # For MPS, need to manually move the model to the device
        if device == "mps":
            self.model = self.model.to("mps")

        # 编译优化
        # Compilation optimization
        if torch.__version__ >= "2.2":
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # SDP优化
        # SDP optimization
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        # 加载处理器和分词器
        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # 修复rope_scaling配置警告
        # Fix rope_scaling configuration warning
        if hasattr(self.model.config, "rope_scaling"):
            self.model.config.rope_scaling["mrope_section"] = "none"  # 禁用 MROPE 优化 | Disable MROPE optimization

    def copy_cached_model_to_local(self, cached_path, target_path):
        """将缓存的模型文件复制到目标路径"""
        """Copy cached model files to target path"""
        print(f"正在将模型从缓存复制到: {target_path}" + " | " + f"Copying model from cache to: {target_path}")
        target_path = Path(target_path)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # 使用shutil进行递归复制
        # Use shutil for recursive copying
        import shutil
        for item in Path(cached_path).iterdir():
            if item.is_dir():
                shutil.copytree(item, target_path / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target_path / item.name)
        
        # 验证复制是否成功
        # Verify if the copy was successful
        if validate_model_path(target_path, self.current_model_name):
            print(f"模型已成功复制到 {target_path}" + " | " + f"Model successfully copied to {target_path}")
        else:
            raise RuntimeError(f"复制后模型文件仍不完整: {target_path}" + " | " + f"Model files still incomplete after copy: {target_path}")

    def tensor_to_pil(self, image_tensor):
        """将图像张量转换为PIL图像"""
        """Convert image tensor to PIL image"""
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def preprocess_image(self, image):
        """预处理图像，包括尺寸调整和优化"""
        """Preprocess image, including resizing and optimization"""
        pil_image = self.tensor_to_pil(image)
        
        # 限制最大尺寸，避免过大的输入
        # Limit maximum size to avoid excessively large inputs
        max_res = 1024
        if max(pil_image.size) > max_res:
            pil_image.thumbnail((max_res, max_res))
        
        # 转换回张量并归一化
        # Convert back to tensor and normalize
        img_np = np.array(pil_image)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # 转回PIL图像
        # Convert back to PIL image
        pil_image = Image.fromarray((img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        return pil_image

    def preprocess_video(self, video_path):
        """预处理视频，包括帧提取和尺寸调整"""
        """Preprocess video, including frame extraction and resizing"""
        # 使用视频处理器读取视频
        # Read video using video processor
        frames, fps, total_frames = self.video_processor.read_video(video_path)
        
        if frames is None:
            print(f"无法处理视频: {video_path}" + " | " + f"Failed to process video: {video_path}")
            return None, None, None
        
        # 打印原始帧信息（用于调试）
        # Print original frame information (for debugging)
        if frames.numel() > 0:
            print(f"原始帧: 形状={frames.shape}, 类型={frames.dtype}, 最小值={frames.min()}, 最大值={frames.max()}" + " | " + f"Original frames: shape={frames.shape}, type={frames.dtype}, min={frames.min()}, max={frames.max()}")
        
        # 更激进的帧数量限制
        # More aggressive frame count limit
        max_frames = 15
        if total_frames > max_frames:
            # 采样帧
            # Sample frames
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            frames = frames[indices]
            print(f"视频帧数量从 {total_frames} 采样到 {len(frames)}" + " | " + f"Video frames sampled from {total_frames} to {len(frames)}")
        
        # 确保帧数据是(C, H, W)格式，并且是float32类型(0.0-1.0)
        # Ensure frame data is in (C, H, W) format and is float32 type (0.0-1.0)
        processed_frames = []
        for frame in frames:
            # 确保帧是(C, H, W)格式
            # Ensure frame is in (C, H, W) format
            if frame.dim() == 3 and frame.shape[0] not in [1, 3]:
                # 如果第一个维度不是通道数(1或3)，可能是(H, W, C)格式
                # If the first dimension is not the number of channels (1 or 3), it might be in (H, W, C) format
                frame = frame.permute(2, 0, 1)
            
            # 确保帧是float32类型(0.0-1.0)
            # Ensure frame is float32 type (0.0-1.0)
            if frame.dtype != torch.float32:
                frame = frame.float()
            
            if frame.max() > 1.0:
                # 如果像素值范围不是0.0-1.0，进行归一化
                # If pixel value range is not 0.0-1.0, normalize
                frame = frame / 255.0
            
            processed_frames.append(frame)
        
        # 调整帧大小
        # Resize frames
        resized_frames = []
        for frame in processed_frames:
            # 转换为PIL图像进行调整大小
            # Convert to PIL image for resizing
            # 先转换为(H, W, C)格式，再转换为numpy数组和uint8类型
            # First convert to (H, W, C) format, then to numpy array and uint8 type
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
            frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
            frame_pil = Image.fromarray(frame_np)
            
            # 调整大小为384x384
            # Resize to 384x384
            frame_pil = frame_pil.resize((384, 384), Image.Resampling.LANCZOS)
            
            # 转回张量 (C, H, W) 格式，float32类型(0.0-1.0)
            # Convert back to tensor (C, H, W) format, float32 type (0.0-1.0)
            frame_tensor = torch.from_numpy(np.array(frame_pil)).permute(2, 0, 1).float() / 255.0
            resized_frames.append(frame_tensor)
        
        # 转换回张量
        # Convert back to tensor
        if resized_frames:
            resized_frames = torch.stack(resized_frames)
        else:
            resized_frames = torch.zeros(0)
        
        print(f"处理后帧: 形状={resized_frames.shape}, 类型={resized_frames.dtype}" + " | " + f"Processed frames: shape={resized_frames.shape}, type={resized_frames.dtype}")
        return resized_frames, fps, len(frames) # 返回实际采样后的帧数 | Return the actual number of sampled frames
        

    @torch.no_grad()
    def process(self, model_name, quantization, prompt, max_tokens, temperature, top_p,
                repetition_penalty, image=None, video_path=None, unload_after_generation=True):
        start_time = time.time()
        
        # 确保加载正确的模型和量化配置
        # Ensure correct model and quantization configuration are loaded
        # 检查模型是否已加载且是否需要重新加载（即使名称相同）
        # Check if model is already loaded and if it needs to be reloaded (even if the name is the same)
        if (self.model is not None and 
            self.current_model_name == model_name and 
            self.current_quantization == quantization):
            # 额外检查：如果模型是预量化的，但用户选择了量化选项，仍需重新加载
            # Additional check: If the model is pre-quantized but the user selected a quantization option, still reload
            is_quantized_model = MODEL_REGISTRY.get(model_name, {}).get("quantized", False)
            user_selected_quantization = quantization in ["👍 4-bit (VRAM-friendly)", "⚖️ 8-bit (Balanced Precision)"]
            
            if is_quantized_model and user_selected_quantization:
                print(f"模型 {model_name} 已经是量化模型，将忽略用户的量化设置并重新加载" + " | " + f"Model {model_name} is already a quantized model, user quantization settings will be ignored and reloaded")
                self.clear_model_resources()
                self.load_model(model_name, "🚫 None (Original Precision)")
            else:
                print(f"使用已加载的模型: {model_name}，量化: {quantization}" + " | " + f"Using already loaded model: {model_name}, quantization: {quantization}")
        else:
            self.load_model(model_name, quantization)
        
        # 图像预处理
        # Image preprocessing
        pil_image = None
        if image is not None:
            pil_image = self.preprocess_image(image)
        
        # 视频预处理
        # Video preprocessing
        video_frames = None
        if video_path:
            video_frames, video_fps, video_frames_count = self.preprocess_video(video_path)
            if video_frames is not None:
                print(f"视频已处理: {video_path}, 帧数: {video_frames_count}, FPS: {video_fps}" + " | " + f"Video processed: {video_path}, frames: {video_frames_count}, FPS: {video_fps}")
        
        # 构建对话
        # Build conversation
        SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving visual inputs and generating text."
        if model_name.startswith("Minthy"):
            SYSTEM_PROMPT = "You are image captioning expert, creative, unbiased and uncensored."
        print(f"SYSTEM_PROMPT = {SYSTEM_PROMPT}")
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": []}
        ]
        
        # 添加图像和视频到对话
        # Add images and videos to the conversation
        if pil_image is not None:
            conversation[-1]["content"].append({"type": "image", "image": pil_image})
        
        if video_path and video_frames is not None:
            # 转换视频帧为PIL图像列表
            # Convert video frames to list of PIL images
            video_frame_list = []
            for frame in video_frames:
                frame = frame.permute(1, 2, 0).cpu().numpy() * 255
                frame = frame.astype(np.uint8)
                video_frame_list.append(Image.fromarray(frame))
            
            conversation[-1]["content"].append({"type": "video", "video": video_frame_list})
        
        # 处理用户提示
        # Process user prompt
        user_prompt = prompt if prompt.endswith(("?", ".", "！", "。", "？", "！")) else f"{prompt} "
        conversation[-1]["content"].append({"type": "text", "text": user_prompt})
        
        # 应用聊天模板
        # Apply chat template
        input_text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # 准备处理器参数
        # Prepare processor parameters
        processor_args = {
            "text": input_text,
            "return_tensors": "pt",
            "padding": True,
        }
        
        # 调用多模态处理逻辑
        # Call multimodal processing logic
        images, videos = process_vision_info(conversation)
        processor_args["images"] = images
        processor_args["videos"] = videos
        
        # 清理不再需要的大对象
        # Clean up large objects that are no longer needed
        del video_frames, images, videos
        torch.cuda.empty_cache()
        
        # 将输入移至设备
        # Move inputs to device
        inputs = self.processor(**processor_args)
        device = self.default_device
        model_inputs = {
            k: v.to(device)
            for k, v in inputs.items()
            if v is not None
        }
        
        # 确保model_inputs包含所需的键
        # Ensure model_inputs contains required keys
        if "input_ids" not in model_inputs:
            raise ValueError("处理后的输入不包含'input_ids'键" + " | " + "Processed inputs do not contain 'input_ids' key")
        
        # 生成配置
        # Generation configuration
        generate_config = {
            "max_new_tokens": max(max_tokens, 10),
            "temperature": temperature,
            "do_sample": True,
            "use_cache": True,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # 记录GPU内存使用情况
        # Record GPU memory usage
        if torch.cuda.is_available():
            pre_forward_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"生成前GPU内存使用: {pre_forward_memory:.2f} MB" + " | " + f"GPU memory usage before generation: {pre_forward_memory:.2f} MB")
        
        # 使用适当的设备进行生成
        # Generate using appropriate device
        with torch.no_grad():
            # 使用新的autocast API
            # Use new autocast API
            if device == "cuda":
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model.generate(**model_inputs, **generate_config)
            else:
                outputs = self.model.generate(**model_inputs, **generate_config)
        
        # 记录GPU内存使用情况
        # Record GPU memory usage
        if torch.cuda.is_available():
            post_forward_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"生成后GPU内存使用: {post_forward_memory:.2f} MB" + " | " + f"GPU memory usage after generation: {post_forward_memory:.2f} MB")
            print(f"生成过程中GPU内存增加: {post_forward_memory - pre_forward_memory:.2f} MB" + " | " + f"GPU memory increase during generation: {post_forward_memory - pre_forward_memory:.2f} MB")
        
        # 处理输出
        # Process outputs
        text_tokens = outputs if outputs.dim() == 2 else outputs.unsqueeze(0)
        
        # 清理不再需要的大对象
        # Clean up large objects that are no longer needed
        del outputs, inputs
        torch.cuda.empty_cache()
        
        # 截取新生成的token
        # Trim newly generated tokens
        input_length = model_inputs["input_ids"].shape[1]
        text_tokens = text_tokens[:, input_length:]  # 截取新生成的token | Trim newly generated tokens
        
        # 解码文本
        # Decode text
        text = self.tokenizer.decode(
            text_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 保存生成的文本用于调试
        # Save generated text for debugging
        self.last_generated_text = text
        del model_inputs
        torch.cuda.empty_cache()
        
        # 根据选项决定是否卸载模型
        # Decide whether to unload the model based on options
        if unload_after_generation:
            self.clear_model_resources()
            print(f"success_with_unload")
        else:
            print(f"success_keep_loaded")
        
        # 计算处理时间
        # Calculate processing time
        process_time = time.time() - start_time
        self.generation_stats["count"] += 1
        self.generation_stats["total_time"] += process_time
        
        # 打印性能统计
        # Print performance statistics
        print(f"生成完成，耗时: {process_time:.2f} 秒" + " | " + f"Generation completed, time taken: {process_time:.2f} seconds")
        if self.generation_stats["count"] > 0:
            avg_time = self.generation_stats["total_time"] / self.generation_stats["count"]
            print(f"平均生成时间: {avg_time:.2f} 秒/次" + " | " + f"Average generation time: {avg_time:.2f} seconds/time")
        
        return (text.strip(),)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    list(MODEL_REGISTRY.keys()),  # 动态生成模型选项 | Dynamically generate model options
                    {
                        "default": next((name for name, info in MODEL_REGISTRY.items() if info.get("default", False)), 
                                       list(MODEL_REGISTRY.keys())[0]),
                        "tooltip": "Select the available model version." + " | " + "选择可用的模型版本。"
                    }
                ),
                "quantization": (
                    [
                        "👍 4-bit (VRAM-friendly)",
                        "⚖️ 8-bit (Balanced Precision)",
                        "🚫 None (Original Precision)"
                    ],
                    {
                        "default": "👍 4-bit (VRAM-friendly)",
                        "tooltip": "Select the quantization level:\n✅ 4-bit: Significantly reduces VRAM usage, suitable for resource-constrained environments.\n⚖️ 8-bit: Strikes a balance between precision and performance.\n🚫 None: Uses the original floating-point precision (requires a high-end GPU)." + " | " + "选择量化级别:\n✅ 4-bit: 显著减少显存使用，适合资源受限环境。\n⚖️ 8-bit: 平衡精度和性能。\n🚫 None: 使用原始浮点精度（需要高端GPU）。"
                    }
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Describe this image in detail.",
                        "multiline": True,
                        "tooltip": "Enter a text prompt, supporting Chinese and emojis. Example: 'Describe a cat in a painter's style.'" + " | " + "输入文本提示，支持中文和表情符号。示例: '以画家风格描述一只猫。'"
                    }
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 2048,
                        "step": 16,
                        "display": "slider",
                        "tooltip": "Control the maximum length of the generated text (in tokens). \nGenerally, 100 tokens correspond to approximately 50 - 100 Chinese characters or 67 - 100 English words, but the actual number may vary depending on the text content and the model's tokenization strategy. \nRecommended range: 64 - 512." + " | " + "控制生成文本的最大长度（以token为单位）。\n一般来说，100个token大约对应50 - 100个汉字或67 - 100个英文单词，但实际数量可能因文本内容和模型的分词策略而异。\n推荐范围: 64 - 512。"
                    }
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                        "tooltip": "Control the generation diversity:\n▫️ 0.1 - 0.3: Generate structured/technical content.\n▫️ 0.5 - 0.7: Balance creativity and logic.\n▫️ 0.8 - 1.0: High degree of freedom (may produce incoherent content)." + " | " + "控制生成多样性:\n▫️ 0.1 - 0.3: 生成结构化/技术性内容。\n▫️ 0.5 - 0.7: 平衡创造性和逻辑性。\n▫️ 0.8 - 1.0: 高度自由（可能产生不连贯内容）。"
                    }
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Nucleus sampling threshold:\n▪️ Close to 1.0: Retain more candidate words (more random).\n▪️ 0.5 - 0.8: Balance quality and diversity.\n▪️ Below 0.3: Generate more conservative content." + " | " + "核采样阈值:\n▪️ 接近1.0: 保留更多候选词（更随机）。\n▪️ 0.5 - 0.8: 平衡质量和多样性。\n▪️ 低于0.3: 生成更保守的内容。"
                    }
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Control of repeated content:\n⚠️ 1.0: Default behavior.\n⚠️ >1.0 (Recommended 1.2): Suppress repeated phrases.\n⚠️ <1.0 (Recommended 0.8): Encourage repeated emphasis." + " | " + "控制重复内容:\n⚠️ 1.0: 默认行为。\n⚠️ >1.0 (推荐1.2): 抑制重复短语。\n⚠️ <1.0 (推荐0.8): 鼓励重复强调。"
                    }
                ),
                "unload_after_generation": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Process完成后自动卸载模型以释放资源。如果需要连续处理多个请求，可以禁用此选项以提高性能。" + " | " + "Automatically unload the model after processing to free up resources. If you need to process multiple requests consecutively, you can disable this option to improve performance."
                    }
                )
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Upload a reference image (supports PNG/JPG), and the model will adjust the generation result based on the image content." + " | " + "上传参考图像（支持PNG/JPG），模型将根据图像内容调整生成结果。"
                    }
                ),
                "video_path": (
                    "VIDEO_PATH",
                    {
                        "tooltip": "Enter the video file  (supports MP4/WEBM), and the model will extract visual features to assist in generation." + " | " + "输入视频文件路径（支持MP4/WEBM），模型将提取视觉特征辅助生成。"
                    }
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "🐼QwenVL"    


# Register the node
# 注册节点
NODE_CLASS_MAPPINGS = {
    "QwenVisionParser": QwenVisionParser
}

# 节点显示名称映射
# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVisionParser": "Qwen VL 🐼"
}
