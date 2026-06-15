"""训练/推理设备选择与 GPU/CPU 差异化配置"""
import torch


def resolve_device(device_name="auto"):
    """
    解析计算设备。
    device_name: auto | cuda | cpu | cuda:0 | cuda:1 ...
    """
    name = (device_name or "auto").lower().strip()

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求使用 GPU，但未检测到可用的 CUDA 设备。请安装 GPU 版 PyTorch 或改用 --device cpu。")
        return torch.device("cuda")

    if name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"请求使用 {name}，但未检测到可用的 CUDA 设备。")
        index = int(name.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"请求使用 {name}，但仅检测到 {torch.cuda.device_count()} 张 GPU。"
            )
        return torch.device(name)

    if name == "cpu":
        return torch.device("cpu")

    raise ValueError(f"不支持的设备: {device_name}，可选: auto, cuda, cpu, cuda:0")


def apply_device_settings(config, device):
    """根据设备应用 batch_size、num_workers 等差异化配置"""
    settings_key = "cuda" if device.type == "cuda" else "cpu"
    device_settings = config.get("device_settings", {}).get(settings_key, {})

    if device_settings.get("batch_size") is not None:
        config["training"]["batch_size"] = device_settings["batch_size"]
    if device_settings.get("num_workers") is not None:
        config["project"]["num_workers"] = device_settings["num_workers"]

    config["project"]["pin_memory"] = device_settings.get("pin_memory", device.type == "cuda")
    config["project"]["resolved_device"] = str(device)

    return device_settings


def configure_cudnn(config, device):
    """按设备配置 cuDNN 行为"""
    settings_key = "cuda" if device.type == "cuda" else "cpu"
    device_settings = config.get("device_settings", {}).get(settings_key, {})

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = device_settings.get("cudnn_benchmark", True)
        torch.backends.cudnn.deterministic = device_settings.get("deterministic", False)
    else:
        torch.backends.cudnn.deterministic = device_settings.get("deterministic", True)
        torch.backends.cudnn.benchmark = device_settings.get("cudnn_benchmark", False)


def print_device_info(device):
    """打印设备信息"""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        total_gb = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
        print(f"使用设备: GPU ({name}, {total_gb:.1f} GB, cuda:{index})")
    else:
        print("使用设备: CPU")

    if device.type == "cuda":
        print("  - 混合精度 (AMP): 已启用")
        print("  - cuDNN benchmark: 已启用（GPU 加速模式）")
