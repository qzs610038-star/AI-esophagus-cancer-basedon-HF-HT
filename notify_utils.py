"""
训练完成/中断通知工具
支持：系统通知（toast）、日志文件标记、暂停信号检测
"""
import os
import sys
import datetime


def notify_training_complete(model_name, epoch, best_epoch, best_pcc, status="completed"):
    """
    训练完成时通知用户
    
    参数:
        model_name: 模型名称（如 "HisToGene", "UNI2-h"）
        epoch: 当前/最终 epoch
        best_epoch: 最佳 epoch
        best_pcc: 最佳验证 PCC 值
        status: 状态 ("completed", "early_stop", "error")
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. 写入状态文件
    status_file = f"training_status_{model_name.replace(' ', '_').replace('-', '_')}.txt"
    try:
        with open(status_file, 'w', encoding='utf-8') as f:
            f.write(f"模型: {model_name}\n")
            f.write(f"状态: {status}\n")
            f.write(f"时间: {timestamp}\n")
            f.write(f"总Epoch: {epoch}\n")
            f.write(f"最佳Epoch: {best_epoch}\n")
            f.write(f"最佳Val PCC: {best_pcc:.4f}\n")
    except Exception as e:
        print(f"[WARNING] 无法写入状态文件: {e}")
    
    # 2. 控制台醒目输出
    print("\n" + "=" * 60)
    if status == "completed":
        print(f"  ✅ 训练完成！模型: {model_name}")
    elif status == "early_stop":
        print(f"  ⏹️ 早停触发！模型: {model_name}")
    elif status == "error":
        print(f"  ❌ 训练异常中断！模型: {model_name}")
    print(f"  时间: {timestamp}")
    print(f"  总Epoch: {epoch} | 最佳Epoch: {best_epoch} | 最佳PCC: {best_pcc:.4f}")
    print("=" * 60 + "\n")
    
    # 3. Windows Toast 通知（需要 plyer）
    _send_toast_notification(model_name, epoch, best_pcc, status)


def notify_training_error(model_name, epoch, error_msg):
    """
    训练异常时通知
    
    参数:
        model_name: 模型名称
        epoch: 中断时的 epoch
        error_msg: 错误信息
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    status_file = f"training_status_{model_name.replace(' ', '_').replace('-', '_')}.txt"
    try:
        with open(status_file, 'w', encoding='utf-8') as f:
            f.write(f"模型: {model_name}\n")
            f.write(f"状态: ERROR\n")
            f.write(f"时间: {timestamp}\n")
            f.write(f"中断Epoch: {epoch}\n")
            f.write(f"错误信息: {error_msg}\n")
    except Exception as e:
        print(f"[WARNING] 无法写入状态文件: {e}")
    
    print("\n" + "!" * 60)
    print(f"  ❌ 训练异常中断！模型: {model_name}")
    print(f"  时间: {timestamp}")
    print(f"  中断于Epoch: {epoch}")
    print(f"  错误: {error_msg}")
    print("!" * 60 + "\n")
    



def _send_toast_notification(model_name, epoch, best_pcc, status):
    """发送系统 Toast 通知"""
    try:
        from plyer import notification
        title = f"PFMval 训练{'完成' if status != 'error' else '中断'}"
        message = f"{model_name}: Epoch {epoch}, Best PCC {best_pcc:.4f}"
        notification.notify(
            title=title,
            message=message,
            timeout=10
        )
    except Exception:
        pass


def check_pause_signal(project_root=None):
    """
    检查是否有暂停信号文件
    
    参数:
        project_root: 项目根目录，默认为当前文件所在目录
    
    返回:
        bool: 是否存在暂停信号文件
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
    pause_file = os.path.join(project_root, "PAUSE_TRAINING")
    return os.path.exists(pause_file)


def clear_pause_signal(project_root=None):
    """
    清除暂停信号文件
    
    参数:
        project_root: 项目根目录，默认为当前文件所在目录
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
    pause_file = os.path.join(project_root, "PAUSE_TRAINING")
    if os.path.exists(pause_file):
        os.remove(pause_file)
        print(f"[INFO] 已清除暂停信号文件: {pause_file}")
