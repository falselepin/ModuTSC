# 提取工具函数, 生成车辆流量曲线等
import random

def generate_vehicle_curve(steps=100, min_val=10, max_val=80, traffic_pattern="uniform"):
    arr = []
    if traffic_pattern == "grid":
        base = random.randint(min_val, max_val // 2)
        for i in range(steps):
            base += random.randint(-3, 3)
            base = max(min_val, min(max_val, base))
            arr.append(base)
    elif traffic_pattern == "arterial":
        base = min_val + 5
        for i in range(steps):
            if i < steps // 3:
                base += random.randint(0, 4)
            elif i < 2 * steps // 3:
                base += random.randint(-2, 2)
            else:
                base -= random.randint(0, 3)
            base = max(min_val, min(max_val, base))
            arr.append(base)
    elif traffic_pattern == "ring":
        base = (min_val + max_val) // 2
        for i in range(steps):
            base += random.randint(-5, 5)
            base = max(min_val, min(max_val, base))
            arr.append(base)
    else:
        base = (min_val + max_val) // 2
        for i in range(steps):
            base += random.randint(-2, 2)
            base = max(min_val, min(max_val, base))
            arr.append(base)
    return arr