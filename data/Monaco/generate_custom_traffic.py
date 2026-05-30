#!/usr/bin/env python3
"""
生成 Monaco 路网的时变流量文件
特征：基础流量 325 veh/h，峰值倍数 4 → 单条 1300 veh/h
高峰时段（15-30分钟）并发流数量严格 ≤ 4 条，总峰值流量 = 1300 * 4 = 5200 veh/h
"""

import os
import numpy as np

# ========== 流量参数 ==========
BASE_FLOW = 325
multipliers = [1, 2, 3, 4, 4, 4, 3, 2, 1, 1, 1]
TIMES = np.arange(0, 3301, 300)

# ========== 并发控制（修正后）==========
vols_a = [1, 1, 2, 2, 2, 2, 2, 1, 1, 0, 0]
vols_b = [0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1]   # 关键修改：索引3改为2

# 可选：验证高峰时段（索引3~5）总并发数 = 4
for i in [3, 4, 5]:
    total = vols_a[i] + vols_b[i]
    if total != 4:
        print(f"警告：时段{i}总并发数为{total}，建议调整为4")
        # 不强制断言，允许用户根据需求微调

# ========== OD 对库（每个子列表4条）==========
flows = [
    [  # flows[0]
        ('-10114#1', '-10079', '10115#2 -10109'),
        ('-10114#1', '-10079', '-10114#0 10108#0 gneE5'),
        ('-10114#1', '-10079', '-10114#0 10108#0 10102'),
        ('-10114#1', '10076', '-10114#0 10107 10102')
    ],
    [  # flows[1]
        ('10096#1', '10063', '10089#3'),
        ('-10185#1', '-10071#3', 'gneE20'),
        ('10096#1', '10063', '10109'),
        ('-10185#1', '-10061#5', 'gneE19')
    ],
    [  # flows[2]
        ('10052#1', '10104', '10181#1 -10089#3'),
        ('-10064#9', '10104', '-10068 10102'),
        ('-10051#2', '10043', '10181#1 gneE4'),
        ('-10064#9', '-10110', '-10064#4 -10064#3')
    ],
    [  # flows[3]
        ('10061#4', '-10085', '10065#2 10102'),
        ('10071#3', '10085', '10065#2 -10064#3'),
        ('-10070#1', '-10086', 'gneE9'),
        ('-10063', '10085', 'gneE8')
    ]
]

OUTPUT_DIR = "./data/Monaco/"

def generate_rou_xml():
    flow_str = '  <flow id="f%s" departPos="random_free" from="%s" to="%s" via="%s" begin="%d" end="%d" vehsPerHour="%d" type="car"/>\n'
    output = '<routes>\n  <vType id="car" length="5" accel="5" decel="10" speedDev="0.1"/>\n'
    for i in range(len(TIMES)-1):
        t_begin, t_end = TIMES[i], TIMES[i+1]
        rate = int(BASE_FLOW * multipliers[i])
        k = 0
        
        # 从 flows[0] 和 flows[1] 中轮流取，总共取 vols_a[i] 条
        count_a = vols_a[i]
        for idx in range(count_a):
            # 轮流使用 flows[0] 和 flows[1]
            j = idx % 2
            ind = idx // 2   # 注意：如果 count_a > 2 可能会超出范围，但此处 count_a <=2
            src, sink, via = flows[j][ind]
            output += flow_str % (f"{i}_{k}", src, sink, via, t_begin, t_end, rate)
            k += 1
        
        # 从 flows[2] 和 flows[3] 中轮流取，总共取 vols_b[i] 条
        count_b = vols_b[i]
        for idx in range(count_b):
            j = 2 + (idx % 2)
            ind = idx // 2
            src, sink, via = flows[j][ind]
            output += flow_str % (f"{i}_{k}", src, sink, via, t_begin, t_end, rate)
            k += 1
    output += '</routes>\n'
    return output

def generate_sumocfg_xml(rou_filename):
    return f'''<configuration>
  <input>
    <net-file value="in/most.net.xml"/>
    <route-files value="in/{rou_filename}"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="3600"/>
  </time>
</configuration>
'''

def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "in"), exist_ok=True)
    rou_filename = "most.rou.xml"
    rou_content = generate_rou_xml()
    rou_path = os.path.join(OUTPUT_DIR, "in", rou_filename)
    with open(rou_path, 'w') as f:
        f.write(rou_content)
    print(f"生成路由文件: {rou_path}")
    
    sumocfg_content = generate_sumocfg_xml(rou_filename)
    sumocfg_path = os.path.join(OUTPUT_DIR, "most.sumocfg")
    with open(sumocfg_path, 'w') as f:
        f.write(sumocfg_content)
    print(f"生成 SUMO 配置: {sumocfg_path}")
    
    peak_rate = int(BASE_FLOW * max(multipliers))
    peak_concurrent = max([vols_a[i] + vols_b[i] for i in range(len(TIMES)-1)])
    peak_total = peak_rate * peak_concurrent
    print(f"\n流量特征:")
    print(f"  基础流量: {BASE_FLOW} veh/h")
    print(f"  峰值倍数: {max(multipliers)}")
    print(f"  单条峰值: {peak_rate} veh/h")
    print(f"  高峰并发数: {peak_concurrent} 条")
    print(f"  总峰值流量: {peak_total} veh/h")

if __name__ == "__main__":
    main()