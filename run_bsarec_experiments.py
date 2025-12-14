import subprocess
import re
import csv
import os
import time

# -----------------------------------------------------------------
# 配置区域
# -----------------------------------------------------------------

# ReChorus 文件夹的名称 (根据你的实际目录结构修改)
RECHORUS_DIR_NAME = 'ReChorus'

# 日志文件名
LOG_FILE = 'bsarec_experiment_results.csv'

# === 论文最优参数 (Benchmark Best Params) ===
# 依据 Table 6 (image_186a5e.png) 针对 ML-1M 数据集的配置
BEST_DATASET = 'MovieLens_1M/ML_1MTOPK'
BEST_LR = 0.0005        # lr: 5e-4
BEST_ALPHA = 0.3        # alpha: 0.3
BEST_C = 9              # c: 9
BEST_HEADS = 4          # h: 4 (ML-1M 是 4, Beauty/Toys 是 1)

# === 通用固定参数 ===
COMMON_ARGS = [
    '--dataset', BEST_DATASET,
    '--emb_size', '64',     # 论文设置 D=64
    '--num_layers', '2',    # 论文设置 L=2
    '--batch_size', '256',  # 论文设置
    '--history_max', '50',  # 论文设置 N=50
    '--test_all', '1',      # 强制全量排序测试
    '--l2', '1e-6',         # L2 正则
    '--gpu', '0',           # 指定 GPU
    '--loss', 'BPR'         # 暂时用 BPR (如果你修好了 CE，请改为 'CE')
]

# -----------------------------------------------------------------
# 第 1 部分：日志记录器
# -----------------------------------------------------------------

def parse_and_log(experiment_type, params, raw_output):
    """
    解析 ReChorus 输出并记录到 CSV
    """
    metrics = {
        'HR@5': -1, 'NDCG@5': -1,
        'HR@10': -1, 'NDCG@10': -1,
        'HR@20': -1, 'NDCG@20': -1
    }

    try:
        # 抓取 Test 结果块
        test_block_match = re.search(r"Test After Training:\s*\((.*?)\)", raw_output)
        
        if test_block_match:
            content = test_block_match.group(1)
            for key in metrics.keys():
                match = re.search(fr"{key}:(\d+\.\d+)", content)
                if match:
                    metrics[key] = float(match.group(1))
        else:
            print("  [Warning] Could not find 'Test After Training' block in output.")

    except Exception as e:
        print(f"  [Error] Parsing output failed: {e}")

    # 准备日志数据
    log_data = {
        'Experiment_Type': experiment_type,
        'Model': params.get('model_name', 'Unknown'),
        'alpha': params.get('alpha', 'N/A'),
        'c': params.get('c', 'N/A'),
        'num_heads': params.get('num_heads', 'N/A'), # 记录 num_heads
        'lr': params.get('lr', 'N/A'),
        **metrics
    }

    # 写入 CSV
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        fieldnames = ['Experiment_Type', 'Model', 'alpha', 'c', 'num_heads', 'lr', 
                      'HR@5', 'NDCG@5', 'HR@10', 'NDCG@10', 'HR@20', 'NDCG@20']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
        print(f"  -> Logged: HR@20={log_data['HR@20']}, NDCG@20={log_data['NDCG@20']}")

# -----------------------------------------------------------------
# 第 2 部分：命令执行器
# -----------------------------------------------------------------

def run_command(experiment_name, specific_args):
    """
    执行单次实验
    """
    print(f"\n=== Running: {experiment_name} ===")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(base_dir, RECHORUS_DIR_NAME)

    # 组装完整命令
    cmd = ['python', 'src/main.py', '--model_name', 'BSARec'] + COMMON_ARGS + specific_args
    
    # 打印命令 (方便你复制调试)
    print(f"Command: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        duration = time.time() - start_time
        print(f"  -> Finished in {duration:.2f}s")
        
        # 简单解析参数用于日志
        params = {'model_name': 'BSARec'}
        # 先把 COMMON_ARGS 里的默认值放进去 (比如 dataset)
        for i in range(len(COMMON_ARGS)):
            if COMMON_ARGS[i].startswith('--') and i+1 < len(COMMON_ARGS):
                 params[COMMON_ARGS[i].lstrip('-')] = COMMON_ARGS[i+1]
        # 再把 specific_args 覆盖进去
        for i in range(len(specific_args)):
            if specific_args[i].startswith('--') and i+1 < len(specific_args):
                params[specific_args[i].lstrip('-')] = specific_args[i+1]

        parse_and_log(experiment_name, params, result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"!!! Error during execution !!!")
        # 打印最后几行错误信息
        print('\n'.join(e.stderr.splitlines()[-20:]))

# -----------------------------------------------------------------
# 第 3 部分：实验编排 (Main)
# -----------------------------------------------------------------

def main():
    print(f"Output will be saved to: {os.path.abspath(LOG_FILE)}")
    
    # 定义标准参数模板，方便复用
    # 注意：这里把 num_heads 也加进去了
    def get_args(alpha, c, heads=BEST_HEADS, lr=BEST_LR):
        return [
            '--lr', str(lr),
            '--alpha', str(alpha),
            '--c', str(c),
            '--num_heads', str(heads)
        ]

    # ==========================
    # 1. Baseline (复现最优结果)
    # ==========================
    # 预期: 这是你的“天花板”分数
    run_command(
        "Baseline (Best Params)",
        get_args(alpha=BEST_ALPHA, c=BEST_C)
    )

    # ==========================
    # 2. Ablation Studies (消融实验)
    # ==========================
    # 验证架构有效性 (Table 3 & 7 的逻辑)
    
    # 2.1 Only Self-Attention (去掉归纳偏置) -> alpha = 0
    run_command(
        "Ablation: Only Self-Attention (alpha=0)",
        get_args(alpha=0.0, c=BEST_C)
    )

    # 2.2 Only AIB (去掉自注意力) -> alpha = 1
    run_command(
        "Ablation: Only AIB (alpha=1)",
        get_args(alpha=1.0, c=BEST_C)
    )

    # ==========================
    # 3. Sensitivity Analysis (超参敏感性)
    # ==========================
    # 对应 Figure 5 & 6 & 8 & 9
    
    # 3.1 Alpha 敏感性 (固定 c=9, 变 alpha)
    # Baseline 已经跑过 0.3 了，这里跑剩下的
    for alpha in [0.1, 0.5, 0.7, 0.9]:
        run_command(
            f"Sensitivity: alpha={alpha}",
            get_args(alpha=alpha, c=BEST_C)
        )

    # 3.2 C 敏感性 (固定 alpha=0.3, 变 c)
    # Baseline 已经跑过 9 了，这里跑剩下的
    for c in [1, 3, 5, 7]:
        run_command(
            f"Sensitivity: c={c}",
            get_args(alpha=BEST_ALPHA, c=c)
        )

    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()