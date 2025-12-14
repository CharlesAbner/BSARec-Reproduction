import subprocess
import re
import csv
import os
import time
from itertools import product

# -----------------------------------------------------------------
# 配置区域
# -----------------------------------------------------------------

# ReChorus 文件夹名称
RECHORUS_DIR_NAME = 'ReChorus'

# 日志文件名
LOG_FILE = 'gru4rec_tuning_results.csv'

# 要搜索的参数网格
# 注意：在 ReChorus 等大多数库中，GRU4Rec 的 hidden_size 通常直接等于 emb_size
PARAM_GRID = {
    'lr': [1e-3, 5e-4],
    # 如果你想搜索 hidden_size，可以取消下面这行的注释，并从 COMMON_ARGS 中移除 emb_size
    # 'emb_size': [64, 128] 
}

# 固定参数 (GRU4Rec 通用设置)
COMMON_ARGS = [
    '--model_name', 'GRU4Rec',
    '--dataset', 'MovieLens_1M/ML_1MTOPK',
    '--emb_size', '64',      # 对应 hidden_size
    '--num_layers', '2',     # 对应 GRU 层数，通常保持和 SASRec 一致
    '--batch_size', '256',
    '--history_max', '50',
    '--test_all', '1',
    '--l2', '1e-6',
    '--gpu', '0',
    '--loss', 'BPR'          # 强制使用 BPR，避免 CE Loss 的 0 指标问题
]

# -----------------------------------------------------------------
# 第 1 部分：日志记录器
# -----------------------------------------------------------------

def parse_and_log(params, raw_output):
    """
    解析输出并将结果记录到 CSV
    """
    metrics = {
        'HR@5': -1, 'NDCG@5': -1,
        'HR@10': -1, 'NDCG@10': -1,
        'HR@20': -1, 'NDCG@20': -1
    }

    try:
        # 抓取 Test After Training 块
        test_block_match = re.search(r"Test After Training:\s*\((.*?)\)", raw_output)
        
        if test_block_match:
            content = test_block_match.group(1)
            for key in metrics.keys():
                match = re.search(fr"{key}:(\d+\.\d+)", content)
                if match:
                    metrics[key] = float(match.group(1))
        else:
            print("  [Warning] Could not find 'Test After Training' block.")

    except Exception as e:
        print(f"  [Error] Parsing failed: {e}")

    # 准备日志数据 (动态获取 params 中的 key)
    log_data = {
        'Model': 'GRU4Rec',
        **params, # 自动解包当前的参数组合
        **metrics
    }

    # 写入 CSV
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        # 动态生成表头：Model + 参数名 + 指标名
        fieldnames = ['Model'] + list(params.keys()) + \
                     ['HR@5', 'NDCG@5', 'HR@10', 'NDCG@10', 'HR@20', 'NDCG@20']
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(log_data)
        print(f"  -> Result Logged: HR@20={log_data['HR@20']}, NDCG@20={log_data['NDCG@20']}")

# -----------------------------------------------------------------
# 第 2 部分：实验运行器
# -----------------------------------------------------------------

def run_grid_search():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(base_dir, RECHORUS_DIR_NAME)

    # 生成所有参数组合
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    total = len(combinations)
    print(f"Total experiments to run: {total}")
    print(f"Results will be saved to: {LOG_FILE}\n")

    for i, params in enumerate(combinations):
        print(f"=== Running Experiment {i+1}/{total} ===")
        # 打印当前参数组合
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        print(f"Parameters: {param_str}")

        # 组装动态参数
        specific_args = []
        for k, v in params.items():
            specific_args.append(f"--{k}")
            specific_args.append(str(v))
        
        # 完整命令
        cmd = ['python', 'src/main.py'] + COMMON_ARGS + specific_args
        
        print(f"Command: {' '.join(cmd)}")

        try:
            start_time = time.time()
            # 执行命令
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
            
            # 记录结果
            parse_and_log(params, result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"!!! Error in Experiment {i+1} !!!")
            # 打印错误日志
            print('\n'.join(e.stderr.splitlines()[-20:]))
            
        print("-" * 50)

    print("\nAll GRU4Rec grid search experiments completed!")

if __name__ == "__main__":
    run_grid_search()