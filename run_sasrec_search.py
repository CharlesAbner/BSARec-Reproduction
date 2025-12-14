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

# 日志文件名 (为了区分，我们用独立的文件名)
LOG_FILE = 'sasrec_tuning_results.csv'

# 要搜索的参数网格 (2 x 3 = 6 种组合)
PARAM_GRID = {
    'lr': [1e-3, 5e-4],       # 0.001, 0.0005
    'num_heads': [1, 2, 4]    # 不同的头数
}

# 固定参数 (SASRec 通用设置)
# 注意：SASRec 不需要 alpha 和 c
COMMON_ARGS = [
    '--model_name', 'SASRec',
    '--dataset', 'MovieLens_1M/ML_1MTOPK',
    '--emb_size', '64',
    '--num_layers', '2',
    '--batch_size', '256',
    '--history_max', '50',
    '--test_all', '1',
    '--l2', '1e-6',
    '--gpu', '0',
    '--loss', 'BPR' # 如果你使用的是 CE Loss，请在这里修改
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

    # 准备日志数据
    log_data = {
        'Model': 'SASRec',
        'lr': params['lr'],
        'num_heads': params['num_heads'],
        **metrics
    }

    # 写入 CSV
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        fieldnames = ['Model', 'lr', 'num_heads', 
                      'HR@5', 'NDCG@5', 'HR@10', 'NDCG@10', 'HR@20', 'NDCG@20']
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
        print(f"Parameters: lr={params['lr']}, num_heads={params['num_heads']}")

        # 组装动态参数
        specific_args = [
            '--lr', str(params['lr']),
            '--num_heads', str(params['num_heads'])
        ]
        
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
            # 打印最后几行错误日志
            print('\n'.join(e.stderr.splitlines()[-20:]))
            
        print("-" * 50)

    print("\nAll grid search experiments completed!")

if __name__ == "__main__":
    run_grid_search()