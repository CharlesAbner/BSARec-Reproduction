import subprocess
import re
import csv
import os
import time

# -----------------------------------------------------------------
# 配置区域
# -----------------------------------------------------------------

# ReChorus 文件夹名称
RECHORUS_DIR_NAME = 'ReChorus'

# 日志文件名
LOG_FILE = 'grocery_experiment_results.csv'

# 通用固定参数 (所有模型公用)
COMMON_ARGS = [
    '--dataset', 'Grocery_and_Gourmet_Food',
    '--test_all', '1',      # 强制全量排序
    '--emb_size', '64',
    '--num_layers', '2',
    '--batch_size', '256',
    '--history_max', '50',
    '--l2', '1e-6',
    '--gpu', '0',
    '--loss', 'BPR'         # 统一使用 BPR Loss
]

# 定义要运行的三个实验配置
# 格式: (模型名称, 该模型特有的参数列表)
EXPERIMENTS = [
    # 1. BSARec (Ours) - 参数迁移自 Amazon Beauty (alpha=0.7, c=5, h=1)
    (
        'BSARec', 
        ['--lr', '0.0005', '--num_heads', '1', '--alpha', '0.7', '--c', '5']
    ),
    
    # 2. SASRec (Baseline 1) - 标准参数
    (
        'SASRec', 
        ['--lr', '0.0005', '--num_heads', '1']
    ),
    
    # 3. GRU4Rec (Baseline 2) - 标准参数
    (
        'GRU4Rec', 
        ['--lr', '0.0005'] # GRU4Rec 没有 num_heads
    )
]

# -----------------------------------------------------------------
# 第 1 部分：日志记录器
# -----------------------------------------------------------------

def parse_and_log(model_name, specific_args, raw_output):
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

    # 简单的参数解析，用于记录日志
    # 将 list 形式的参数转为字符串记录，方便查看
    args_str = " ".join(specific_args)

    # 准备日志数据
    log_data = {
        'Model': model_name,
        'Specific_Args': args_str,
        **metrics
    }

    # 写入 CSV
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        fieldnames = ['Model', 'Specific_Args', 
                      'HR@5', 'NDCG@5', 'HR@10', 'NDCG@10', 'HR@20', 'NDCG@20']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(log_data)
        print(f"  -> Result Logged: HR@20={log_data['HR@20']}, NDCG@20={log_data['NDCG@20']}")

# -----------------------------------------------------------------
# 第 2 部分：实验运行器
# -----------------------------------------------------------------

def run_experiments():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(base_dir, RECHORUS_DIR_NAME)

    total = len(EXPERIMENTS)
    print(f"Total experiments to run: {total}")
    print(f"Results will be saved to: {LOG_FILE}\n")

    for i, (model_name, specific_args) in enumerate(EXPERIMENTS):
        print(f"=== Running Experiment {i+1}/{total}: {model_name} ===")
        
        # 完整命令
        cmd = ['python', 'src/main.py', '--model_name', model_name] + COMMON_ARGS + specific_args
        
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
            parse_and_log(model_name, specific_args, result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"!!! Error in Experiment {i+1} ({model_name}) !!!")
            # 打印最后几行错误日志
            print('\n'.join(e.stderr.splitlines()[-20:]))
            
        print("-" * 50)

    print("\nAll Grocery experiments completed!")

if __name__ == "__main__":
    run_experiments()