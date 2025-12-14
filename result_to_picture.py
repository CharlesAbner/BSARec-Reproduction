import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. 全局审美设置 (学术风) ---
sns.set_theme(style="ticks", context="notebook", font_scale=1.0)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = True        
plt.rcParams['grid.alpha'] = 0.3        
plt.rcParams['grid.linestyle'] = '--'   

# 学术配色
COLORS = {
    "blue": "#4c72b0",   
    "orange": "#dd8452", 
    "green": "#55a868",  
    "red": "#c44e52",    
    "purple": "#8172b3", 
    "grey": "#64b5cd"    
}

OUTPUT_DIR = 'plots_smart_ylim'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_plot(filename):
    save_path = os.path.join(OUTPUT_DIR, filename)
    sns.despine() 
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 图片已保存: {save_path}")

# --- 2. 核心辅助函数：智能 Y 轴范围 ---
def get_smart_ylim(data_series_list, padding_bottom=0.3, padding_top=0.2):
    """
    根据数据动态计算 Y 轴范围。
    padding_bottom=0.3: 底部留出 30% 的极差空间
    padding_top=0.2: 顶部留出 20% 的极差空间
    这样既不会贴底，也不会贴顶，也不会强制从0开始。
    """
    # 展平所有数据以找到全局最大最小值
    all_values = pd.concat(data_series_list)
    min_val = all_values.min()
    max_val = all_values.max()
    
    diff = max_val - min_val
    if diff == 0: diff = min_val * 0.1 # 防止单点数据导致错误

    # 动态计算上下限
    y_bottom = min_val - diff * padding_bottom
    y_top = max_val + diff * padding_top
    
    # 保护机制：如果是正数指标（如NDCG），尽量不要让轴变成负数，除非数据本身这就接近0
    if y_bottom < 0 and min_val >= 0:
        y_bottom = 0 if min_val < 0.05 else min_val * 0.8 # 如果数据很小才归零，否则给个折扣
        
    return y_bottom, y_top

# --- 3. 绘图函数 ---

def plot_best_model_comparison(file_path, dataset_name="Grocery"):
    if not os.path.exists(file_path): return

    df = pd.read_csv(file_path)
    plot_df = pd.melt(df, id_vars=['Model'], value_vars=['NDCG@10', 'HR@10'], 
                      var_name='Metric', value_name='Score')

    plt.figure(figsize=(6, 4))
    
    ax = sns.barplot(
        x='Model', y='Score', hue='Metric', data=plot_df,
        palette=[COLORS["blue"], COLORS["orange"]], 
        edgecolor="white", linewidth=1.5, alpha=0.9,
        zorder=3
    )
    
    plt.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    plt.grid(axis='x', visible=False)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=2, fontsize=9, color='#333333')

    plt.title(f'Performance Comparison on {dataset_name}', fontsize=13, weight='bold', pad=15)
    plt.xlabel("") 
    plt.ylabel("Metrics Score", fontsize=11)
    
    plt.legend(frameon=False, ncol=2, loc='upper left', fontsize=10)
    
    # 柱状图还是建议从0开始，体现绝对值大小
    plt.ylim(0, plot_df['Score'].max() * 1.25)
    
    save_plot(f'comparison_{dataset_name.lower()}.png')

def plot_hyperparameter_sensitivity(file_path):
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    best_row = df[df['Experiment_Type'] == 'Baseline (Best Params)'].copy()

    # --- Alpha 曲线 ---
    alpha_df = df[df['Experiment_Type'].str.contains('alpha=', na=False)].copy()
    if not best_row.empty: alpha_df = pd.concat([alpha_df, best_row], ignore_index=True)
    
    if not alpha_df.empty and 'alpha' in alpha_df.columns:
        alpha_df['alpha'] = pd.to_numeric(alpha_df['alpha'], errors='coerce')
        alpha_df = alpha_df[~alpha_df['alpha'].isin([0, 1, 0.0, 1.0])]
        alpha_df = alpha_df.drop_duplicates(subset=['alpha']).sort_values(by='alpha')

        plt.figure(figsize=(6, 4))
        
        plt.plot(alpha_df['alpha'], alpha_df['NDCG@10'], marker='o', markersize=8, 
                 color=COLORS["blue"], label='NDCG@10', linewidth=2.5)
        plt.plot(alpha_df['alpha'], alpha_df['HR@10'], marker='s', markersize=8, 
                 color=COLORS["orange"], label='HR@10', linewidth=2.5, linestyle='--')
        
        plt.title(r'Sensitivity Analysis: $\alpha$ (Weight)', fontsize=13, weight='bold', pad=15)
        plt.xlabel(r'Alpha ($\alpha$)', fontsize=11)
        plt.ylabel('Score', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(frameon=True, fontsize=10)

        # [关键修改] 使用智能 Y 轴范围
        y_bot, y_top = get_smart_ylim([alpha_df['NDCG@10'], alpha_df['HR@10']])
        plt.ylim(y_bot, y_top)
        
        save_plot('sensitivity_alpha.png')

    # --- C 曲线 ---
    c_df = df[df['Experiment_Type'].str.contains('c=', na=False)].copy()
    if not best_row.empty: c_df = pd.concat([c_df, best_row], ignore_index=True)

    if not c_df.empty and 'c' in c_df.columns:
        c_df['c'] = pd.to_numeric(c_df['c'], errors='coerce')
        c_df = c_df.drop_duplicates(subset=['c']).sort_values(by='c')

        plt.figure(figsize=(6, 4))
        
        plt.plot(c_df['c'], c_df['NDCG@10'], marker='D', markersize=7, 
                 color=COLORS["purple"], label='NDCG@10', linewidth=2.5)
        plt.plot(c_df['c'], c_df['HR@10'], marker='^', markersize=8, 
                 color=COLORS["green"], label='HR@10', linewidth=2.5, linestyle='--')
        
        plt.title('Sensitivity Analysis: Context Window (c)', fontsize=13, weight='bold', pad=15)
        plt.xlabel('Window Size (c)', fontsize=11)
        plt.ylabel('Score', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(frameon=True, fontsize=10)
        
        plt.xticks(c_df['c'].unique()) 

        # [关键修改] 使用智能 Y 轴范围
        y_bot, y_top = get_smart_ylim([c_df['NDCG@10'], c_df['HR@10']])
        plt.ylim(y_bot, y_top)
        
        save_plot('sensitivity_c.png')

def plot_ablation_study(file_path):
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    mask = df['Experiment_Type'].str.contains('Baseline') | df['Experiment_Type'].str.contains('Ablation')
    ablation_df = df[mask].copy()
    if ablation_df.empty: return

    def rename_label(label):
        if 'Baseline' in label: return 'BSARec\n(Full)'
        if 'Only Self-Attention' in label: return 'w/o AIB'
        if 'Only AIB' in label: return 'w/o SA'
        return label
    ablation_df['Short_Label'] = ablation_df['Experiment_Type'].apply(rename_label)

    plt.figure(figsize=(5, 3.5)) 
    
    bar_colors = [COLORS["green"], COLORS["red"], COLORS["blue"]]
    bars = plt.bar(ablation_df['Short_Label'], ablation_df['NDCG@10'], 
                   color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6,
                   alpha=0.9, zorder=3)
    
    plt.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    plt.title('Ablation Study (NDCG@10)', fontsize=13, weight='bold', pad=15)
    
    # 柱状图建议保持从0开始
    plt.ylim(0, ablation_df['NDCG@10'].max() * 1.3) 
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10, weight='bold')

    save_plot('ablation_study.png')

if __name__ == "__main__":
    GROCERY_FILE = 'grocery_experiment_results.csv'
    ML1M_FILE    = 'ml1m_experiment_results.csv'
    BSA_FILE     = 'bsarec_experiment_results.csv'
    
    print("--- 正在生成优化版图表 (智能 Y 轴) ---")
    plot_best_model_comparison(GROCERY_FILE, dataset_name="Grocery")
    plot_best_model_comparison(ML1M_FILE, dataset_name="ML-1M")
    plot_hyperparameter_sensitivity(BSA_FILE)
    plot_ablation_study(BSA_FILE)
    print(f"--- 完成！请查看 {OUTPUT_DIR} 文件夹 ---")