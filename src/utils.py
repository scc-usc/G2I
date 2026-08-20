import os
import errno
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, to_dense_adj, subgraph
from sklearn.metrics import auc
import seaborn as sns

def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise


def safe_open(path, w):
	''' Open "path" for writing, creating any parent directories as needed.'''
	mkdir_p(os.path.dirname(path))
	return open(path, w)


def accuracy(output, labels):
	preds = output.max(1)[1].type_as(labels)
	correct = preds.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)


def get_degree_matrix(adj):
	# Fixed: use adj.sum(dim=0) instead of sum(adj) to handle all tensor shapes
	return torch.diag(adj.sum(dim=0))


def normalize_adj(adj):
	# Normalize adjacancy matrix according to reparam trick in GCN paper
	A_tilde = adj + torch.eye(adj.shape[0])
	D_tilde = get_degree_matrix(A_tilde)
	# Raise to power -1/2, set all infs to 0s
	D_tilde_exp = D_tilde ** (-1 / 2)
	D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

	# Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
	norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
	return norm_adj

def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
	# Fix: edge_index should be the full edge_index tensor, not edge_index[0]
	# If edge_index is a tuple (from dense_to_sparse), use edge_index[0]
	if isinstance(edge_index, tuple):
		edge_index_tensor = edge_index[0]
	else:
		edge_index_tensor = edge_index

	edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index_tensor)     # Get all nodes involved
	edge_subset_relabel = subgraph(edge_subset[0], edge_index_tensor, relabel_nodes=True)       # Get relabelled subset of edges
	sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
	sub_feat = features[edge_subset[0], :]
	sub_labels = labels[edge_subset[0]]
	new_index = np.array([i for i in range(len(edge_subset[0]))])
	node_dict = dict(zip(edge_subset[0].numpy(), new_index))        # Maps orig labels to new
	# print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
	return sub_adj, sub_feat, sub_labels, node_dict


def create_symm_matrix_from_vec(vector, n_rows):
	matrix = torch.zeros(n_rows, n_rows)
	idx = torch.tril_indices(n_rows, n_rows)
	matrix[idx[0], idx[1]] = vector
	symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
	return symm_matrix


def create_vec_from_symm_matrix(matrix, P_vec_size):
	idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
	vector = matrix[idx[0], idx[1]]
	return vector


def index_to_mask(index, size):
	mask = torch.zeros(size, dtype=torch.bool, device=index.device)
	mask[index] = 1
	return mask

def get_S_values(pickled_results, header):
	df_prep = []
	for example in pickled_results:
		if example != []:
			df_prep.append(example[0])
	return pd.DataFrame(df_prep, columns=header)


def redo_dataset_pgexplainer_format(dataset, train_idx, test_idx):

	dataset.data.train_mask = index_to_mask(train_idx, size=dataset.data.num_nodes)
	dataset.data.test_mask = index_to_mask(test_idx[len(test_idx)], size=dataset.data.num_nodes)





def calculate_auccc(cost_history, coverage_history, budget_limit, total_positive_nodes):
    """
    计算归一化的 AUCCC (0-1之间)，并自动处理曲线延伸。
    """
    # 1. 拷贝数据，避免修改原始列表
    x = list(cost_history)
    y = list(coverage_history)
    
    # 2. 确保起点是 (0, 0)
    if x[0] != 0:
        x.insert(0, 0)
        y.insert(0, 0)
        
    # 3. 关键：水平延伸到 Budget Limit
    # 无论是因为提前完成了，还是因为没招了提前退出了，最后的状态都应该保持直到预算耗尽
    if x[-1] < budget_limit:
        x.append(budget_limit)
        y.append(y[-1])
        
    # 4. 归一化 (Normalization)
    # 将 Cost 映射到 [0, 1]，将 Coverage 映射到 [0, 1]
    x_norm = np.array(x) / budget_limit
    y_norm = np.array(y) / total_positive_nodes
    
    # 5. 计算面积
    # 为了防止 x 有超出 1 的情况（万一 Cost > Budget），截断一下
    valid_mask = x_norm <= 1.0 + 1e-9
    x_norm = x_norm[valid_mask]
    y_norm = y_norm[valid_mask]
    
    # 如果最后一个点因为截断丢了，需要补一个 (1.0, last_y)
    if x_norm[-1] < 1.0:
        x_norm = np.append(x_norm, 1.0)
        y_norm = np.append(y_norm, y_norm[-1])
        
    score = auc(x_norm, y_norm)
    return score


def calculate_auccc_from_results(results):
    """
    Apply calculate_auccc to the full results dictionary from run_ALL.
    
    Args:
        results: Dictionary output from run_ALL
        
    Returns:
        Dictionary summarizing AUCCC scores:
        {
            "cf-greedy": {
                "greedy": [score_run1, score_run2, ...],
                "frequency": [...],
                "random": [...],
                "greedy_mean": 0.85, "greedy_std": 0.02, ...
            },
            ...
        }
    """
    config = results.get("config", {})
    budget = config.get("budget", 5.0)
    epsilon = config.get("epsilon", 0.1)
    
    # Re-calculate budget limit as done in the experiment
    # Note: intervention_design_model uses: budget_limit = budget * np.log(1 / epsilon)
    # But for frequency/random baselines, it uses int(...) of that.
    # To be safe and consistent with normalization, use the float value.
    if epsilon > 0:
        budget_limit = budget * np.log(1 / epsilon)
    else:
        budget_limit = budget # Fallback if epsilon is weird
        
    # Storage for scores
    # Structure: method -> strategy -> list of scores
    scores = {} 
    
    for run in results["runs"]:
        # Get total positive nodes for this run (denominator for coverage)
        # run["num_target_nodes"] is the number of True Positives that we actually tried to explain
        total_positive_nodes = run.get("num_target_nodes", 0)
        
        if total_positive_nodes == 0:
            continue
            
        methods_results = run.get("methods", {})
        
        for method_name, method_data in methods_results.items():
            if method_name not in scores:
                scores[method_name] = {}
                
            # Strategies are keys like "greedy", "frequency", "random"
            for strategy in ["greedy", "frequency", "random"]:
                if strategy not in method_data:
                    continue
                    
                strat_data = method_data[strategy]
                cost_hist = strat_data.get("cost_history", [])
                cov_hist = strat_data.get("coverage_history", [])
                
                # Check consistency
                if not cost_hist or not cov_hist:
                    val = 0.0
                else:
                    val = calculate_auccc(cost_hist, cov_hist, budget_limit, total_positive_nodes)
                
                if strategy not in scores[method_name]:
                    scores[method_name][strategy] = []
                scores[method_name][strategy].append(val)
    
    # Calculate means and stds
    summary = {}
    for method, strategies in scores.items():
        summary[method] = {}
        for strat, values in strategies.items():
            arr = np.array(values)
            summary[method][strat] = values
            summary[method][f"{strat}_mean"] = float(np.mean(arr))
            summary[method][f"{strat}_std"] = float(np.std(arr))
            
    return summary


def plot_violin_auccc(summary, title="Distribution of AUCCC Performance"):
    """
    Plot grouped violin plot for AUCCC summary.
    Requires raw data lists in the summary dict, not just means.
    """
    # 1. 转换数据为 DataFrame (Long Format)
    data_list = []
    
    methods = list(summary.keys())
    strategies = ['greedy', 'frequency', 'random']
    
    for method in methods:
        for strat in strategies:
            # 获取原始分数列表
            scores = summary[method].get(strat, [])
            # 处理单值情况 (如果只有一个run，std=0，violin画不像)
            # 为了画图效果，如果只有一个点，seabon可能会warning，但依然能画
            for score in scores:
                data_list.append({
                    'Method': method.upper(),
                    'Strategy': strat.capitalize(),
                    'AUCCC': score
                })
    
    df = pd.DataFrame(data_list)
    
    if df.empty:
        print("No data available to plot.")
        return

    # 2. 设置绘图风格
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # 3. 绘制小提琴图
    # inner='box' 会在小提琴里再画一个迷你箱线图，非常信息丰富
    # cut=0 表示不延伸超过数据范围
    ax = sns.violinplot(data=df, x='Method', y='AUCCC', hue='Strategy',
                        palette="muted", inner="box", cut=0, scale="width")
    
    # 4. (可选) 叠加散点图，以便在数据量少时也能看到真实分布
    # stripplot 会把点抖动一下画在上面
    sns.stripplot(data=df, x='Method', y='AUCCC', hue='Strategy',
                  dodge=True, color='black', alpha=0.6, size=4, legend=False, ax=ax)

    # 5. 美化
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('Normalized AUCCC', fontsize=12)
    plt.xlabel('Explanation Method', fontsize=12)
    plt.ylim(0, 1.1)
    
    # 调整图例位置
    plt.legend(title='Selection Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


def plot_coverage_comparison(
    results_file: str,
    title: str = None,
    save_path: str = None,
    figsize: tuple = None,
    explanation_methods: list = None,
    show_plot: bool = True
):
    """
    Plot coverage vs cost comparison for all explanation methods and selection strategies.

    Args:
        results_file: Path to the JSON results file from run_ALL
        title: Plot title (auto-generated if None)
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple (auto-calculated if None)
        explanation_methods: List of methods to plot (auto-detected if None)
        show_plot: Whether to display the plot

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Auto-detect explanation methods if not specified
    if explanation_methods is None:
        # Get methods from the first run
        if results['runs']:
            explanation_methods = list(results['runs'][0]['methods'].keys())
        else:
            raise ValueError("No runs found in results file")

    selection_methods = ['greedy', 'frequency', 'random']

    # Style configuration: highlight greedy (blue)
    styles = {
        'greedy': {
            'color': 'blue',
            'linestyle': '-',
            'linewidth': 3.5,
            'marker': 'o',
            'markevery': 6,
            'markersize': 8,
            'alpha': 1.0,
            'zorder': 10
        },
        'frequency': {
            'color': 'green',
            'linestyle': '--',
            'linewidth': 2.0,
            'marker': 's',
            'markevery': 6,
            'markersize': 5,
            'alpha': 0.7,
            'zorder': 5
        },
        'random': {
            'color': 'red',
            'linestyle': '-.',
            'linewidth': 2.0,
            'marker': '^',
            'markevery': 6,
            'markersize': 5,
            'alpha': 0.7,
            'zorder': 5
        }
    }

    labels = {'greedy': 'Greedy (L2G)', 'frequency': 'Frequency', 'random': 'Random'}

    # Get true positive label count (mean across runs)
    true_positive_counts = [run['num_positive_labels'] for run in results['runs']]
    mean_true_positive = np.mean(true_positive_counts)

    # Auto-calculate figure size
    n_methods = len(explanation_methods)
    if figsize is None:
        figsize = (5 * n_methods, 5)

    fig, axes = plt.subplots(1, n_methods, figsize=figsize)

    # Handle single method case
    if n_methods == 1:
        axes = [axes]

    for ax_idx, exp_method in enumerate(explanation_methods):
        ax = axes[ax_idx]

        # Draw frequency and random first, then greedy (ensure greedy is on top)
        for sel_method in ['frequency', 'random', 'greedy']:
            # Get coverage_history and cost_history
            try:
                coverages_list = [run['methods'][exp_method][sel_method]['coverage_history']
                                  for run in results['runs']]
                costs_list = [run['methods'][exp_method][sel_method]['cost_history']
                              for run in results['runs']]
            except KeyError:
                continue

            # Find max cost to align all runs
            valid_costs = [costs for costs in costs_list if costs]
            if not valid_costs:
                continue
            max_cost = max(max(costs) for costs in valid_costs)

            # Interpolate coverage at unified cost points
            cost_points = np.linspace(0, max_cost, 100)
            interpolated_coverages = []

            for coverages, costs in zip(coverages_list, costs_list):
                if len(costs) > 0:
                    # Linear interpolation
                    interp_cov = np.interp(cost_points, costs, coverages)
                    interpolated_coverages.append(interp_cov)

            if interpolated_coverages:
                mean_cov = np.mean(interpolated_coverages, axis=0)
                std_cov = np.std(interpolated_coverages, axis=0)

                style = styles[sel_method]
                ax.plot(cost_points, mean_cov, label=labels[sel_method], **style)
                ax.fill_between(cost_points, mean_cov - std_cov, mean_cov + std_cov,
                                alpha=0.15 if sel_method == 'greedy' else 0.1,
                                color=style['color'])

        # True positive label line
        ax.axhline(y=mean_true_positive, color='black', linestyle='--', linewidth=1.5,
                   label=f'True Positive Labels ({mean_true_positive:.0f})')

        ax.set_xlabel('Cumulative Cost (# of interventions)')
        ax.set_ylabel('Coverage')
        ax.set_title(f'{exp_method.upper()}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    # Set main title
    if title is None:
        title = 'Coverage vs Cost Comparison'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()

    return fig, axes


def plot_coverage_from_results(
    results: dict,
    title: str = None,
    save_path: str = None,
    figsize: tuple = None,
    explanation_methods: list = None,
    show_plot: bool = True
):
    """
    Plot coverage vs cost comparison directly from results dictionary.

    Args:
        results: Results dictionary from run_ALL
        title: Plot title (auto-generated if None)
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple (auto-calculated if None)
        explanation_methods: List of methods to plot (auto-detected if None)
        show_plot: Whether to display the plot

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    # Auto-detect explanation methods if not specified
    if explanation_methods is None:
        if results['runs']:
            explanation_methods = list(results['runs'][0]['methods'].keys())
        else:
            raise ValueError("No runs found in results")

    selection_methods = ['greedy', 'frequency', 'random']

    # Style configuration
    styles = {
        'greedy': {
            'color': 'blue',
            'linestyle': '-',
            'linewidth': 3.5,
            'marker': 'o',
            'markevery': 6,
            'markersize': 8,
            'alpha': 1.0,
            'zorder': 10
        },
        'frequency': {
            'color': 'green',
            'linestyle': '--',
            'linewidth': 2.0,
            'marker': 's',
            'markevery': 6,
            'markersize': 5,
            'alpha': 0.7,
            'zorder': 5
        },
        'random': {
            'color': 'red',
            'linestyle': '-.',
            'linewidth': 2.0,
            'marker': '^',
            'markevery': 6,
            'markersize': 5,
            'alpha': 0.7,
            'zorder': 5
        }
    }

    labels = {'greedy': 'Greedy (L2G)', 'frequency': 'Frequency', 'random': 'Random'}

    # Get true positive label count
    true_positive_counts = [run['num_positive_labels'] for run in results['runs']]
    mean_true_positive = np.mean(true_positive_counts)

    # Auto-calculate figure size
    n_methods = len(explanation_methods)
    if figsize is None:
        figsize = (5 * n_methods, 5)

    fig, axes = plt.subplots(1, n_methods, figsize=figsize)

    if n_methods == 1:
        axes = [axes]

    for ax_idx, exp_method in enumerate(explanation_methods):
        ax = axes[ax_idx]

        for sel_method in ['frequency', 'random', 'greedy']:
            try:
                coverages_list = [run['methods'][exp_method][sel_method]['coverage_history']
                                  for run in results['runs']]
                costs_list = [run['methods'][exp_method][sel_method]['cost_history']
                              for run in results['runs']]
            except KeyError:
                continue

            valid_costs = [costs for costs in costs_list if costs]
            if not valid_costs:
                continue
            max_cost = max(max(costs) for costs in valid_costs)

            cost_points = np.linspace(0, max_cost, 100)
            interpolated_coverages = []

            for coverages, costs in zip(coverages_list, costs_list):
                if len(costs) > 0:
                    interp_cov = np.interp(cost_points, costs, coverages)
                    interpolated_coverages.append(interp_cov)

            if interpolated_coverages:
                mean_cov = np.mean(interpolated_coverages, axis=0)
                std_cov = np.std(interpolated_coverages, axis=0)

                style = styles[sel_method]
                ax.plot(cost_points, mean_cov, label=labels[sel_method], **style)
                ax.fill_between(cost_points, mean_cov - std_cov, mean_cov + std_cov,
                                alpha=0.15 if sel_method == 'greedy' else 0.1,
                                color=style['color'])

        ax.axhline(y=mean_true_positive, color='black', linestyle='--', linewidth=1.5,
                   label=f'True Positive Labels ({mean_true_positive:.0f})')

        ax.set_xlabel('Cumulative Cost (# of interventions)')
        ax.set_ylabel('Coverage')
        ax.set_title(f'{exp_method.upper()}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    if title is None:
        title = 'Coverage vs Cost Comparison'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()

    return fig, axes


def print_summary_table(results_path):
    """
    Print a summary table from a results JSON file.
    
    For each method (CF, CFF, cf-greedy), shows:
      - AUCCC for Rand., Freq., DNF (greedy) as mean +- std
      - Final coverage % (final_coverage / num_target_nodes) as mean +- std
      - Running time as mean +- std
    
    Args:
        results_path: path to the results JSON, e.g.
            'results/synthetic_neighbor_feature/node250_attr10_edge300_N10_...json'
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # AUCCC
    auccc = calculate_auccc_from_results(results)
    
    # Collect coverage and time per method per run
    method_names = list(results['runs'][0]['methods'].keys())
    
    coverage_data = {}   # method -> strategy -> [ratio values]
    time_data = {}       # method -> [values]

    for run in results['runs']:
        num_target = run['num_target_nodes']
        for method in method_names:
            md = run['methods'][method]
            if method not in time_data:
                time_data[method] = []
                coverage_data[method] = {}
            time_data[method].append(md['running_time'])
            for strat in ['greedy', 'frequency', 'random']:
                if strat in md:
                    if strat not in coverage_data[method]:
                        coverage_data[method][strat] = []
                    ratio = md[strat]['final_coverage'] / num_target if num_target > 0 else 0
                    coverage_data[method][strat].append(ratio)
    
    # Print header
    print(f'Results: {results_path}')
    print(f'Config: N={results["config"]["N"]}, budget={results["config"]["budget"]}, '
          f'epsilon={results["config"]["epsilon"]}')
    print()
    
    header = f'{"Method":<12} {"Rand.":<18} {"Freq.":<18} {"DNF":<18} {"Cov%(DNF)":<18} {"Time(s)":<18}'
    print(header)
    print('-' * len(header))
    
    def fmt(vals):
        arr = np.array(vals)
        return f'{arr.mean():.4f}+-{arr.std():.4f}'
    
    def fmt_pct(vals):
        arr = np.array(vals) * 100
        return f'{arr.mean():.2f}%+-{arr.std():.2f}%'
    
    for method in method_names:
        # AUCCC
        rand_str = fmt(auccc[method]['random']) if 'random' in auccc.get(method, {}) else 'N/A'
        freq_str = fmt(auccc[method]['frequency']) if 'frequency' in auccc.get(method, {}) else 'N/A'
        dnf_str  = fmt(auccc[method]['greedy']) if 'greedy' in auccc.get(method, {}) else 'N/A'
        
        # Coverage % (greedy/DNF)
        cov_str = fmt_pct(coverage_data[method]['greedy']) if 'greedy' in coverage_data.get(method, {}) else 'N/A'
        
        # Time
        time_str = fmt(time_data[method])
        
        print(f'{method:<12} {rand_str:<18} {freq_str:<18} {dnf_str:<18} {cov_str:<18} {time_str:<18}')
    