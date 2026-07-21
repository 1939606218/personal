import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from load_data import loadData, applyPCA, normalization, generater
from model_2 import ASGTNet
from loss_func import DynamicCombinedLoss
from train_test import run_training, test


def get_pca_channel(dataname):
    if dataname == 'hermiston':
        return 8
    if dataname == 'farmland':
        return 8
    if dataname == 'river':
        return 120
    if dataname == 'bayArea':
        return 64
    if dataname == 'santaBarbara':
        return 64
    return 64


def run_single_experiment(dataname, encoder_dim, global_layers, gnn_layers, train_ratio,
                          num_epochs=200, batch_size=64, learning_rate=0.0005,
                          patch_size=5, device=None, run_id=0, output_dir='main_pater',
                          X1_pca=None, X2_pca=None, Y=None, pca_channel=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    print(f"\n--- Experiment: dataset={dataname} enc={encoder_dim}, gbl={global_layers}, gnn={gnn_layers}, tr={train_ratio} ---")

    # If precomputed PCA data provided, use it; otherwise fall back to loading
    if X1_pca is not None and X2_pca is not None and Y is not None and pca_channel is not None:
        pass
    else:
        pca_channel = get_pca_channel(dataname)
        X1, X2, Y = loadData(dataname)
        X1 = normalization(X1)
        X2 = normalization(X2)
        X1_pca = applyPCA(X1, channel=pca_channel)
        X2_pca = applyPCA(X2, channel=pca_channel)

    (TRAIN_SIZE, TEST_SIZE, train_iter, test_iter,
     all_iter, all_position_indices, height, width,
     ce_criterion, alpha) = generater(X1_pca, X2_pca, Y, batch_size, train_ratio, device, windowSize=patch_size, noise_std=0)

    criterion = DynamicCombinedLoss(num_classes=2, lambda_=0.5)

    model = ASGTNet(num_channels=pca_channel, patch_size=patch_size,
                    global_layers=global_layers, gnn_layers=gnn_layers, encoder_dim=encoder_dim)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # ensure output directories exist
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    best_model_filename = f"{dataname}_enc{encoder_dim}_gbl{global_layers}_gnn{gnn_layers}_tr{str(train_ratio).replace('.','p')}_run{run_id}.pth"
    best_model_path = os.path.join(models_dir, best_model_filename)

    total_train_seconds, best_epoch, best_value = run_training(
        model, train_iter, test_iter, num_epochs, optimizer, criterion, device, scheduler,
        best_model_path=best_model_path
    )

    # load best model and evaluate
    best_model = ASGTNet(num_channels=pca_channel, patch_size=patch_size,
                         global_layers=global_layers, gnn_layers=gnn_layers, encoder_dim=encoder_dim)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)

    test_loss, test_oa, test_f1, test_precision, test_recall, test_kappa = test(best_model, test_iter, criterion, device)
    print(f"[Result] OA={test_oa:.4f} | F1={test_f1:.4f} | Value={test_oa+test_f1+test_precision+test_recall+test_kappa:.4f}")

    return {
        'oa': float(test_oa),
        'f1': float(test_f1),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'kappa': float(test_kappa),
        'best_epoch': int(best_epoch),
        'best_value': float(best_value)
    }


def plot_and_save(x, y, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    # 可配置项
    datanames = ['farmland', 'bayArea', 'santaBarbara']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 200
    batch_size = 64

    encoder_dims = [32, 64, 128, 256, 512, 1024]
    global_layers_list = [1, 2, 3, 4, 5]
    gnn_layers_list = [1, 2, 3, 4, 5]
    train_ratios = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]

    # results organized by dataset -> hyperparam -> value
    results = {dn: {'encoder_dim': {}, 'global_layers': {}, 'gnn_layers': {}, 'train_ratio': {}} for dn in datanames}

    output_root = 'main_pater'
    os.makedirs(output_root, exist_ok=True)

    # iterate datasets and hyperparameters
    for dn in datanames:
        print(f"\n=== Running experiments for dataset: {dn} ===")

        # load and preprocess dataset once
        pca_channel = get_pca_channel(dn)
        X1, X2, Y = loadData(dn)
        X1 = normalization(X1)
        X2 = normalization(X2)
        X1_pca = applyPCA(X1, channel=pca_channel)
        X2_pca = applyPCA(X2, channel=pca_channel)

        # encoder dim sweep
        for enc in encoder_dims:
            res = run_single_experiment(dn, encoder_dim=enc, global_layers=2, gnn_layers=2,
                                        train_ratio=0.01, num_epochs=num_epochs, batch_size=batch_size,
                                        device=device, output_dir=output_root,
                                        X1_pca=X1_pca, X2_pca=X2_pca, Y=Y, pca_channel=pca_channel)
            results[dn]['encoder_dim'][str(enc)] = res
            with open(os.path.join(output_root, 'hyperparam_results.json'), 'w') as f:
                json.dump(results, f, indent=4)

        # global layers sweep
        for gl in global_layers_list:
            res = run_single_experiment(dn, encoder_dim=512, global_layers=gl, gnn_layers=2,
                                        train_ratio=0.01, num_epochs=num_epochs, batch_size=batch_size,
                                        device=device, output_dir=output_root,
                                        X1_pca=X1_pca, X2_pca=X2_pca, Y=Y, pca_channel=pca_channel)
            results[dn]['global_layers'][str(gl)] = res
            with open(os.path.join(output_root, 'hyperparam_results.json'), 'w') as f:
                json.dump(results, f, indent=4)

        # gnn layers sweep
        for gl in gnn_layers_list:
            res = run_single_experiment(dn, encoder_dim=512, global_layers=2, gnn_layers=gl,
                                        train_ratio=0.01, num_epochs=num_epochs, batch_size=batch_size,
                                        device=device, output_dir=output_root,
                                        X1_pca=X1_pca, X2_pca=X2_pca, Y=Y, pca_channel=pca_channel)
            results[dn]['gnn_layers'][str(gl)] = res
            with open(os.path.join(output_root, 'hyperparam_results.json'), 'w') as f:
                json.dump(results, f, indent=4)

        # train ratio sweep
        for tr in train_ratios:
            res = run_single_experiment(dn, encoder_dim=512, global_layers=2, gnn_layers=2,
                                        train_ratio=tr, num_epochs=num_epochs, batch_size=batch_size,
                                        device=device, output_dir=output_root,
                                        X1_pca=X1_pca, X2_pca=X2_pca, Y=Y, pca_channel=pca_channel)
            results[dn]['train_ratio'][str(tr)] = res
            with open(os.path.join(output_root, 'hyperparam_results.json'), 'w') as f:
                json.dump(results, f, indent=4)

    # 绘图：改进版 - 多指标对比（每张图包含所有指标）
    plots_dir = os.path.join(output_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    metrics = ['oa', 'f1', 'precision', 'recall', 'kappa']
    metric_names = ['OA', 'F1', 'Precision', 'Recall', 'Kappa']
    metric_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'd', 'v']

    # ==================== 方案1: 单指标多数据集的对比图（保持原有风格，优化美观度）====================
    for metric, metric_name, color, marker in zip(metrics, metric_names, metric_colors, markers):
        plt.figure(figsize=(10, 6))

        for dn in datanames:
            # encoder_dim
            enc_x = encoder_dims
            enc_y = [results[dn]['encoder_dim'][str(x)][metric] for x in enc_x]
            plt.plot(enc_x, enc_y, marker=marker, markersize=8, linewidth=2,
                    color=color if len(datanames) == 1 else None, label=f'{dn} - {metric_name}', alpha=0.8)

        plt.xlabel('Encoder Dimension', fontsize=12, fontweight='bold')
        plt.ylabel(metric_name, fontsize=12, fontweight='bold')
        plt.title(f'{metric_name} vs Encoder Dimension', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=10, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{metric_name.lower()}_encoder_dim.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ==================== 方案2: 多指标单数据集的对比图（推荐用于论文）====================
    for dn in datanames:
        # encoder_dim 多指标对比
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for idx, (metric, metric_name, color, marker) in enumerate(zip(metrics, metric_names, metric_colors, markers)):
            ax = axes[idx]
            enc_x = encoder_dims
            enc_y = [results[dn]['encoder_dim'][str(x)][metric] for x in enc_x]

            ax.plot(enc_x, enc_y, marker=marker, markersize=6, linewidth=2,
                   color=color, alpha=0.8)
            ax.set_xlabel('Encoder Dim', fontsize=10, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
            ax.set_title(metric_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', labelsize=9)

        plt.suptitle(f'{dn.upper()}: Metrics vs Encoder Dimension', fontsize=13, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{dn}_all_metrics_encoder_dim.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # global_layers 多指标对比
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for idx, (metric, metric_name, color, marker) in enumerate(zip(metrics, metric_names, metric_colors, markers)):
            ax = axes[idx]
            gl_x = global_layers_list
            gl_y = [results[dn]['global_layers'][str(x)][metric] for x in gl_x]

            ax.plot(gl_x, gl_y, marker=marker, markersize=6, linewidth=2,
                   color=color, alpha=0.8)
            ax.set_xlabel('Global Layers', fontsize=10, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
            ax.set_title(metric_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', labelsize=9)

        plt.suptitle(f'{dn.upper()}: Metrics vs Global Layers', fontsize=13, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{dn}_all_metrics_global_layers.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # gnn_layers 多指标对比
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for idx, (metric, metric_name, color, marker) in enumerate(zip(metrics, metric_names, metric_colors, markers)):
            ax = axes[idx]
            gnn_x = gnn_layers_list
            gnn_y = [results[dn]['gnn_layers'][str(x)][metric] for x in gnn_x]

            ax.plot(gnn_x, gnn_y, marker=marker, markersize=6, linewidth=2,
                   color=color, alpha=0.8)
            ax.set_xlabel('GNN Layers', fontsize=10, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
            ax.set_title(metric_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', labelsize=9)

        plt.suptitle(f'{dn.upper()}: Metrics vs GNN Layers', fontsize=13, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{dn}_all_metrics_gnn_layers.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # train_ratio 多指标对比（对数x轴可能更合适）
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for idx, (metric, metric_name, color, marker) in enumerate(zip(metrics, metric_names, metric_colors, markers)):
            ax = axes[idx]
            tr_x = train_ratios
            tr_y = [results[dn]['train_ratio'][str(x)][metric] for x in tr_x]

            ax.plot([x * 100 for x in tr_x], tr_y, marker=marker, markersize=6,
                   linewidth=2, color=color, alpha=0.8)
            ax.set_xlabel('Train Ratio (%)', fontsize=10, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
            ax.set_title(metric_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', labelsize=9)
            ax.set_xscale('log')  # 对数坐标更适合展示训练比例

        plt.suptitle(f'{dn.upper()}: Metrics vs Train Ratio', fontsize=13, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{dn}_all_metrics_train_ratio.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ==================== 方案3: 组合图 - 同一图上显示所有指标（适合快速概览）====================
    for dn in datanames:
        plt.figure(figsize=(12, 7))
        for metric, metric_name, color, marker in zip(metrics, metric_names, metric_colors, markers):
            enc_x = encoder_dims
            enc_y = [results[dn]['encoder_dim'][str(x)][metric] for x in enc_x]
            plt.plot(enc_x, enc_y, marker=marker, markersize=6, linewidth=2.5,
                    color=color, label=metric_name, alpha=0.8)

        plt.xlabel('Encoder Dimension', fontsize=12, fontweight='bold')
        plt.ylabel('Metric Value', fontsize=12, fontweight='bold')
        plt.title(f'{dn.upper()}: All Metrics vs Encoder Dimension', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11, framealpha=0.9, loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{dn}_combined_encoder_dim.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 保存最终结果
    with open(os.path.join(output_root, 'hyperparam_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # 保存结果到Excel文件
    excel_path = os.path.join(output_root, 'hyperparam_results.xlsx')
    save_results_to_excel(results, excel_path, datanames, encoder_dims,
                         global_layers_list, gnn_layers_list, train_ratios)

    print(f"\nAll experiments finished. Results saved to:")
    print(f"  - {output_root}/hyperparam_results.json")
    print(f"  - {excel_path}")
    print(f"  - {plots_dir}/")


def save_results_to_excel(results, output_path, datanames, encoder_dims, global_layers_list, gnn_layers_list, train_ratios):
    """
    将实验结果保存到Excel文件，每个数据集一个sheet，每个sheet包含所有超参数实验结果
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for dn in datanames:
            # 为每个数据集创建一个sheet
            all_rows = []

            # encoder_dim 实验结果
            for enc in encoder_dims:
                res = results[dn]['encoder_dim'][str(enc)]
                all_rows.append({
                    'Dataset': dn,
                    'Hyperparam_Type': 'Encoder_Dim',
                    'Hyperparam_Value': enc,
                    'OA': res['oa'],
                    'F1': res['f1'],
                    'Precision': res['precision'],
                    'Recall': res['recall'],
                    'Kappa': res['kappa'],
                    'Best_Epoch': res['best_epoch'],
                    'Best_Value': res['best_value']
                })

            # global_layers 实验结果
            for gl in global_layers_list:
                res = results[dn]['global_layers'][str(gl)]
                all_rows.append({
                    'Dataset': dn,
                    'Hyperparam_Type': 'Global_Layers',
                    'Hyperparam_Value': gl,
                    'OA': res['oa'],
                    'F1': res['f1'],
                    'Precision': res['precision'],
                    'Recall': res['recall'],
                    'Kappa': res['kappa'],
                    'Best_Epoch': res['best_epoch'],
                    'Best_Value': res['best_value']
                })

            # gnn_layers 实验结果
            for gnn in gnn_layers_list:
                res = results[dn]['gnn_layers'][str(gnn)]
                all_rows.append({
                    'Dataset': dn,
                    'Hyperparam_Type': 'GNN_Layers',
                    'Hyperparam_Value': gnn,
                    'OA': res['oa'],
                    'F1': res['f1'],
                    'Precision': res['precision'],
                    'Recall': res['recall'],
                    'Kappa': res['kappa'],
                    'Best_Epoch': res['best_epoch'],
                    'Best_Value': res['best_value']
                })

            # train_ratio 实验结果
            for tr in train_ratios:
                res = results[dn]['train_ratio'][str(tr)]
                all_rows.append({
                    'Dataset': dn,
                    'Hyperparam_Type': 'Train_Ratio',
                    'Hyperparam_Value': tr,
                    'OA': res['oa'],
                    'F1': res['f1'],
                    'Precision': res['precision'],
                    'Recall': res['recall'],
                    'Kappa': res['kappa'],
                    'Best_Epoch': res['best_epoch'],
                    'Best_Value': res['best_value']
                })

            # 创建DataFrame并保存到对应的sheet
            df = pd.DataFrame(all_rows)
            df.to_excel(writer, sheet_name=dn, index=False)

            # 格式化数值（保留4位小数）
            worksheet = writer.sheets[dn]
            for row in worksheet.iter_rows(min_row=2, max_row=len(all_rows)+1, min_col=4, max_col=9):
                for cell in row:
                    cell.number_format = '0.0000'

        # 创建一个汇总sheet，显示每个数据集每个超参数的最佳配置
        summary_rows = []
        for dn in datanames:
            # 找出每个指标在每种超参数类型下的最佳值
            for param_type, param_list in [('Encoder_Dim', encoder_dims),
                                           ('Global_Layers', global_layers_list),
                                           ('GNN_Layers', gnn_layers_list),
                                           ('Train_Ratio', train_ratios)]:
                param_key = param_type.lower().replace('_', '')
                best_oa = max(results[dn][param_key][str(v)]['oa'] for v in param_list)
                best_f1 = max(results[dn][param_key][str(v)]['f1'] for v in param_list)
                best_oa_value = [v for v in param_list if results[dn][param_key][str(v)]['oa'] == best_oa][0]
                best_f1_value = [v for v in param_list if results[dn][param_key][str(v)]['f1'] == best_f1][0]

                summary_rows.append({
                    'Dataset': dn,
                    'Param_Type': param_type,
                    'Best_OA': best_oa,
                    'Best_OA_Value': best_oa_value,
                    'Best_F1': best_f1,
                    'Best_F1_Value': best_f1_value
                })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # 格式化汇总sheet的数值
        summary_ws = writer.sheets['Summary']
        for row in summary_ws.iter_rows(min_row=2, max_row=len(summary_rows)+1, min_col=3, max_col=6):
            for cell in row:
                cell.number_format = '0.0000'

    print(f"Results saved to Excel: {output_path}")


if __name__ == '__main__':
    main()
