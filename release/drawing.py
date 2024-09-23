import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate


def draw_train(target_dir, postfix, file_type, data_range):
    """
    epoch_idx | loss | sr_loss | sr_weight | rec_loss | rec_weight
    """
    data_file = f'{target_dir}/train.{file_type}'
    if file_type == 'csv':
        df = pd.read_csv(data_file)
        epoch = np.array(list(df.index)) + 1
        loss = df['loss'].values[data_range]
        sr_loss = df['sr_loss'].values[data_range]
        rec_loss = df['rec_loss'].values[data_range]
        sr_weight = df['sr_weight'].values[data_range]
        rec_weight = df['rec_weight'].values[data_range]
    elif file_type == 'txt':
        epoch = []
        loss = []
        sr_loss = []
        sr_weight = []
        rec_loss = []
        rec_weight = []
        with open(data_file) as df:
            for epoch_idx, line in enumerate(df.readlines()):
                if epoch_idx < data_range[0]:
                    continue
                if epoch_idx >= data_range[1]:
                    continue
                epoch.append(epoch_idx + 1)
                line = line.split()
                loss.append(float(line[1]))
                sr_loss.append(float(line[3]))
                sr_weight.append(float(line[5]))
                rec_loss.append(float(line[7]))
                rec_weight.append(float(line[9]))
        loss = np.array(loss)
        sr_loss = np.array(sr_loss)
        sr_weight = np.array(sr_weight)
        rec_loss = np.array(rec_loss)
        rec_weight = np.array(rec_weight)
    else:
        raise ValueError(f'Unknown file type: {file_type}')

    loss_file = f'{target_dir}/loss.{postfix}'
    losses_file = f'{target_dir}/losses.{postfix}'
    weight_file = f'{target_dir}/weight.{postfix}'
    weight_file2 = f'{target_dir}/weight_ratio.{postfix}'
    ratio_file = f'{target_dir}/ratio.{postfix}'

    inter = interpolate.make_interp_spline(epoch[::5], loss[::5])
    epoch_range = np.linspace(epoch[0], epoch[-1], 500)
    smooth_loss = inter(epoch_range)
    plt.figure(figsize=(9, 6))
    plt.title('Total Loss', fontsize=20)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('loss', fontsize=15)
    plt.grid()
    plt.plot(epoch, loss, linewidth=3, label='total loss', color='#00b33c', alpha=0.5)
    plt.plot(epoch_range, smooth_loss, linewidth=3, linestyle='--', color='#595959', label='smooth loss')
    plt.legend(fontsize=20)
    plt.savefig(loss_file)

    plt.figure(figsize=(9, 6))
    plt.title('Losses', fontsize=20)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('loss', fontsize=15)
    plt.grid()
    plt.plot(epoch, sr_loss, linewidth=3, label='sr loss', color='#00b33c')
    plt.plot(epoch, rec_loss, linewidth=3, label='rec loss', color='#13005A')
    plt.legend(fontsize=20)
    plt.savefig(losses_file)

    plt.figure(figsize=(9, 6))
    plt.title('Loss Weights', fontsize=20)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('weight', fontsize=15)
    plt.grid()
    plt.plot(epoch, sr_weight, linewidth=3, label='$\sigma_s$', color='#00b33c')
    plt.plot(epoch, rec_weight, linewidth=3, label='$\sigma_r$', color='#13005A')
    plt.legend(fontsize=20)
    plt.savefig(weight_file)

    plt.figure(figsize=(9, 6))
    plt.title('Loss Weights', fontsize=20)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('weight', fontsize=15)
    # plt.xlim(1, 50)
    # plt.ylim(0.7, 1.2)
    plt.grid()
    plt.plot(epoch, sr_weight, linewidth=3, label='$\sigma_s$', color='#00b33c')
    plt.plot(epoch, rec_weight, linewidth=3, label='$\sigma_r$', color='#13005A')
    plt.plot(epoch, sr_weight/rec_weight, linestyle='--', linewidth=3, label='$\sigma_s/\sigma_r$', color='#595959')
    plt.legend(fontsize=20)
    plt.savefig(weight_file2)

    plt.figure(figsize=(9, 6))
    plt.title('Loss Weights Ratio', fontsize=20)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('ratio', fontsize=15)
    plt.grid()
    plt.plot(epoch, sr_weight/rec_weight, linewidth=3, label='weights ratio', color='#00b33c')
    plt.legend(fontsize=20)
    plt.savefig(ratio_file)


def draw_eval(target_dir, postfix, file_type, data_range):
    """
    epoch_idx | loss | sr_loss | sr_weight | rec_loss | rec_weight
    """
    data_file = f'{target_dir}/train.{file_type}'
    if file_type == 'csv':
        df = pd.read_csv(data_file)
        epoch = np.array(list(df.index)) + 1
        loss = df['loss'].values[data_range]
        sr_loss = df['sr_loss'].values[data_range]
        rec_loss = df['rec_loss'].values[data_range]
        sr_weight = df['sr_weight'].values[data_range]
        rec_weight = df['rec_weight'].values[data_range]
    elif file_type == 'txt':
        epoch = []
        loss = []
        sr_loss = []
        sr_weight = []
        rec_loss = []
        rec_weight = []
        with open(data_file) as df:
            for epoch_idx, line in enumerate(df.readlines()):
                if epoch_idx < data_range[0]:
                    continue
                if epoch_idx >= data_range[1]:
                    continue
                epoch.append(epoch_idx + 1)
                line = line.split()
                loss.append(float(line[1]))
                sr_loss.append(float(line[3]))
                sr_weight.append(float(line[5]))
                rec_loss.append(float(line[7]))
                rec_weight.append(float(line[9]))
        loss = np.array(loss)
        sr_loss = np.array(sr_loss)
        sr_weight = np.array(sr_weight)
        rec_loss = np.array(rec_loss)
        rec_weight = np.array(rec_weight)
    else:
        raise ValueError(f'Unknown file type: {file_type}')

    loss_file = f'{target_dir}/loss.{postfix}'
    losses_file = f'{target_dir}/losses.{postfix}'
    weight_file = f'{target_dir}/weight.{postfix}'
    weight_file2 = f'{target_dir}/weight_ratio.{postfix}'
    ratio_file = f'{target_dir}/ratio.{postfix}'

    inter = interpolate.make_interp_spline(epoch[::5], loss[::5])
    epoch_range = np.linspace(epoch[0], epoch[-1], 500)
    smooth_loss = inter(epoch_range)
    plt.figure(figsize=(9, 6))
    plt.title('Total Loss', fontsize=20)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('loss', fontsize=15)
    plt.grid()
    plt.plot(epoch, loss, linewidth=3, label='total loss', color='#00b33c', alpha=0.5)
    plt.plot(epoch_range, smooth_loss, linewidth=3, linestyle='--', color='#595959', label='smooth loss')
    plt.legend(fontsize=20)
    plt.savefig(loss_file)

    plt.figure(figsize=(9, 6))
    plt.title('Losses', fontsize=20)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('loss', fontsize=15)
    plt.grid()
    plt.plot(epoch, sr_loss, linewidth=3, label='sr loss', color='#00b33c')
    plt.plot(epoch, rec_loss, linewidth=3, label='rec loss', color='#13005A')
    plt.legend(fontsize=20)
    plt.savefig(losses_file)

    plt.figure(figsize=(9, 6))
    plt.title('Loss Weights', fontsize=20)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('weight', fontsize=15)
    plt.grid()
    plt.plot(epoch, sr_weight, linewidth=3, label='$\sigma_s$', color='#00b33c')
    plt.plot(epoch, rec_weight, linewidth=3, label='$\sigma_r$', color='#13005A')
    plt.legend(fontsize=20)
    plt.savefig(weight_file)

    plt.figure(figsize=(9, 6))
    plt.title('Loss Weights', fontsize=20)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('weight', fontsize=15)
    # plt.xlim(1, 50)
    # plt.ylim(0.7, 1.2)
    plt.grid()
    plt.plot(epoch, sr_weight, linewidth=3, label='$\sigma_s$', color='#00b33c')
    plt.plot(epoch, rec_weight, linewidth=3, label='$\sigma_r$', color='#13005A')
    plt.plot(epoch, sr_weight/rec_weight, linestyle='--', linewidth=3, label='$\sigma_s/\sigma_r$', color='#595959')
    plt.legend(fontsize=20)
    plt.savefig(weight_file2)

    plt.figure(figsize=(9, 6))
    plt.title('Loss Weights Ratio', fontsize=20)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('ratio', fontsize=15)
    plt.grid()
    plt.plot(epoch, sr_weight/rec_weight, linewidth=3, label='weights ratio', color='#00b33c')
    plt.legend(fontsize=20)
    plt.savefig(ratio_file)


if __name__ == '__main__':
    # draw_train(f'pretrain_weights/best_model2', 'png')
    # draw_train(f'pretrain_weights/best_model3', 'png')
    # draw_train(f'pretrain_weights/best_model4', 'png')
    # draw_train(f'pretrain_weights/best_model5', 'png')

    # draw_eval(f'pretrain_weights/best_model2', 'png')
    # draw_eval(f'pretrain_weights/best_model3', 'png')
    # draw_eval(f'pretrain_weights/best_model4', 'png')
    # draw_eval(f'pretrain_weights/best_model5', 'png')

    # draw_train(f'pretrain_weights/uncertainty_best_model1', 'png')
    # draw_eval(f'pretrain_weights/uncertainty_best_model1', 'png')

    draw_train('expr/weighted_loss_sidernn/stat', 'png', 'txt', [0, 50])
    draw_train('expr/weighted_loss_sidernn/stat', 'svg', 'txt', [0, 50])
