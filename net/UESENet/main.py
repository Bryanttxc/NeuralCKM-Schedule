import os
import sys
import argparse
from utils.entry import vit_entry, cdf_entry, mlp_entry, lstm_entry
from utils.config import Config


def start_train(model, model_path, loss_path, num_fold = None):
    model.train(model_path, loss_path, num_fold)


def start_test(model, test_data_path, model_path, is_plot=False):
    model.test(test_data_path, model_path, is_plot=is_plot)


def start_cal(model, cal_in_name, cal_out_name, infile_type, model_path, config):
    for numUE_idx in range(len(config.numUE_list)):
        numUE = config.numUE_list[numUE_idx]
        for lab in range(1, config.lab+1):
            for sub_lab in range(1, config.sublab+1):
                cal_data_path = os.path.join(config.data_dir, 'cal', 'input', f'{cal_in_name}_lab_{lab}_{sub_lab}_numUE_{numUE}.{infile_type}')
                if os.path.exists(cal_data_path):
                    save_path = os.path.join(config.data_dir, 'cal', 'output', f'{cal_out_name}_lab_{lab}_{sub_lab}_numUE_{numUE}')
                    model.cal(cal_data_path, model_path, save_path)
                else:
                    break


def start_program(config):

    if config.vit_start:

        train_data_path = os.path.join(config.data_dir, 'train', 'vit', 'train_data.xlsx')
        test_data_path = os.path.join(config.data_dir, 'test', 'vit', 'test_data.xlsx')
        # test_data_path = os.path.join(config.data_dir, 'others', 'sys_test_data.xlsx')
        vit_model_path = os.path.join(config.model_dir, config.model_name)
        vit_loss_path = os.path.join(config.data_dir, 'others', 'vit_loss.json')
        vit_model = vit_entry(train_data_path, config.num_vit_feat, config.num_vit_sample, 
                    config.num_vit_output, config.batch_size, config.num_epoch, config.num_fold,
                    config.loss_delta)

        if config.train_start:
            start_train(vit_model, vit_model_path, vit_loss_path, num_fold=config.num_fold)
        elif config.test_start:
            start_test(vit_model, test_data_path, vit_model_path, is_plot=config.plot_cdf)
        elif config.cal_start:
            start_cal(vit_model, 'cal_data', 'cdf_data', 'xlsx', vit_model_path, config)

    elif config.cdf_start:

        train_data_path = []
        test_data_path = []
        for case_idx in range(1, config.num_cdf_case+1):

            train_path = os.path.join(config.data_dir, 'train', 'cdf', f'train_SE_data_{case_idx}.mat')
            if os.path.exists(train_path):
                train_data_path.append(train_path)

            test_path = os.path.join(config.data_dir, 'test', 'cdf', f'test_SE_data_{case_idx}.mat')
            if os.path.exists(test_path):
                test_data_path.append(test_path)

        cdf_model_path = os.path.join(config.model_dir, config.model_name)
        cdf_loss_path = os.path.join(config.data_dir, 'others', 'cdf_loss.json')
        cdf_model = cdf_entry(train_data_path, config.num_cdf_case, config.num_cdf_sample, 
                    config.num_cdf_output, config.batch_size, config.num_epoch, config.num_fold)

        if config.train_start:
            start_train(cdf_model, cdf_model_path, cdf_loss_path, num_fold=config.num_fold)
        elif config.test_start:
            start_test(cdf_model, test_data_path, cdf_model_path, is_plot=config.plot_loss)
        elif config.cal_start:
            start_cal(cdf_model, 'cal_SE_data_1', 'SE_data', 'mat', cdf_model_path, config)

    elif config.MLP_start:

        train_data_path = os.path.join(config.data_dir, 'train', 'vit', 'train_data.xlsx')
        test_data_path = os.path.join(config.data_dir, 'test', 'vit', 'test_data.xlsx')
        mlp_model_path = os.path.join(config.model_dir, 'mlp.pth')
        mlp_loss_path = os.path.join(config.data_dir, 'others', 'mlp_loss.json')
        mlp_model = mlp_entry(train_data_path, config.num_vit_feat, config.num_vit_sample, 
                    config.num_vit_output, config.batch_size, config.num_epoch, config.num_fold)

        if config.train_start:
            start_train(mlp_model, mlp_model_path, mlp_loss_path, num_fold=config.num_fold)
        elif config.test_start:
            start_test(mlp_model, test_data_path, mlp_model_path, is_plot=config.plot_cdf)

    elif config.LSTM_start:

        train_data_path = []
        test_data_path = []
        for case_idx in range(1, config.num_cdf_case+1):

            train_path = os.path.join(config.data_dir, 'train', 'cdf', f'train_SE_data_{case_idx}.mat')
            if os.path.exists(train_path):
                train_data_path.append(train_path)

            test_path = os.path.join(config.data_dir, 'test', 'cdf', f'test_SE_data_{case_idx}.mat')
            if os.path.exists(test_path):
                test_data_path.append(test_path)

        lstm_model_path = os.path.join(config.model_dir, 'lstm.pth')  # Path to save the model
        lstm_loss_path = os.path.join(config.data_dir, 'others', 'lstm_loss.json')
        lstm_model = lstm_entry(train_data_path, config.num_cdf_case, config.num_cdf_sample,  
                    config.num_cdf_output, config.batch_size, config.num_epoch, config.num_fold)

        if config.train_start:
            start_train(lstm_model, lstm_model_path, lstm_loss_path, num_fold=config.num_fold)
        elif config.test_start:
            start_test(lstm_model, test_data_path, lstm_model_path)


if __name__ == '__main__':

    SUPPORT_MODEL = ['vit', 'cdf', 'mlp', 'lstm']
    SUPPORT_TASK = ['train', 'test', 'cal']

    # create parser
    parser = argparse.ArgumentParser('Config UESENet')
    parser.add_argument('--model', action='store', choices=SUPPORT_MODEL, dest='model', default='vit', 
                        required=True, type=str, help='model')
    parser.add_argument('--model_name', action='store', dest='model_name', default='vit.pth', 
                        required=True, type=str, help='model name')
    parser.add_argument('--task', action='store', choices=SUPPORT_TASK, dest='task', default='train', 
                        required=True, type=str, help='task')
    parser.add_argument('--num_fold', dest='num_fold', default=5, 
                        type=int, help='Number of fold')
    parser.add_argument('--loss_delta', dest='loss_delta', default=0.5, 
                        type=float, help='parameter of loss_fn')
    
    # for cal
    parser.add_argument('--UE_list', dest='numUE_list', nargs='+', 
                        type=int, help='List of UE')
    parser.add_argument('--num_lab', dest='num_lab', default=5, 
                        type=int, help='Number of labs')
    parser.add_argument('--max_num_sublab', dest='max_num_sublab', default=30, 
                        type=int, help='Max number of sublabs')

    # parse
    args = parser.parse_args()

    config = Config()
    config.numUE_list = args.numUE_list
    config.lab = args.num_lab
    config.sublab = args.max_num_sublab
    config.model_name = args.model_name
    config.num_fold = args.num_fold
    config.loss_delta = args.loss_delta

    # model
    model = args.model
    if model == 'vit':
        config.vit_start = True
    elif model == 'cdf':
        config.cdf_start = True
    elif model == 'mlp':
        config.MLP_start = True
    elif model == 'lstm':
        config.LSTM_start = True
    else:
        print(f"[Error] cannot find the '{model}' model")
        sys.exit(1)

    # task
    task = args.task
    if task == 'train':
        config.train_start = True
    elif task == 'test':
        config.test_start = True
    elif task == 'cal':
        config.cal_start = True
    else:
        print(f"[Error] cannot find the '{task}' task")
        sys.exit(1)

    start_program(config)
