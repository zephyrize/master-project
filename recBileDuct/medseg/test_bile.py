import os
from os.path import join
import pandas as pd
import sys
import argparse
sys.path.append('../')

from medseg.models.advanced_triplet_recon_segmentation_model import AdvancedTripletReconSegmentationModel
from medseg.models.advanced_triplet_recon_segmentation_model_2 import AdvancedTripletReconSegmentationModel_2
from medseg.models.advanced_triplet_recon_segmentation_model_3 import AdvancedTripletReconSegmentationModel_3
from medseg.common_utils.basic_operations import check_dir
from medseg.dataset_loader.BileDuct_dataset_v3 import BileDuctDataset
from medseg.test_basic_segmentation_solver import TestSegmentationNetwork


sample_slices = 3
def evaluate(method_name, segmentation_model, test_dataset_name, frames=['bile'], metrics_list=['Dice'],
             save_report_dir=None,
             save_predict=False, save_soft_prediction=False, foreground_only=False):
    n_iter = segmentation_model.n_iter
    # evaluation settings
    # save_path = checkpoint_dir.replace(
    #     'checkpoints', f'report/{test_dataset_name}')
    # check_dir(save_path, create=True)

    summary_report_file_name = 'iter_{}_summary.csv'.format(n_iter)
    detailed_report_file_name = 'iter_{}_detailed.csv'.format(n_iter)
    # test_dataset = get_testset(test_dataset_name, frames=frames)
    test_dataset = BileDuctDataset(split='test', sample_slices=sample_slices)
    tester = TestSegmentationNetwork(test_dataset=test_dataset,
                                     crop_size=None, segmentation_model=segmentation_model, use_gpu=True,
                                     save_path=save_report_dir, summary_report_file_name=summary_report_file_name,
                                     detailed_report_file_name=detailed_report_file_name, patient_wise=True, metrics_list=metrics_list,
                                     foreground_only=foreground_only,
                                     save_prediction=save_predict, save_soft_prediction=save_soft_prediction)

    tester.run()

    print('<Summary> {} on dataset {} across {}'.format(
        method_name, test_dataset_name, str(frames)))
    print(tester.df.describe())
    # save each method's result summary/details on each test dataset
    tester.df.describe().to_csv(join(save_report_dir + f'/{test_dataset_name}' + '{}_iter_{}_summary.csv'.format(
        str(frames), str(n_iter))))
    tester.df.to_csv(join(save_report_dir + f'/{test_dataset_name}' + '{}_iter_{}_detailed.csv'.format(
        str(frames), str(n_iter))))

    means = [round(v, 4) for k, v in tester.df.mean(axis=0).items()]
    stds = [round(v, 4) for k, v in tester.df.std(axis=0).items()]
    return means, stds, tester.df



if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description='test bileduct')
    parser.add_argument("--cval", type=int, default=0,
                        help="cross validation subset")

    parser.add_argument("--epoch", type=str, default="best",
                        help="cross validation subset")
    
    parser.add_argument("--data_setting", type=str, default="10",
                        help="data_setting:['keep-origin-false','keep-origin-true']")
    
    parser.add_argument("--type", type=str, default="standard_training",
                        help="type:['standard_tCUDA_VISIBLE_DEVICES=2 python -W ignore test_bile.py --data_setting keep-origin-false --type standard_training --cval 2raining','cooperative_training']")
    
    test_opt = parser.parse_args()

    use_gpu = True
    # model config
    num_classes = 2
    network_type = 'TAGNet' # FCN_16_standard, TAGNet
    n_iter = 2  # 1 for FTN's prediction, 2 for FTN+STN's refinements

    test_dataset_name_list = ['BileDuct']

    segmentor_resume_dir_dict = {
            'standard_training': f'./saved/train_BileDuct_' + test_opt.data_setting + \
                f'_n_cls_2/BileDuct/standard_training/{test_opt.cval}/model/' + test_opt.epoch + '/checkpoints',
            'cooperative_training': f'./saved/train_BileDuct_' + test_opt.data_setting + \
                f'_n_cls_2/BileDuct/cooperative_training/{test_opt.cval}/model/' + test_opt.epoch + '/checkpoints',
         
    }

    # load model
    model_dict = {}
    for method, checkpoint_dir in segmentor_resume_dir_dict.items():
        if method != test_opt.type: # standard_training , cooperative_training
            continue
        
        print(method, checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            print(f'{method}:{checkpoint_dir} not found. ')
            continue
        model_dict[method] = AdvancedTripletReconSegmentationModel_3(network_type=network_type, image_ch=sample_slices, decoder_dropout=None,
                                                                   checkpoint_dir=checkpoint_dir,
                                                                   num_classes=num_classes, n_iter=n_iter, use_gpu=True)

    df_dict = {}
    for test_dataset_name in test_dataset_name_list:
        result_summary = []
        for method_name, model in model_dict.items():
            save_report_dir = join(
                segmentor_resume_dir_dict[method_name], 'report')
            check_dir(save_report_dir, create=True)
            means, stds, concatenated_df = evaluate(
                segmentation_model=model, test_dataset_name=test_dataset_name, method_name=method_name, save_report_dir=save_report_dir,
                foreground_only = True, save_predict=True, save_soft_prediction=True)
            result_summary.append(
                [test_dataset_name, method_name, means, stds])
            df_dict[method_name] = concatenated_df
        aggregated_df = pd.DataFrame(data=result_summary, columns=[
            'dataset', 'method', 'Dice mean', 'Dice std'])
        print(aggregated_df)




'''
CUDA_VISIBLE_DEVICES=7 python -W ignore test_bile.py --data_setting keep-origin --cval 2 --epoch 239

CUDA_VISIBLE_DEVICES=2 python -W ignore test_bile.py --data_setting keep-origin-false --type standard_training --cval 0

CUDA_VISIBLE_DEVICES=7 python -W ignore test_bile.py --data_setting keep-origin-false --type cooperative_training --cval 4

CUDA_VISIBLE_DEVICES=1 python -W ignore test_bile.py --data_setting keep-origin-false --type cooperative_training --cval 5
'''
