import os
import lightning as L
import argparse
import yaml
import torch
import logging

from utils import *
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from models.loss import MaskCrossEnrtopyLoss, BalanceBCELoss


class LightingModelWrapper(L.LightningModule):
    def __init__(self, args, config):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.config = config

        # define model 
        model_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.py')
        if os.path.exists(model_file_path):
            model_cls, package_path = import_class_from_path(args.model, model_file_path)
        else:
            model_cls, package_path = import_class(args.model)

        self.model = model_cls(args,config)
        self.model_file_path = package_path
        print(f'Load model file from {package_path}.')

        # define loss
        self.ce_loss_fn = MaskCrossEnrtopyLoss()
        self.bce_loss_fn = BalanceBCELoss()

        self.train_pred_labels = []
        self.train_step_outputs = []
        self.test_pred_labels = []
        self.test_step_outputs = []
        self.validate_pred_labels = []
        self.validate_step_outputs = []
        self.utt_id_list = []
        self.b_train_pred_labels = []
        self.b_train_step_outputs = []
        self.b_validate_pred_labels = []
        self.b_validate_step_outputs = []
        self.b_test_pred_labels = []
        self.b_test_step_outputs = []

    def setup(self, stage: str) -> None:
        # save runing model and train python script
        save_running_script(os.path.abspath(__file__),f'{trainer.logger.root_dir}/version_{trainer.logger.version}/run.py')
        save_running_script(os.path.abspath(self.model_file_path),f'{trainer.logger.root_dir}/version_{trainer.logger.version}/model.py')

        # console log configuration
        if self.local_rank == 0:
            self.console_logger = logging.getLogger(f'lightning.pytorch.{stage}')
            file_handler = logging.FileHandler(f'{trainer.logger.root_dir}/version_{trainer.logger.version}/{stage}.log')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
            self.console_logger.addHandler(file_handler)
            self.console_logger.info(f'Start training.')

    def training_step(self, batch, batch_idx):
        utt_id, input, ori_label, boundary_label, ori_label_length, boundary_length = batch
        output, boundary = self.model(input)

        # compute loss
        spoof_mask = get_src_mask(ori_label, ori_label_length)
        boundary_mask = get_src_mask(boundary_label, boundary_length)
        boundary_loss = self.bce_loss_fn(boundary, boundary_label, mask=boundary_mask)
        spoof_loss = self.ce_loss_fn(output.transpose(-1,-2), ori_label.to(torch.long), mask=spoof_mask)
        total_loss = spoof_loss + 0.5*boundary_loss

        # record loss
        self.log('train_loss', total_loss.item(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('spoof_loss', spoof_loss.item())
        self.log('boundary_loss', boundary_loss.item())

        pred_src, label_src = cut_according_length(output.detach(), ori_label, ori_label_length)  # remove padding part
        b_pred_src, b_label_src = cut_according_length(boundary.detach(), boundary_label, boundary_length)
        self.train_step_outputs.extend(pred_src)
        self.train_pred_labels.extend(label_src)
        self.b_train_step_outputs.extend(b_pred_src)
        self.b_train_pred_labels.extend(b_label_src)
        self.utt_id_list.extend(utt_id)
        return total_loss

    def configure_optimizers(self):
        optimizer= torch.optim.Adam(self.model.parameters(),lr=self.args.base_lr, betas=(0.9,0.999), \
                                    weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler':scheduler} 

    def evaluation_run_model(self, batch, type):
        outputs_collector = getattr(self, f'{type}_step_outputs')
        labels_clooector = getattr(self, f'{type}_pred_labels')
        b_outputs_collector = getattr(self, f'b_{type}_step_outputs')
        b_labels_collector = getattr(self, f'b_{type}_pred_labels')

        utt_id, input, ori_label, boundary_label, ori_label_length, boundary_length = batch

        # Make the data length an integer multiple of the resolution
        scale = int(self.args.resolution * self.args.samplerate)
        input = torch.nn.functional.pad(input, (0,ori_label.size(1)*scale-input.size(1)))
        output, binary = self.model(input)

        # computre validate loss
        if type == 'validate':
            validate_loss = self.ce_loss_fn(output.transpose(-1,-2), ori_label.to(torch.long), mask=None)
            self.log('validate_loss', validate_loss.item(), sync_dist=True)

        outputs_collector.extend(output.tolist())
        labels_clooector.extend(ori_label.tolist())
        b_outputs_collector.extend(binary.tolist())
        b_labels_collector.extend(boundary_label.tolist())
        self.utt_id_list.extend(utt_id)

    def evaluation_on_epoch_end(self, type):
        outputs = getattr(self, f'{type}_step_outputs')
        labels = getattr(self, f'{type}_pred_labels')
        b_outputs = getattr(self, f'b_{type}_step_outputs')
        b_labels = getattr(self, f'b_{type}_pred_labels')
        utt_ids = self.utt_id_list

        # frame level
        frame_preds = torch.tensor([i for utt in outputs for i in utt]).detach().cpu()
        frame_labels = torch.tensor([i for utt in labels for i in utt]).detach().cpu()
        eer, threshold = compute_eer(frame_preds[:,1],frame_labels)
        accuracy, precision, recall, fbeta_score = computer_precision_recall_fscore(frame_preds.argmax(dim=-1), frame_labels)

        self.log(f'{type}/{type}_F1', fbeta_score, sync_dist=True)
        self.log(f'{type}/{type}_eer', eer, sync_dist=True)
        self.log(f'{type}/{type}_acc', accuracy, sync_dist=True)

        if self.local_rank == 0:
            self.console_logger.info(f'Epoch [{self.current_epoch}]: {type} eer {eer}')
            self.console_logger.info(f'Epoch [{self.current_epoch}]: {type} acc {accuracy}')
            self.console_logger.info(f'Epoch [{self.current_epoch}]: {type} precision {precision}')
            self.console_logger.info(f'Epoch [{self.current_epoch}]: {type} recall {recall}')
            self.console_logger.info(f'Epoch [{self.current_epoch}]: {type} F1 {fbeta_score}')
            self.console_logger.info('---------------------------------------------------------')

        if self.args.test_only:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_result.txt'),'a') as result_file:
                result_file.write(f'Checkpoint :{self.args.checkpoint} \n')
                result_file.write(f'EER :{eer*100}% \n')
                result_file.write(f'F1 :{fbeta_score} \n')
                result_file.write(f'Precision :{precision} \n')
                result_file.write(f'Recall :{recall} \n')
                result_file.write(f'Test log :{trainer.logger.root_dir}/version_{trainer.logger.version} \n')
                result_file.write(f'\n')

        # binary 
        b_frame_preds = torch.tensor([i for utt in b_outputs for i in utt])
        b_frame_labels = torch.tensor([i for utt in b_labels for i in utt])
        eer, threshold = compute_eer(b_frame_preds,b_frame_labels)
        accuracy, precision, recall, fbeta_score = computer_precision_recall_fscore(torch.where(b_frame_preds>0.5,1,0), b_frame_labels)

        self.log(f'{type}/b_{type}_F1', fbeta_score, sync_dist=True)
        self.log(f'{type}/b_{type}_eer', eer, sync_dist=True)
        self.log(f'{type}/b_{type}_acc', accuracy, sync_dist=True)

        if self.local_rank == 0:
            self.console_logger.info(f'Epoch [{self.current_epoch}]: binary {type} eer {eer}')
            self.console_logger.info(f'Epoch [{self.current_epoch}]: binary {type} acc {accuracy}')
            self.console_logger.info(f'Epoch [{self.current_epoch}]: binary {type} precision {precision}')
            self.console_logger.info(f'Epoch [{self.current_epoch}]: binary {type} recall {recall}')
            self.console_logger.info(f'Epoch [{self.current_epoch}]: binary {type} F1 {fbeta_score}')
            self.console_logger.info('---------------------------------------------------------')

        outputs.clear()
        labels.clear()
        b_outputs.clear()
        b_labels.clear()
        utt_ids.clear()

    def validation_step(self, batch, batch_idx):
        self.evaluation_run_model(batch, type='validate')

    def test_step(self, batch, batch_idx):
        self.evaluation_run_model(batch, type='test')

    def on_validation_epoch_end(self):
        self.evaluation_on_epoch_end(type='validate')

    def on_test_epoch_end(self):
        self.evaluation_on_epoch_end(type='test')

    def on_train_epoch_end(self):
        self.evaluation_on_epoch_end(type='train')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=str, help="dataset module", default='dataset.partialspoof.PartialSpoofDataModule')
    parser.add_argument('--model', type=str, help="model module", default='models.bam.BAM')

    parser.add_argument('--train_root', type=str, help="train data path, all query file in this folder will as a train sample.", 
                    default='data/raw/train')
    parser.add_argument('--dev_root', type=str, help="validate data path, all query file in this folder will as a validate sample.", 
                    default='data/raw/dev')
    parser.add_argument('--eval_root', type=str, help="test data path, all query file in this folder will as a test sample.", 
                    default='data/raw/eval')
    parser.add_argument('--label_root', type=str, default='./data', help="segment label path")


    # training configuration
    parser.add_argument('--max_epochs', type=int, default=50, help='max train epoch.')
    parser.add_argument('--batch_size', type=int, default=8, help='train dataloader batch size.')
    parser.add_argument('--num_workers', type=int, default=8, help='train dataloader of num workers')
    parser.add_argument('--base_lr', type=int, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=int, default=1e-4, help='weight_decay')
    parser.add_argument('--samplerate', type=int, default=16000, help="samplerate")
    parser.add_argument('--resolution', type=int, default=0.16, help="segment label reoslution.")
    parser.add_argument('--input_maxlength', type=int,  default=None, help="unuselesss")
    parser.add_argument('--input_minlength', type=int,  default=None, help="min length of label or audio")
    parser.add_argument('--label_maxlength', type=int,  default=25, help="max length of label or audio")
    parser.add_argument('--pad_mode', type=str,  default='label', help='how to pad data')
    
    parser.add_argument('--gpu', type=list, default=[0], help="gpu index")
    parser.add_argument('--test_only', action='store_true' ,help="test model")
    parser.add_argument('--exp_name', type=str, default='bam_wavlm', help="experiment name.")
    parser.add_argument('--validate_interval', type=int, default=3, help="do validate epoch number")

    # model configuration
    parser.add_argument('--checkpoint', type=str, help='model checkpoint',
                    default='exp/bam_wavlm/train/lightning_logs/version_6/checkpoints/30-0.05991.ckpt')
    parser.add_argument('--continue_train', type=bool, default=False, help='continue training')

    args = parser.parse_args()
    L.seed_everything(42, workers=True)
    # model define 
    with open(f'config/{args.exp_name}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = Attribution_Config(**config)

    if hasattr(args, 'checkpoint') and (args.test_only or args.continue_train):
        args.checkpoint = os.readlink(args.checkpoint) if os.path.islink(args.checkpoint) else args.checkpoint
        model = LightingModelWrapper.load_from_checkpoint(args.checkpoint, map_location='cpu', args=args)
        print(f'Load model from {args.checkpoint}.')
    else:
        model = LightingModelWrapper(args, config)
        print(f'Train model from scratch.')

    # define dataset 
    dataset_cls, _ = import_class(args.dataset)
    Lightning_dataset = dataset_cls(args)

    checkponint_callback = ModelCheckpoint(
        filename='{epoch}-{validate_loss:.5f}',
        every_n_epochs=1,
        save_top_k=-1,
        # monitor='validate/validate_eer',
        save_weights_only=True,
        enable_version_counter=True,
        auto_insert_metric_name=False,
        )

    # start training
    trainer = L.Trainer(
        accelerator='gpu',
        devices=args.gpu,
        max_epochs=args.max_epochs,
        strategy='auto' if len(args.gpu)==1 else 'ddp_find_unused_parameters_true',
        logger=TensorBoardLogger(save_dir=f'exp/{args.exp_name}/test' if args.test_only else f'exp/{args.exp_name}/train'),
        check_val_every_n_epoch=args.validate_interval,
        callbacks=checkponint_callback,
        )

    if not args.test_only:
        trainer.fit(model=model, datamodule=Lightning_dataset)
        print('Train finish.')

    else:
        trainer.test(model=model, datamodule=Lightning_dataset)
        print('Test finish.')
