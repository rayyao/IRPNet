# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args
import scipy.io as scio
# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader
# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param import  load_dataset, load_param
# Model
from model.IRPNet import  Res_CBAM_block, IRPNet
import matplotlib.pyplot as plt
import os


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, 10)
        self.PD_FA = PD_FA(1, 10)
        self.mIoU  = mIoU(1)
        self.nIoU  = nIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)
        self.img_ids = val_img_ids

        # Choose and load model (this paper is finished by one GPU)
        model       = IRPNet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        model           = model.cuda()
        
        print("Model Initializing")
        self.model      = model

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load trained model
        checkpoint        = torch.load('result/' + args.model_dir)
        
        self.model.load_state_dict(checkpoint['state_dict'])


        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred =preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)

                num += 1

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.nIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)

                ture_positive_rate, false_positive_rate, recall, precision= self.ROC.get()
                best_f1 = np.max(2.0 * precision * recall / (np.spacing(1) + precision + recall))
                _, mean_IOU = self.mIoU.get()
                _, normal_IOU = self.nIoU.get()

            FA, PD = self.PD_FA.get(len(val_img_ids))

            save_result_for_test(dataset_dir, args.st_model,args.epochs, mean_IOU, normal_IOU, PD[0], FA[0]*1000000, best_f1)


def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)

