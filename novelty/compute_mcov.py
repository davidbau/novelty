import os, torch, argparse
from cmc.models.resnet import InsResNet50
from torchvision import transforms
from netdissect import tally, runningstats, renormalize, parallelfolder

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--expdir', default='results')
    aa('--dataset', choices=['imagenet', 'places'], default='imagenet')
    aa('--split', choices=['train', 'val'], default='val')
    aa('--layernum', type=int, default=7)
    aa('--selected_classes', type=int, default=None)
    aa('--batch_size', type=int, default=100)
    args = parser.parse_args()
    return args

def main(args):
    dataset = args.dataset
    layernum = args.layernum
    split = args.split
    selected_classes = args.selected_classes
    def ef(s):
        return os.path.join(args.expdir, s)
    model_dir = "/data/vision/torralba/dissect/novelty/models"
    model_name = f"{dataset}_moco_resnet50.pth"
    model_path = os.path.join(model_dir, model_name)
    val_path = f"datasets/{dataset}/val"
    train_path = f"datasets/{dataset}/train"
    img_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        renormalize.NORMALIZER['imagenet']
    ])
    dsv = parallelfolder.ParallelImageFolders([val_path],
         transform=img_trans, classification=True)
    dst = parallelfolder.ParallelImageFolders([train_path],
         transform=img_trans, classification=True)
    if selected_classes is None:
        selected_classes = len(dst.classes) // 2
    dsm = dict(val=dsv, train=dst)
    dp_model = InsResNet50()
    checkpoint = torch.load(model_path)
    dp_model.load_state_dict(checkpoint['model_ema'])
    model = dp_model.encoder.module
    model.cuda()
    def batch_features(imgbatch, cls):
        result = model(imgbatch.cuda(), layer=layernum)
        if len(result.shape) == 4:
            result = result.permute(0, 2, 3, 1).reshape(-1, result.shape[1])
        return result
    mcov = tally.tally_covariance(batch_features, dsm[split],
         num_workers=100, batch_size=args.batch_size, pin_memory=True,
         cachefile=ef(f'{dataset}-{split}-layer{layernum}-mcov.npz'))
    def selclass_features(imgbatch, cls):
        result = model(imgbatch.cuda(), layer=layernum)
        if len(result.shape) == 4:
            cls = cls[:,None,None].expand(result.shape[0],
                    result.shape[2], result.shape[3]).reshape(-1)
            result = result.permute(0, 2, 3, 1).reshape(-1, result.shape[1])
        selected = result[cls < selected_classes]
        return selected
    selcov = tally.tally_covariance(selclass_features, dsm[split],
         num_workers=100, batch_size=args.batch_size, pin_memory=True,
         cachefile=ef(f'{dataset}-{split}-layer{layernum}' +
                      f'-sel{selected_classes}-mcov.npz'))

if __name__ == '__main__':
    main(parseargs())
