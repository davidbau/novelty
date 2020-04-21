import torch, numpy, os, shutil, math, re, torchvision, argparse
from netdissect import parallelfolder, renormalize, pbar
from torchvision import transforms
from torch.optim import Optimizer

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--dataset', choices=['imagenet'], default='imagenet')
    aa('--selected_classes', type=int, default=500)
    aa('--training_iterations', type=int, default=100001)
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    experiment_dir = 'results/baseline-%d-%s-resnet' % (
            args.selected_classes, args.dataset)
    training_dir = 'datasets/%s/train' % args.dataset
    val_dir = 'datasets/%s/val' % args.dataset
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'args.txt'), 'w') as f:
        f.write(str(args) + '\n')
    def printstat(s):
        with open(os.path.join(experiment_dir, 'log.txt'), 'a') as f:
            f.write(str(s) + '\n')
        pbar.print(s)
    def filter_tuple(item):
        return item[1] < args.selected_classes
    # Here's our data
    train_loader = torch.utils.data.DataLoader(
        parallelfolder.ParallelImageFolders([training_dir],
            classification=True,
            filter_tuples=filter_tuple,
            transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        renormalize.NORMALIZER['imagenet'],
                        ])),
        batch_size=128, shuffle=True,
        num_workers=48, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        parallelfolder.ParallelImageFolders([val_dir],
            classification=True,
            filter_tuples=filter_tuple,
            transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        renormalize.NORMALIZER['imagenet'],
                        ])),
        batch_size=128, shuffle=False,
        num_workers=24, pin_memory=True)
    late_model = torchvision.models.resnet50(num_classes=args.selected_classes)
    for n, p in late_model.named_parameters():
        if 'bias' in n:
            torch.nn.init.zeros_(p)
        elif len(p.shape) <= 1:
            torch.nn.init.ones_(p)
        else:
            torch.nn.init.kaiming_normal_(p, nonlinearity='relu')
    late_model.train()
    late_model.cuda()

    model = late_model

    max_lr = 1e-2
    max_iter = args.training_iterations
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr,
            total_steps=max_iter - 1)
    iter_num = 0
    best = dict(val_accuracy=0.0)
    # Oh, hold on.  Let's actually resume training if we already have a model.
    checkpoint_filename = 'weights.pth'
    best_filename = 'best_%s' % checkpoint_filename
    best_checkpoint = os.path.join(experiment_dir, best_filename)
    try_to_resume_training = False
    if try_to_resume_training and os.path.exists(best_checkpoint):
        checkpoint = torch.load(os.path.join(experiment_dir, best_filename))
        iter_num = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best['val_accuracy'] = checkpoint['accuracy']

    def save_checkpoint(state, is_best):
        filename = os.path.join(experiment_dir, checkpoint_filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename,
                    os.path.join(experiment_dir, best_filename))

    def validate_and_checkpoint():
        model.eval()
        val_loss, val_acc = AverageMeter(), AverageMeter()
        for input, target in pbar(val_loader):
            # Load data
            input_var, target_var = [d.cuda() for d in [input, target]]
            # Evaluate model
            with torch.no_grad():
                output = model(input_var)
                loss = criterion(output, target_var)
                _, pred = output.max(1)
                accuracy = (target_var.eq(pred)
                        ).data.float().sum().item() / input.size(0)
            val_loss.update(loss.data.item(), input.size(0))
            val_acc.update(accuracy, input.size(0))
            # Check accuracy
            pbar.post(l=val_loss.avg, a=val_acc.avg)
        # Save checkpoint
        save_checkpoint({
            'iter': iter_num,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'accuracy': val_acc.avg,
            'loss': val_loss.avg,
        }, val_acc.avg > best['val_accuracy'])
        best['val_accuracy'] = max(val_acc.avg, best['val_accuracy'])
        printstat('Iteration %d val accuracy %.2f' %
                (iter_num, val_acc.avg * 100.0))

    # Here is our training loop.
    while iter_num < max_iter:
        for filtered_input, filtered_target in pbar(train_loader):
            # Track the average training loss/accuracy for each epoch.
            train_loss, train_acc = AverageMeter(), AverageMeter()
            # Load data
            input_var, target_var = [d.cuda()
                    for d in [filtered_input, filtered_target]]
            # Evaluate model
            output = model(input_var)
            loss = criterion(output, target_var)
            train_loss.update(loss.data.item(), filtered_input.size(0))
            # Perform one step of SGD
            if iter_num > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Learning rate schedule
                scheduler.step()
            # Also check training set accuracy
            _, pred = output.max(1)
            accuracy = (target_var.eq(pred)).data.float().sum().item() / (
                    filtered_input.size(0))
            train_acc.update(accuracy)
            remaining = 1 - iter_num / float(max_iter)
            pbar.post(l=train_loss.avg, a=train_acc.avg,
                    v=best['val_accuracy'])
            # Ocassionally check validation set accuracy and checkpoint
            if iter_num % 100 == 0:
                validate_and_checkpoint()
                model.train()
            # Advance
            iter_num += 1
            if iter_num >= max_iter:
                break

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
