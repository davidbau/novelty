import argparse, os, json, numpy, PIL.Image, torch, torchvision, collections
import math, shutil, shapebias.shapebias_models
from . import pidfile, setting, tally, nethook, parallelfolder
from . import upsample, imgviz, imgsave, renormalize, bargraph
from . import runningstats

def load_network(arch, data):
    if data == 'stylized-imagenet':
        if arch == 'vgg16':
            modelname = 'vgg16_trained_on_SIN'
        elif arch == 'resnet':
            modelname = 'resnet50_trained_on_SIN'
        else:
            assert False
        net = shapebias.shapebias_models.load_model(modelname)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        if arch == 'vgg16' and isinstance(net.features, torch.nn.DataParallel):
            net.features = net.features.module
    elif data == 'hybrid-imagenet':
        if arch == 'resnet':
            modelname = 'resnet50_trained_on_SIN_and_IN'
        else:
            assert False
        net = shapebias.shapebias_models.load_model(modelname)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
    elif data == 'finetuned-imagenet':
        if arch == 'resnet':
            modelname = 'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN'
        else:
            assert False
        net = shapebias.shapebias_models.load_model(modelname)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module

    elif data == 'imagenet':
        if arch == 'vgg16':
            net = torchvision.models.vgg16(pretrained=True)
        elif arch == 'resnet':
            net = torchvision.models.resnet50(pretrained=True)
        else:
            assert False
    elif data == 'places':
        if arch == 'vgg16':
            net = setting.load_vgg16(data)
        else:
            assert False
    else:
        fn = 'backup/%s/%s/best_weights.pth' % (arch, data)
        factory = dict(resnet=torchvision.models.resnet50,
                vgg16=torchvision.models.vgg16)[arch]
        net = factory(num_classes=365)
        net.load_state_dict(torch.load(fn)['state_dict'])
    return net

def main():
    parser = argparse.ArgumentParser(description='quickdissect')
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--data', type=str, default='places')
    parser.add_argument('--layer', type=str, default='layer4')
    parser.add_argument('--seg', type=str, default='netpqc')
    parser.add_argument('--sample_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=48)
    args = parser.parse_args()

    resfn = pidfile.exclusive_dirfn(
        args.outdir, args.arch, args.data, args.layer,
        args.seg, args.sample_size)

    import torch
    torch.backends.cudnn.profile = True

    model = nethook.InstrumentedModel(load_network(args.arch, args.data)).cuda()
    model.retain_layer(args.layer)

    testdata = args.data
    if testdata.startswith('stylized-'):
        testdata = testdata[9:]
    if testdata.startswith('hybrid-'):
        testdata = testdata[7:]
    if testdata.startswith('finetuned-'):
        testdata = testdata[10:]

    ds = parallelfolder.ParallelImageFolders(['dataset/%s/val' % testdata],
            classification=False,
            shuffle=True,
            size=args.sample_size,
            transform=torchvision.transforms.Compose([
                        torchvision.transforms.Resize(256),
                        # transforms.CenterCrop(224),
                        torchvision.transforms.CenterCrop(256),
                        torchvision.transforms.ToTensor(),
                        renormalize.NORMALIZER['imagenet'],
                        ]))

    model(ds[0][0][None].cuda())
    sample_act = model.retained_layer(args.layer).cpu()
    upfn = upsample.upsampler((64, 64), sample_act.shape[2:])

    def flat_acts(zbatch):
        _ = model(zbatch.cuda())
        acts = upfn(model.retained_layer(args.layer))
        return acts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])

    rq = tally.tally_quantile(flat_acts, ds, cachefile=resfn('rq.npz'),
            batch_size=args.batch_size)
    level_at_cutoff = rq.quantiles(0.99)[None,:,None,None].cuda()

    segmodel, seglabels, segcats = setting.load_segmenter(args.seg)

    def compute_cond_indicator(zbatch):
        image_batch = zbatch.cuda()
        model(image_batch)
        seg = segmodel.segment_batch(image_batch, downsample=4)
        acts = upfn(model.retained_layer(args.layer))
        iacts = (acts > level_at_cutoff).float()
        return tally.conditional_samples(iacts, seg)

    cmv = tally.tally_conditional_quantile(compute_cond_indicator, ds,
            cachefile=resfn('cmv.npz'), pin_memory=True,
            batch_size=args.batch_size)

    iou_table = tally.iou_from_conditional_indicator_mean(cmv).permute(1, 0)
    numpy.save(resfn('iou.npy'), iou_table.numpy())

    unit_list = enumerate(zip(*iou_table.max(1)))
    unit_records = {
        'units': [ {
            'unit': unit,
            'iou': iou.item() if not math.isnan(iou.item()) else None,
            'label': seglabels[segc],
            'cat': segcats[segc],
            'cls': segc.item()
        } for unit, (iou, segc) in unit_list],
        'header': {
            'name': resfn(''),
            'image': 'bargraph.svg',
        },
    }
    with open(resfn('labels.json'), 'w') as f:
        json.dump(unit_records, f)
    with open(resfn('seglabels.json'), 'w') as f:
        json.dump(seglabels, f)
    with open(resfn('segcats.json'), 'w') as f:
        json.dump(segcats, f)

    def compute_image_max(zbatch):
        model(zbatch.cuda())
        return model.retained_layer(args.layer).max(3)[0].max(2)[0]

    topk = tally.tally_topk(compute_image_max, ds,
            cachefile=resfn('topk.npz'),
            batch_size=args.batch_size)

    def compute_acts(image_batch):
        model(image_batch.cuda())
        acts_batch = model.retained_layer(args.layer)
        return (acts_batch, image_batch)

    iv = imgviz.ImageVisualizer(128, quantiles=rq, source=ds)
    unit_images = iv.masked_images_for_topk(compute_acts, ds, topk, k=5)
    imgsave.save_image_set(unit_images, resfn('imgs/unit_%d.png'))

    shutil.copy('netdissect/report.html', resfn('report.html'))

    dv = DissectVis(args.outdir, args.arch + '/' + args.data,
             layers=[args.layer], seg=args.seg, sample_size=args.sample_size)
    dv.save_bargraph(resfn('bargraph.svg'), args.layer, min_iou=0.04)

    pidfile.mark_job_done(resfn.dir)

class DissectVis:
    '''
    Code to read out the dissection computed in the program above.
    '''
    def __init__(self, outdir='results', model='church', layers=None,
            seg='netpqc', sample_size=1000):
        if not layers:
            layers = ['layer%d' % i for i in range(1, 15)]

        basedir = 'results/church'
        setting = 'netpqc/1000'
        labels = {}
        iou = {}
        images = {}
        rq = {}
        dirs = {}
        for k in layers:
            dirname = os.path.join(outdir, model, k, seg, str(sample_size))
            dirs[k] = dirname
            with open(os.path.join(dirname, 'labels.json')) as f:
                labels[k] = json.load(f)['units']
            iou[k] = numpy.load(os.path.join(dirname, 'iou.npy'))
            images[k] = [None] * len(iou[k])
            rq[k] = runningstats.RunningQuantile(
                    state=numpy.load(os.path.join(dirname, 'rq.npz'),
                        allow_pickle=True))
        with open(os.path.join(dirname, 'seglabels.json')) as f:
            self.seglabels = json.load(f)
        self.dirs = dirs
        self.labels = labels
        self.ioutable = iou
        self.rqtable = rq
        self.images = images
        self.basedir = os.path.join(outdir, model)
        self.setting = os.path.join(seg, str(sample_size))
        
    def label(self, layer, unit):
        return self.labels[layer][unit]['label']
    def iou(self, layer, unit):
        return self.labels[layer][unit]['iou'] or 0.0
    def dir(self, layer):
        return self.dirs[layer]
    def rq(self, layer):
        return self.rqtable[layer]
    def top_units(self, layer, seglabel, k=20):
        return self.ioutable[layer][:,self.seglabels.index(seglabel)
                ].argsort()[::-1][:k].tolist()
    def image(self, layer, unit):
        result = self.images[layer][unit]
        # Lazy loading of images.
        if result is None:
            result = PIL.Image.open(os.path.join(
                self.basedir, layer,
                self.setting, 'imgs/unit_%d.png' % unit))
            result.load()
            self.images[layer][unit] = result
        return result

    def save_bargraph(self, filename, layer, min_iou=0.04):
        svg = self.bargraph(layer, min_iou=min_iou, file_header=True)
        with open(filename, 'w') as f:
            f.write(svg)

    def img_bargraph(self, layer, min_iou=0.04):
        url = self.bargraph(layer, min_iou=min_iou, data_url=True)
        class H:
            def __init__(self, url):
                self.url = url
            def _repr_html_(self):
                return '<img src="%s">' % self.url
        return H(url)

    def bargraph(self, layer, min_iou=0.04, **kwargs):
        labelcat_list = []
        for rec in self.labels[layer]:
            if rec['iou'] and rec['iou'] >= min_iou:
                labelcat_list.append(tuple(rec['cat']))
        return self.bargraph_from_conceptcatlist(labelcat_list, **kwargs)

    def bargraph_from_conceptcatlist(self, conceptcatlist, **kwargs):
        count = collections.defaultdict(int)
        catcount = collections.defaultdict(int)
        for c in conceptcatlist:
            count[c] += 1
        for c in count.keys():
            catcount[c[1]] += 1
        cats = ['object', 'part', 'material', 'texture', 'color']
        catorder = dict((c, i) for i, c in enumerate(cats))
        sorted_labels = sorted(count.keys(),
            key=lambda x: (catorder[x[1]], -count[x]))
        sorted_labels
        return bargraph.make_svg_bargraph(
            [label for label, cat in sorted_labels],
            [count[k] for k in sorted_labels],
            [(c, catcount[c]) for c in cats], **kwargs)

if __name__ == '__main__':
    main()

