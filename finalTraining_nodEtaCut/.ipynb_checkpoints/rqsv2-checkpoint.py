from quakSpaceMaker import QuakSpaceMaker
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("-b","--bkg_axis",required=True)
parser.add_argument("-s","--sig_axis",action="append",required=True)
parser.add_argument("-i","--inject",required=True)
parser.add_argument("-x","--xsec")
parser.add_argument("-n","--ninj")
parser.add_argument("-r","--reduction",default="None")
parser.add_argument("-t","--transformation",default="None")
parser.add_argument("-d","--decorrelation",default="linear")
parser.add_argument("-l","--lossdir",default="lossEvals")

args = parser.parse_args()

if args.xsec and args.ninj:
    print("Can't specify xsec and num_injected at the same time!")
    exit()

if not (args.xsec or args.ninj):
    print("Need to specify injection xsec or number of events with -x xsec or -n n_inject")
    exit()

if not os.path.isdir(args.lossdir):
    print(f"No input directory {args.lossdir} exists!")
    exit()

if args.ninj:
    qmaker = QuakSpaceMaker(args.bkg_axis,args.sig_axis,toInject=args.inject,nInject=int(args.ninj),
                            loss_reduction=args.reduction,loss_transform=args.transformation,decorrelation=args.decorrelation,loss_dir=args.lossdir)
else:
    qmaker = QuakSpaceMaker(args.bkg_axis,args.sig_axis,toInject=args.inject,xs_inject=float(args.xsec),
                            loss_reduction=args.reduction,loss_transform=args.transformation,decorrelation=args.decorrelation,loss_dir=args.lossdir)

qmaker.run()
