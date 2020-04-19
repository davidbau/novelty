import os, random, sys, torch
import matplotlib.pyplot as plt
from cmc.models.resnet import InsResNet50
from cmc.models.LinearModel import LinearClassifierResNet
from cmc.quickdataset import QuickImageFolder
