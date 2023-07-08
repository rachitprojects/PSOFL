# Temp


import pickle
from vgg import *
from config import *
import torch

def get_model(location, model_name, layer, device, cfg):
        cfg = cfg.copy()
        net = VGG(location, model_name, layer, cfg)
        net = net.to(device)
#        logger.debug(str(net))
        return net

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_model("Unit", "VGG5", 6, device, model_cfg)
with open("temp.pickle", 'wb') as cnn:
	pickle.dump(model, cnn)

