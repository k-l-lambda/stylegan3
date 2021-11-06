
import os
import torch
from collections import OrderedDict

from torch_utils import persistence
from .networks_stylegan2 import Discriminator as SG2Discriminator
from .schp import networks as schp_networks



SCHP_CLASSES = int(os.environ.get('SCHP_CLASSES', 18))
SCHP_PRETRAINED = os.environ['SCHP_PRETRAINED']


@persistence.persistent_class
class HumanDiscriminator(torch.nn.Module):
	def __init__(self, img_channels, **args):
		super().__init__()

		self.human_parser = schp_networks.init_model('resnet101', num_classes=SCHP_CLASSES, pretrained=None)

		# load schp weights
		state_dict = torch.load(SCHP_PRETRAINED)['state_dict']
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			name = k[7:]  # remove `module.`
			new_state_dict[name] = v
		self.human_parser.load_state_dict(new_state_dict)

		self.human_parser.eval()
		for param in self.human_parser.parameters():
			param.requires_grad = False

		self.sg2 = SG2Discriminator(img_channels=img_channels + SCHP_CLASSES, **args)


	def forward(self, img, c, update_emas=False, **block_kwargs):
		pass
