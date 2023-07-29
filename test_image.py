
import torch
from torch.autograd import Variable
from net import TUFusion_net
import utils
from args_fusion import args
import numpy as np
import os
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, input_nc, output_nc):

	TUFusion_model = TUFusion_net(input_nc, output_nc)
	TUFusion_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in TUFusion_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(TUFusion_model._get_name(), para * type_size / 1000 / 1000))

	TUFusion_model.eval()
	TUFusion_model.to(device)

	return TUFusion_model


def _generate_fusion_image(model, strategy_type, img1, img2, p_type):
	# encoder
	en_r = model.encoder(img1)
	en_v = model.encoder(img2)

	# fusion: hybrid, channel and spatial
	# f = model.fusion(en_r, en_v, p_type)

	# fusion: addition
	# f = model.fusion1(en_r, en_v)

	# fusion: composite attention
	f = model.fusion2(en_r, en_v, p_type)

	# decoder
	img_fusion = model.decoder(f)
	return img_fusion[0]


def run_demo(model, infrared_path, visible_path, output_path_root, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode, p_type):
	# if mode == 'L':
	ir_img = utils.get_test_images(infrared_path, height=None, width=None, mode=mode)
	vis_img = utils.get_test_images(visible_path, height=None, width=None, mode=mode)

	if args.cuda:
		ir_img = ir_img.to(device)
		vis_img = vis_img.to(device)
	ir_img = Variable(ir_img, requires_grad=False)
	vis_img = Variable(vis_img, requires_grad=False)
	dimension = ir_img.size()

	ir_img_resize = F.interpolate(ir_img, size=(256,256), mode='bilinear', align_corners=False)
	vis_img_resize = F.interpolate(vis_img, size=(256,256), mode='bilinear', align_corners=False)

	img_fusion = _generate_fusion_image(model, strategy_type, ir_img_resize, vis_img_resize, p_type)
	img_fusion = F.interpolate(img_fusion, size=(dimension[2], dimension[3]), mode='bilinear', align_corners=False)

	# multi outputs
	file_name = str(index) + '.jpg'
	output_path = output_path_root + file_name
	
	# save images
	if args.cuda:
		img = img_fusion.cpu().clamp(0, 255).data[0].numpy()
	else:
		img = img_fusion.clamp(0, 255).data[0].numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	utils.save_images(output_path, img)

	print(output_path)


def vision_features(feature_maps, img_type):
	count = 0
	for features in feature_maps:
		count += 1
		for index in range(features.size(1)):
			file_name = 'feature_maps_' + img_type + '_level_' + str(count) + '_channel_' + str(index) + '.png'
			output_path = 'outputs/feature_maps/' + file_name
			map = features[:, index, :, :].view(1,1,features.size(2),features.size(3))
			map = map*255
			# save images
			utils.save_image_test(map, output_path)


def main():

	test_path = "images/IV_images/"
	network_type = 'TUfusion'
	strategy_type_list = ['addition', 'attention_weight']

	output_path = './outputs/'
	strategy_type = strategy_type_list[0]
	fusion_type = ['attention_max']
	p_type = fusion_type[0]

	if os.path.exists(output_path) is False:
		os.mkdir(output_path)

	# in_c = 3 for RGB images; in_c = 1 for gray images
	in_c = 1
	if in_c == 1:
		out_c = in_c
		mode = 'L'
		model_path = args.model_path_gray
	else:
		out_c = in_c
		mode = 'RGB'
		model_path = args.model_path_rgb

	with torch.no_grad():
		print('SSIM weight ----- ' + args.ssim_path[2])
		ssim_weight_str = args.ssim_path[2]
		model = load_model(model_path, in_c, out_c)
		for i in range(21):
			index = i + 1
			infrared_path = test_path + 'IR' + str(index) + '.jpg'
			visible_path = test_path + 'VIS' + str(index) + '.jpg'
			run_demo(model, infrared_path, visible_path, output_path, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode, p_type)
	print('Done......')

if __name__ == '__main__':
	main()
