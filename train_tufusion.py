
import os
import time
import numpy as np
from tqdm import trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import TUFusion_net
from args_fusion import args
import pytorch_msssim




def main():
	# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
	original_imgs_path = utils.list_images(args.dataset)
	train_num = 20000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)
	# for i in range(5):
	i = 0
	j = 2
	train(i, j, original_imgs_path)


def train(i, j, original_imgs_path):

	batch_size = args.batch_size

	# load network model, RGB
	in_c = 1 # 1 - gray; 3 - RGB
	if in_c == 1:
		img_model = 'L'
	else:
		img_model = 'RGB'
	input_nc = in_c
	output_nc = in_c
	tufusion_model = TUFusion_net(input_nc, output_nc)

	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		tufusion_model.load_state_dict(torch.load(args.resume))
	print(tufusion_model)
	optimizer = Adam(tufusion_model.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim
	crossentropyLoss = torch.nn.CrossEntropyLoss()

	if args.cuda:
		tufusion_model.cuda()

	tbar = trange(args.epochs)
	print('Start training.....')



	# creating save path
	temp_path_model = os.path.join(args.save_model_dir, args.ssim_path[i])
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	temp_path_loss = os.path.join(args.save_loss_dir, args.ssim_path[i])
	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	Loss_pixel = []
	Loss_ssim = []
	Loss_grd =[]
	Loss_all = []
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	all_grd_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		tufusion_model.train()
		count = 0
		for batch in range(batches):
			image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH, mode=img_model)

			count += 1
			optimizer.zero_grad()
			img = Variable(img, requires_grad=False)

			if args.cuda:
				img = img.cuda()
			# get fusion image
			# encoder
			en = tufusion_model.encoder(img)
			# decoder
			outputs = tufusion_model.decoder(en)

			# resolution loss
			x = Variable(img.data.clone(), requires_grad=False)

			ssim_loss_value = 0.
			pixel_loss_value = 0.
			gra_loss_value = 0.
			for output in outputs:
				pixel_loss_temp = mse_loss(output, x)
				ssim_loss_temp = ssim_loss(output, x, normalize=True)
				gra_loss_temp = mse_loss(utils.gradient(output), utils.gradient(x))

				ssim_loss_value += (1-ssim_loss_temp)
				pixel_loss_value += pixel_loss_temp
				gra_loss_value += gra_loss_temp

			ssim_loss_value /= len(outputs)
			pixel_loss_value /= len(outputs)
			gra_loss_value /= len(outputs)

			# total loss
			total_loss = pixel_loss_value + args.ssim_weight[i] * ssim_loss_value + args.grd_weight[j] * gra_loss_value
			total_loss.backward()
			optimizer.step()

			all_ssim_loss += ssim_loss_value.item()
			all_pixel_loss += pixel_loss_value.item()
			all_grd_loss += gra_loss_value.item()

			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss:{:.6f}\t ssim loss:{:.6f}\t " \
					"gra loss:{:.6f}\t   total: {:.6f}".format(
					time.ctime(), e + 1, count, batches,
								  all_pixel_loss / args.log_interval,
								  all_ssim_loss / args.log_interval,
					              all_grd_loss / args.log_interval,
								  (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss + args.grd_weight[j] * all_grd_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_grd.append(all_grd_loss / args.log_interval)

				Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss + args.grd_weight[j] * all_grd_loss ) / args.log_interval)

				all_ssim_loss = 0.
				all_pixel_loss = 0.
				all_grd_loss = 0.


	# pixel loss
	loss_data_pixel = np.array(Loss_pixel)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':','_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})

	# gra loss
	loss_data_gra = np.array(Loss_grd)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_grd_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_grd': loss_data_gra})

	# SSIM loss
	loss_data_ssim = np.array(Loss_ssim)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
	# all loss
	loss_data_total = np.array(Loss_all)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_total_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_total': loss_data_total})
	# save model
	tufusion_model.eval()
	tufusion_model.cpu()
	save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_model_path = os.path.join(args.save_model_dir, save_model_filename)
	torch.save(tufusion_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
	main()
