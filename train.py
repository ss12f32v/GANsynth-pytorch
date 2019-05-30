import os
import random
import warnings
from PIL import Image
from PGGAN import *
from normalizer import DataNormalizer
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

import torch.utils.data as udata
import torchvision.datasets as vdatasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import h5py 
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


folder ='test'
BATCH_SIZE= 4
DEVICE = torch.device("cuda:0")
G_LR = 2e-4
D_LR = 4e-4
ADAM_BETA = (0.0, 0.99)
ADAM_EPS = 1e-8
LAMBDA_FOR_WGANGP = 1
CRITIC_FOR_WGANGP = 1

TOTAL_DATA_SIZE=11864 #validation Nsynth dataset size. If use trainingset, this argument should be changed


class NsynthDataLoader(object):
    def __init__(self):
        cuda = torch.device("cuda:0")
        instr='guitar'
        # Nsynth_spec_IF_pitch.hdf5
        # self.dataset = h5py.File('../data/Nsynth_spec_IF_pitch.hdf5','r')     
        self.dataset = h5py.File('data/Nsynth_valid_spec_IF_pitch_and_melSpec.hdf5','r')     
        spec = self.dataset["Spec"][:]
        IF = self.dataset["IF"][:]
        pitch = self.dataset["pitch"][:]
        mel_spec = self.dataset["mel_Spec"][:]
        mel_IF = self.dataset["mel_IF"][:]
        
        self.train_dataset = udata.TensorDataset(torch.Tensor(spec),
                                        torch.Tensor(IF),
                                        torch.LongTensor(pitch),
                                        torch.Tensor(mel_spec),
                                        torch.Tensor(mel_IF)
                                       )
        self.train_loader = udata.DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    def change_batch_size(self, batch_size):
        self.train_loader = udata.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)


class PGGAN(object):
    def __init__(self,
                 resolution,            # Resolution.
                 latent_size,           # Dimensionality of the latent vectors.
                 dataloader,
                 criterion_type="GAN",  # ["GAN", "WGAN-GP"]
                 rgb_channel=2,         # Output channel size, for rgb always 3
                 fmap_base=2 ** 11,     # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,        # log2 feature map reduction when doubling the resolution.
                 fmap_max=2 ** 6,       # Maximum number of feature maps in any layer.
                 is_tanh=True,
                 is_sigmoid=True
                 ):
        self.dataloader=dataloader
        self.latent_size_ = latent_size
        self.rgb_channel_ = rgb_channel
        self.fmap_base_ = fmap_base
        self.fmap_decay_ = fmap_decay
        self.fmap_max_ = fmap_max
        
        # self.stable_and_fadein_step = [6, 6, 6, 6, 6, 6, 30]
        self.stable_and_fadein_step = [1,1,1,1,1,1,1]



        
        self.criterion_type_ = criterion_type
        self.is_tanh_ = is_tanh
        self.is_sigmoid_ = False if self.criterion_type_ in ["WGAN-GP"] else is_sigmoid

        self.gradient_weight_real_ = torch.FloatTensor([-1]).cuda()
        self.gradient_weight_fake_ = torch.FloatTensor([1]).cuda()

    
        self.init_data_normalizer()
        self._init_network()
        self.avg_layer = torch.nn.AvgPool2d((2,2),stride=(2,2))
    
    
    def train(self):
        
        self._init_optimizer()
        self._init_criterion()
        
        # Declare Model status
        net_level = 0
        net_status = "stable"
        net_alpha = 1.0
        
        
        for cur_level in range(7):
            self.stable_steps = self.stable_and_fadein_step[cur_level]
            self.fadein_steps = self.stable_and_fadein_step[cur_level] 

        
            if cur_level==6:
                self.dataloader.change_batch_size(batch_size=4)
                self.stable_steps = 100
                
            if cur_level ==0:
                net_status == "stable"
                for step in range(self.stable_steps):
                    self._train(cur_level, net_status, net_alpha, step)
            else:
                net_status = "fadein"
                
                for step in range(self.fadein_steps):
#                     net_alpha = 1.0 - (step + 1) / fadein_steps
                    self._train(cur_level, "fadein", net_alpha, step)
                    
                for step in range(self.stable_steps*2):
                    net_alpha = 1.0
                    self._train(cur_level, "stable", net_alpha, step)
                    if cur_level ==6:
                        torch.save(self.g_net.state_dict(), folder + '/Gnet_%dx%d_step%d.pth' % (2 ** (cur_level + 1), 2 ** (cur_level + 4), step))
                        torch.save(self.d_net.state_dict(), folder + '/Dnet_%dx%d_step%d.pth' % (2 ** (cur_level + 1), 2 ** (cur_level + 4), step))
            torch.save(self.g_net.state_dict(), folder + '/Gnet_%dx%d.pth' % (2 ** (cur_level + 1), 2 ** (cur_level + 4)))
            torch.save(self.d_net.state_dict(), folder + '/Dnet_%dx%d.pth' % (2 ** (cur_level + 1), 2 ** (cur_level + 4)))

             
    def _train(self, net_level, net_status, net_alpha, cur_step):
            
        current_level_res = 2 ** (net_level + 1)
            
        # for batch_idx, (score, target_spectrum, IF) in enumerate(self.dataloader.train_loader): 
        for batch_idx, (spec, IF, pitch_label, mel_spec, mel_IF) in enumerate(self.dataloader.train_loader): 
            

            # train mel spec IF
            spec = mel_spec
            IF = mel_IF

            stack_real_image = torch.stack((spec,IF),dim=1).cuda()
            stack_real_image = torch.transpose(stack_real_image,2,3)

            little_batch_size = spec.size()[0]


            
            while stack_real_image.size()[2] != current_level_res:
                stack_real_image = self.avg_layer(stack_real_image)
            stack_real_image = stack_real_image.cuda(0)

        
            if net_status =='stable':
                net_alpha = 1.0
            elif net_status =='fadein':
                
                if little_batch_size==BATCH_SIZE:
                    net_alpha = 1.0 - (cur_step * TOTAL_DATA_SIZE + batch_idx * little_batch_size) / (self.fadein_steps * TOTAL_DATA_SIZE)
                else:
                    net_alpha = 1.0 - (cur_step * TOTAL_DATA_SIZE + batch_idx*(BATCH_SIZE) + little_batch_size) / (self.fadein_steps * TOTAL_DATA_SIZE)

            if net_alpha< 0.0:
                print("Alpha too small <0")
                return 

            # change net status 
            self.g_net.net_config = [net_level, net_status, net_alpha]
            self.d_net.net_config = [net_level, net_status, net_alpha]
            
            """ Make Fake Condition Vector """
            pitch_label = pitch_label.cuda()
            fake_pitch_label = torch.LongTensor(little_batch_size, 1).random_() % 128            
            fake_one_hot_pitch_condition_vector = torch.zeros(little_batch_size, 128).scatter_(1, fake_pitch_label, 1).unsqueeze(2).unsqueeze(3).cuda()
            fake_pitch_label = fake_pitch_label.cuda().squeeze()


            """ generate random vector """
            fake_seed = torch.randn(little_batch_size, self.latent_size_, 1, 1).cuda()
            fake_seed_and_pitch_condition = torch.cat((fake_seed, fake_one_hot_pitch_condition_vector), dim=1)

            fake_generated_sample = self.g_net(fake_seed_and_pitch_condition)
            

            stack_real_image = self.data_normalizer.normalize(stack_real_image)
            pitch_real, d_real = self.d_net(stack_real_image)
            pitch_fake, d_fake  = self.d_net(fake_generated_sample.detach())

          



            # WGAN-GP
            """ update d_net """
            # real:-1 fake:1
            for p in self.d_net.parameters():
                p.requires_grad = True 
            self.d_net.zero_grad()


            # Train D with Real 
            mean_real = d_real.mean() # wgan loss 
            # mean_real = nn.ReLU()(1.0 - d_real).mean() #hinge loss

            
            # Train D with Fake 
            mean_fake = d_fake.mean() # wgan loss
            # mean_fake = nn.ReLU()(1.0 + d_fake).mean() # hinge loss

            
            # Train D with GP 
            gradient_penalty = 10 * self._gradient_penalty(stack_real_image.data, fake_generated_sample.data, little_batch_size, current_level_res)


            # Train D with classifier Loss
            
            real_pitch_loss = self.NLL_loss(pitch_real, pitch_label)
            fake_pitch_loss = self.NLL_loss(pitch_fake, fake_pitch_label)
            p_loss = 10 *(real_pitch_loss)

            # D_loss = mean_fake + mean_real + gradient_penalty + p_loss # hinge
            D_loss = mean_fake - mean_real + gradient_penalty + p_loss # wgan-gp
            Wasserstein_D = mean_real - mean_fake

            D_loss.backward()
            self.d_optim.step()


            if batch_idx% 3 ==0: # avoid Mode Collpase which caused by strong Generator

                """ update g_net """
                for p in self.d_net.parameters():
                    p.requires_grad = False  # to avoid computation
                self.g_net.zero_grad()
                pitch_fake, d_fake = self.d_net(fake_generated_sample)
                mean_fake = d_fake.mean()

                fake_pitch_loss = self.NLL_loss(pitch_fake, fake_pitch_label)
                timed_fake_pitch_loss = 10 *fake_pitch_loss
                G_loss = -mean_fake + timed_fake_pitch_loss
                G_loss.backward()
                

        
            self.g_optim.step()
            if batch_idx %1400 ==0:
                self.generate_picture(fake_generated_sample[:,0,:,:], current_level_res, cur_step, batch_idx, net_status)
            if batch_idx %200==0:
                print("Resolution:{}x{}, Status:{}, Cur_step:{}, Batch_id:{}, D_loss:{}, W_D:{}, M_fake:{}, M_Real:{}, GP:{}, Real_P_Loss:{}, Fake_P_Loss:{}, G_loss:{}, Net_alpha:{}".format(\
                        current_level_res, current_level_res*(2**3), net_status, cur_step, batch_idx, D_loss, Wasserstein_D, mean_fake, mean_real, real_pitch_loss, fake_pitch_loss, gradient_penalty, G_loss, net_alpha))
        print("self.fadein_steps",self.fadein_steps* TOTAL_DATA_SIZE,\
              "cur_step",(cur_step * TOTAL_DATA_SIZE + batch_idx * little_batch_size),\
              "net_alpha",net_alpha
             
              )
    def generate_picture(self, spec, resolution, step, batch_idx, status):
        spec = spec.data.cpu().numpy()
        stack_spec = np.hstack((spec[0],spec[1],spec[2],spec[3]))
        flip_stack = np.flipud(stack_spec)
        fig = plt.figure(figsize=(20,7))
        plt.imshow(stack_spec,aspect='auto')
        
        plt.savefig( folder + "/{}_{}_{}_{}_{}_sample.png".format(resolution, resolution*(2**3), status, step, batch_idx ))
    
    

    def _init_criterion(self):
        
        self.criterion = self._gradient_penalty
        self.NLL_loss = nn.NLLLoss()
           
    def init_data_normalizer(self):

        self.data_normalizer = DataNormalizer(self.dataloader)

    def _init_network(self):
        # Init Generator and Discriminator
        print("Create Network")
        self.g_net = Generator(256, self.latent_size_, self.rgb_channel_,
                                is_tanh=self.is_tanh_, channel_list=[256,256,256,256,256,128,64,32])
        self.d_net = Discriminator(256, self.rgb_channel_,
                                   is_sigmoid=self.is_sigmoid_, channel_list=[256,256,256,256,128,64,32,32])
        # if TO_GPU:
        self.g_net.cuda(0)
        self.d_net.cuda(0)
        print(self.g_net)
        print(self.d_net)
        
        
    def _init_optimizer(self):
        self.g_optim = optim.Adam(self.g_net.parameters(), lr=G_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        self.d_optim = optim.Adam(self.d_net.parameters(), lr=D_LR, betas=ADAM_BETA, eps=ADAM_EPS)
        
    def _gradient_penalty(self, real_data, fake_data, batch_size, res):
        """
        This algorithm was mentioned on the Page4 of paper
        'Improved Training of Wasserstein GANs'
        This implementation was from 'https://github.com/caogang/wgan-gp'
        """
        
        # print("real_data.nelement() / batch_size",real_data.nelement() / batch_size)
        epsilon = torch.rand(batch_size, 1)
        epsilon = epsilon.expand(batch_size, real_data.nelement() / batch_size).contiguous().view(batch_size, 2, res, res*(2**3))
#         epsilon = epsilon.expand_as(real_data)
#         print("epsilon",epsilon.size())
        epsilon = epsilon.cuda()
        median_x = epsilon * real_data + ((1 - epsilon) * fake_data)

        # if TO_GPU:
        median_x = median_x.cuda()
        median_data = torch.autograd.Variable(median_x, requires_grad=True)

        _, d_median_data = self.d_net(median_data)

        gradients = torch.autograd.grad(outputs=d_median_data, inputs=median_data,
                                        grad_outputs=torch.ones(d_median_data.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA_FOR_WGANGP
        return gradient_penalty




def main():    
    dataloader = NsynthDataLoader()
    p = PGGAN(512, 256, dataloader = dataloader, criterion_type="WGAN-GP")
    
    p.train()   


if __name__ == "__main__":
    main()
