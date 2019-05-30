import numpy as np
import torch 



class DataNormalizer(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        # self.s_a = 0 
        # self.s_b = 0
        # self.p_a = 0
        # self.p_b = 0


        self._range_normalizer(magnitude_margin=0.8, IF_margin=1.0)
        print("s_a:", self.s_a )
        print("s_b:", self.s_b )
        print("p_a:", self.p_a)
        print("p_b:", self.p_b)

    # def _range_normalizer(x, margin):
    # #     x = x.flatten()
    #     min_x = x.min()
    #     max_x = x.max()
        
    #     a = margin * (2.0 / (max_x - min_x))
    #     b = margin * (-2.0 * min_x / (max_x - min_x) - 1.0)
    #     return a, b

    def _range_normalizer(self, magnitude_margin, IF_margin):
    #     x = x.flatten()
        min_spec = 10000
        max_spec = -10000
        min_IF = 10000
        max_IF = -10000

        for batch_idx, (spec, IF, pitch_label, mel_spec, mel_IF) in enumerate(self.dataloader.train_loader): 
            
            # training mel
            spec = mel_spec
            IF = mel_IF


            
            # print("spec",spec.shape)
            # print("IF",IF.shape)
            
            if spec.min() < min_spec: min_spec=spec.min()
            if spec.max() > max_spec: max_spec=spec.max()

            if IF.min() < min_IF: min_IF=IF.min()
            if IF.max() > max_IF: max_IF=IF.max()

            # print(min_spec)
            # print(max_spec)
            # print(min_IF)
            # print(max_IF)
    
        self.s_a = magnitude_margin * (2.0 / (max_spec - min_spec))
        self.s_b = magnitude_margin * (-2.0 * min_spec / (max_spec - min_spec) - 1.0)
        
        self.p_a = IF_margin * (2.0 / (max_IF - min_IF))
        self.p_b = IF_margin * (-2.0 * min_IF / (max_IF - min_IF) - 1.0)


    def normalize(self, feature_map):
        # print("feature_map",feature_map.shape)
        # spec = feature_map[:, :, :, 0]
        # IF = feature_map[:, :, :, 1]

        # s_a, s_b = self._range_normalizer(spec, 0.8)
        # p_a, p_b = self._range_normalizer(IF, 0.8)

        a = np.asarray([self.s_a, self.p_a])[None, :, None, None]
        b = np.asarray([self.s_b, self.p_b])[None, :, None, None]
        a = torch.FloatTensor(a).cuda()
        b = torch.FloatTensor(b).cuda()
        # print("feature_map",feature_map.shape)
        feature_map = feature_map *a + b
        # print("spec Max",feature_map[:,0,:,:].max())
        # print("spec min",feature_map[:,0,:,:].min())
        # print("IF Max",feature_map[:,1,:,:].max())
        # print("IF min",feature_map[:,1,:,:].min())

        # clip_spec = spec *s_a  + s_b
        # clip_IF = IF*p_a + p_b
        # return clip_spec, clip_IF, (s_a, s_b), (p_a, p_b)
        return feature_map

    def denormalize(spec, IF, s_a, s_b, p_a, p_b):
        spec = (spec -s_b) / s_a
        IF = (IF-p_b) / p_a
        return spec, IF