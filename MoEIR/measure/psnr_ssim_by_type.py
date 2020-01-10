from json import loads
import os

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class PSNR_SSIM_by_type:
    def __init__(self, dataset, num_partition, phase_type):
        #phase_type: valid or test
        self.n_partition = num_partition
        self.file_path = os.path.join('/home/tiwlsdi0306/workspace/image_dataset/noiseInfo', dataset, f'{dataset}_{phase_type}_part{num_partition}.json')
        
        json_file = open(self.file_path, 'r').read()
        self.noise_info = loads(json_file)       
        
        self.type_psnr = {'gwn': 0, 'gblur': 0, 'contrast': 0, 'fnoise': 0}
        self.type_ssim = {'gwn': 0, 'gblur': 0, 'contrast': 0, 'fnoise': 0}

        self.num_psnr = {'gwn': 0, 'gblur': 0, 'contrast': 0, 'fnoise': 0}
        self.num_ssim = {'gwn': 0, 'gblur': 0, 'contrast': 0, 'fnoise': 0}

    def get_noise_info(self, image_name):
        #TODO: change search method to be more efficiently
        for infos in self.noise_info:
            if image_name in infos['path']:
                return infos
    
    def get_psnr(self, x, ref, image_name):
        data = self.get_noise_info(image_name)
        try: 
            for d in data['objects']:
                psnr_ = psnr(ref[int(d['ymin']):int(d['ymax']), int(d['xmin']):int(d['xmax']),:] , 
                            x[int(d['ymin']):int(d['ymax']), int(d['xmin']):int(d['xmax']),:] , data_range=1)
                self.type_psnr[d['type']] += psnr_
                self.num_psnr[d['type']] += 1
        except: pass

    def get_ssim(self, x, ref, image_name): 
        data = self.get_noise_info(image_name)
        
        try: 
            for d in data['objects']:
                ssim_ = ssim(ref[int(d['ymin']):int(d['ymax']), int(d['xmin']):int(d['xmax']),:] , 
                        x[int(d['ymin']):int(d['ymax']), int(d['xmin']):int(d['xmax']),:] , data_range=1, multichannel=True)
                self.type_ssim[d['type']] += ssim_
                self.num_ssim[d['type']] += 1
        except: pass

    def get_psnr_result(self):
        psnr_result = {}
        #print(f'Number of types: {self.num_psnr}')
        #print(f'Total psnr of types: {self.type_psnr}')
        for key in self.type_psnr.keys():
            try:
                psnr_result[key] = float(self.type_psnr[key]/self.num_psnr[key])
            except ValueError: 
                psnr_result[key] = 0
        return psnr_result
    
    def get_ssim_result(self):
        ssim_result = {}
        #print(f'Number of types: {self.num_ssim}')
        #print(f'Total ssim of types: {self.type_ssim}')
        for key in self.type_ssim.keys():
            try:
                ssim_result[key] = float(self.type_ssim[key]/self.num_ssim[key])
            except ValueError: 
                ssim_result[key] = 0
        return ssim_result
