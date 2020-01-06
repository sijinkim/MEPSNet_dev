#psnr per noise type
from json import loads
import os

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import sturactural_silimarity as ssim

class PSNR_SSIM_by_type:
    def __init__(self, dataset, num_partition, image_name, phase_type):
        #phase_type: valid or test
        self.phase = phase_type
        self.file_path = os.path.join('/home/tiwlsdi0306/workspace/image_dataset/noiseInfo', dataset, f'{dataset}_{self.phase}_part{num_partition}.json')
        
        json_file = open(self.file_path, 'r').read()
        self.noise_info = loads(json_file)       
        self.image_name = image_name
        
        self.type_psnr = {'gwn': 0, 'gblur': 0, 'contrast': 0, 'fnoise': 0}
        self.type_ssim = {'gwn': 0, 'gblur': 0, 'contrast': 0, 'fnoise': 0}

    def get_noise_info(self):
        #TODO: change search method to be more efficiently
        for infos in self.noise_info:
            if self.image_name in infos['path']:
                return infos

    def get_psnr(self, x, ref):
        data = self.get_noise_info()
        
        for d in data['objects']:
            psnr_ = psnr(x[int(d['ymin']):int(d['ymax']), int(d['xmin']):int(d['xmax']),:] , 
                        ref[int(d['ymin']):int(d['ymax']), int(d['xmin']):int(d['xmax']),:] , data_range=1)
            self.type_psnr[d['type']] += pnsr_

    def get_ssim(self, x, ref): 
        data = self.get_noise_info()
        
        for d in data['objects']:
            ssim_ = ssim(x[int(d['ymin']):int(d['ymax']), int(d['xmin']):int(d['xmax']),:] , 
                        ref[int(d['ymin']):int(d['ymax']), int(d['xmin']):int(d['xmax']),:] , data_range=1, multichannel=True)
            self.type_ssim[d['type']] += ssim_



