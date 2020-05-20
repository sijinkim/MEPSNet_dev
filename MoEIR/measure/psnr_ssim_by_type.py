from json import loads
import os
import math
from .utility import calc_psnr, calc_ssim


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
                psnr_ = calc_psnr(ref[int(d['ymin']):int(d['ymax']), int(d['xmin']):int(d['xmax']),:],
                                  x[int(d['ymin']):int(d['ymax']), int(d['xmin']):int(d['xmax']),:])

                self.type_psnr[d['type']] += psnr_
                self.num_psnr[d['type']] += 1
                
                # For record each noise region info
                print(f'{d["type"]}/H,W({d["ymin"]}-{d["ymax"]}, {d["xmin"]}-{d["xmax"]}): {format(psnr_, ".3f")}')
                
        except: pass

    def get_ssim(self, x, ref, image_name): 
        data = self.get_noise_info(image_name)
        
        try: 
            for d in data['objects']:
                ssim_ = calc_ssim(ref[int(d['ymin']):int(d['ymax']), int(d['xmin']):int(d['xmax']),:], 
                                  x[int(d['ymin']):int(d['ymax']), int(d['xmin']):int(d['xmax']),:])
                
                self.type_ssim[d['type']] += ssim_
                self.num_ssim[d['type']] += 1
        
                # For record each noise region info
                print(f'{d["type"]}/H,W({d["ymin"]}-{d["ymax"]}, {d["xmin"]}-{d["xmax"]}): {format(ssim_, ".3f")}')
                
        except: pass

    def get_psnr_result(self):
        psnr_result = {}
        for key in self.type_psnr.keys():
            try:
                value = float(self.type_psnr[key]/self.num_psnr[key])
                if not value == float('inf'):
                    psnr_result[key] = value
                else:
                    psnr_result[key] = 50.0
            except: 
                raise ValueError
        return psnr_result
    
    def get_ssim_result(self):
        ssim_result = {}
        for key in self.type_ssim.keys():
            try:
                value = float(self.type_ssim[key]/self.num_ssim[key])
                if not value == float('inf'):
                    ssim_result[key] = value
                else:
                    ssim_result[key] = 1
            except:
                raise ValueError 
        return ssim_result
