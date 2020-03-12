import plenopticam
from plenopticam import lfp_reader
from plenopticam import lfp_calibrator
from plenopticam import lfp_aligner
from plenopticam import lfp_extractor
from plenopticam import lfp_refocuser
from plenopticam.cfg.cfg import PlenopticamConfig
from plenopticam import misc
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import sys
import os
import pickle
import matplotlib.pyplot as plt
import time 


cfg = plenopticam.cfg.PlenopticamConfig()
cfg.default_values()
cfg.params['opt_dbug'] = 1
cfg.params['opt_cali'] = 1
cfg.params['opt_cali'] = 0
cfg.params['opt_hotp'] = 0
cfg.params['opt_colo'] = 0
cfg.params['opt_view'] = 1
cfg.params['ptc_leng'] = 33
cfg._dir_path = "calib"
cfg.params[cfg.cal_meta] = '/home/page/code/plenopticam/white2500.json'
cfg.params[cfg.cal_path] = 'cfg.json'


img_dir = "/home/page/code/plenopticam/"
gt_img_name = 'white2500.bmp'
image = Image.open(img_dir+gt_img_name)
whiteImg = TF.to_tensor(image)
whiteImg/=whiteImg.sum()
wht_img = whiteImg.squeeze().numpy()



image = Image.open(img_dir+'Default/img_channel000_position005_time000000000_z000.bmp')
whiteImg = TF.to_tensor(image)
whiteImg/=whiteImg.sum()
lfp_img = whiteImg.squeeze().numpy()


if not cfg.params[cfg.cal_path]:
    # open selection window (at current lfp file directory) to set calibration folder path
    cfg.params[cfg.cal_path] = misc.select_file(cfg.params[cfg.lfp_path], 'Select calibration image')

# instantiate status object
sta = misc.PlenopticamStatus()
sta.bind_to_interrupt(sys.exit)     # set interrupt

meta_cond = not (os.path.exists(cfg.params[cfg.cal_meta]) and cfg.params[cfg.cal_meta].lower().endswith('json'))
if meta_cond or cfg.params[cfg.opt_cali]:
    # perform centroid calibration
    cal_obj = lfp_calibrator.LfpCalibrator(wht_img, cfg, sta)
    cal_obj.main()
    cfg = cal_obj.cfg
    del cal_obj

# load calibration data
cfg.load_cal_data()

if cfg.cond_lfp_align():
    # align light field
    lfp_obj = lfp_aligner.LfpAligner(lfp_img, cfg, sta, wht_img)
    lfp_obj.main()
    lfp_obj = lfp_obj.lfp_img
    del lfp_obj

# load previously computed light field alignment
with open(os.path.join(cfg.exp_path, 'lfp_img_align.pkl'), 'rb') as f:
    lfp_img_align = pickle.load(f)


start = time.time()

# extract viewpoint data
lfp_calibrator.CaliFinder(cfg).main()
obj = lfp_extractor.LfpExtractor(lfp_img_align, cfg=cfg, sta=sta)
obj.main()
vp_img_arr = obj.vp_img_arr
del obj

end = time.time()
print(str(end-start))

# for nLens in range(1,39):
#     plt.imshow(vp_img_arr[:,:,nLens,1,0])
#     plt.show()