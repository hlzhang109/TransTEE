import numpy as np

x= [0.5620914052451201, 0.5215754348851029, 0.784318085586459, 0.7604542166526875, 0.7092027480075239, 0.5182308972804793, 0.8223950050981851, 0.597213861570512, 1.0733737506231125, 0.5725754683155745, 0.6713834581372445, 0.6194506900070181, 0.5697704886905975, 0.530153647675613, 0.7219571119247497, 0.5790409016643813, 0.5226584845724142, 0.5508570166076701]

print('{:.4f} ± {:.4f}'.format(np.mean(x), np.std(x)))