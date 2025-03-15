from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer=SummaryWriter("logs")
image_path= "image/cat.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)

writer.add_image("test",img_array,1,dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=3x",3*i,i)

writer.close()
