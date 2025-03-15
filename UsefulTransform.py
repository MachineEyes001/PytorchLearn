from PIL import Image
from torchvision import  transforms
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("logs")
img=Image.open("image/cat.jpg")

# Totensor
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)
writer.add_image("ToTensor",tensor_img)

# Normalize
trans_norm=transforms.Normalize([1,3,5],[2,4,6])
img_norm=trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm,1)

# Resize
print(img.size)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)
img_resize=tensor_trans(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize)

# Compose - resize - 2
trans_resize2=transforms.Resize(512)
trans_compose=transforms.Compose([tensor_trans,trans_resize2])
img_resize2=trans_compose(img)
writer.add_image("Resize",img_resize2,1)

# RandomCrop
trans_random=transforms.RandomCrop(512)
trans_compose2=transforms.Compose([trans_random,tensor_trans])
for i in range(10):
    img_crop=trans_compose2(img)
    writer.add_image("RandomCrop", img_crop, i)


writer.close()