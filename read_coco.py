import torchvision
from PIL import ImageDraw
coco_dataset=torchvision.datasets.CocoDetection(root=r"../val2017",annFile=r"./annotations/instances_val2017.json")
image,info=coco_dataset[0]
image.show()
image_handler=ImageDraw.ImageDraw(image)

for annotation in info:
    x_min,y_min,width,height=annotation['bbox']
    print(x_min)
    image_handler.rectangle(((x_min,y_min),(x_min+width,y_min+height)))

image.show()
