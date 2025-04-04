import timm

print(timm.list_models(
    filter="eva*224",
    pretrained=False))