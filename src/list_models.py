import timm

print(timm.list_models(
    filter="eva02*clip*224",
    pretrained=False))