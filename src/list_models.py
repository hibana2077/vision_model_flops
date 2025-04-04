import timm

print(timm.list_models(
    filter="vovnet*",
    pretrained=False))