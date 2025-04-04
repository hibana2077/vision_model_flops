import timm

print(timm.list_models(
    filter="dla*",
    pretrained=False))