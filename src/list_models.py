import timm

print(timm.list_models(
    filter="caformer*",
    pretrained=False))