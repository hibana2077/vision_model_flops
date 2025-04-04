import timm

print(timm.list_models(
    filter="volo*224",
    pretrained=False))