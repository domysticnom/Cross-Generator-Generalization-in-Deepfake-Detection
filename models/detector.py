import timm


def build_model(backbone, pretrained=True, num_classes=2):
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    return model
