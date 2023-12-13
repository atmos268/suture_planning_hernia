from PIL import Image, ImageEnhance


def adjust_contrast_saturation(img, saturationFactor, contrastFactor):
    def adjust_saturation(image, factor):
        image_hsv = image.convert("HSV")
        enhancer = ImageEnhance.Color(image_hsv)
        image_saturated = enhancer.enhance(factor)
        image_rgb = image_saturated.convert("RGB")
        return image_rgb

    def adjust_contrast(image, factor):
        enhancer = ImageEnhance.Contrast(image)
        image_contrasted = enhancer.enhance(factor)
        return image_contrasted
    saturated = adjust_saturation(img, saturationFactor)
    contrasted = adjust_contrast(saturated, contrastFactor)
    return contrasted