import cv2, numpy as np
from PIL import Image

def quality_metrics(img):
    rgb = np.array(img.convert("RGB")); gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return dict(sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                brightness_mean=float(gray.mean()), contrast_std=float(gray.std()),
                edge_density=float((edges > 0).mean()))

def artifact_flags(m, blur_thresh=100.0):
    f = []
    if m["sharpness"] < blur_thresh: f.append("blurry")
    if m["contrast_std"] < 35: f.append("low_contrast")
    if m["brightness_mean"] < 60: f.append("dark")
    if m["brightness_mean"] > 200: f.append("overexposed")
    return f

def repair_artifacts(rgb, thresh=35, max_frac=0.01):
    """Member 1's method: median-difference outlier detection -> edge-protected mask -> Telea inpainting.
    Skips the repair if the mask is implausibly large (> max_frac of pixels = texture, not artifacts)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray, cv2.medianBlur(gray, 3))
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    edges = cv2.dilate(cv2.Canny(gray, 80, 160), np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges))          # protect real edges
    n = int(np.count_nonzero(mask)); n_regions = cv2.connectedComponents(mask)[0] - 1
    if n == 0 or n > max_frac * mask.size:
        return rgb, dict(artifact_pixels=n, artifact_regions=n_regions, repaired=False)
    fixed = cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA)
    mse = float(np.mean((fixed.astype(np.float64) - rgb) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    return fixed, dict(artifact_pixels=n, artifact_regions=n_regions, repaired=True, repair_psnr=round(psnr, 2))

def postprocess(img, target_size=None, blur_thresh=100.0):
    """PIL in -> (PIL out, metrics). Resize -> artifact repair -> CLAHE -> bilateral -> conditional sharpen."""
    out = img.convert("RGB")
    if target_size and out.size != (target_size, target_size):
        out = out.resize((target_size, target_size), Image.Resampling.LANCZOS)
    before = quality_metrics(out)
    rgb, art = repair_artifacts(np.array(out))                        # artifact detection + correction
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)                        # colour / contrast
    lab[..., 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb = cv2.bilateralFilter(rgb, d=5, sigmaColor=40, sigmaSpace=40) # edge-aware denoise
    sharpened = False
    if quality_metrics(Image.fromarray(rgb))["sharpness"] < blur_thresh:
        soft = cv2.GaussianBlur(rgb, (0, 0), 1.2)
        rgb = cv2.addWeighted(rgb, 1.4, soft, -0.4, 0); sharpened = True
    out2 = Image.fromarray(rgb)
    after = quality_metrics(out2)
    return out2, dict(before=before, after=after, flags_before=artifact_flags(before, blur_thresh),
                      flags_after=artifact_flags(after, blur_thresh), sharpened=sharpened, **art)
