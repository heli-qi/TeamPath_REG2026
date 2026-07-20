"""UNI2-h encoder — inference-only port of feature_extract.py (build_uni2h + GPU preprocess)."""
import numpy as np
import torch
import torch.nn.functional as F
import timm


def build_uni2h(weights_path):
    # MahmoodLab UNI2-h spec: ViT-Giant/14 with SwiGLU MLP, SiLU, 8 register tokens.
    model = timm.create_model(
        "vit_giant_patch14_224", pretrained=False, init_values=1e-5, num_classes=0,
        dynamic_img_size=True, img_size=224, patch_size=14, depth=24, num_heads=24,
        embed_dim=1536, mlp_ratio=2.66667 * 2, no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8,
    )
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"[uni2] state_dict missing={len(miss)} unexpected={len(unexp)}", flush=True)
    return model.eval().cuda().half()


@torch.inference_mode()
def extract_features(imgs, model, batch_size=64):
    """imgs: [N,256,256,3] uint8 -> features [N,1536] float16. Matches feature_extract.process_slide:
    /255 -> bilinear resize 224 -> ImageNet norm -> UNI2-h."""
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    N = imgs.shape[0]
    feats = np.empty((N, 1536), np.float16)
    for i in range(0, N, batch_size):
        batch = imgs[i:i + batch_size]
        x = torch.from_numpy(batch).cuda(non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        if x.shape[-1] != 224:
            x = F.interpolate(x, 224, mode="bilinear", align_corners=False, antialias=True)
        x = ((x - mean) / std).half()
        emb = model(x)
        feats[i:i + x.shape[0]] = emb.float().cpu().numpy().astype(np.float16)
    return feats
