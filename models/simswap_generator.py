import os
import sys

import cv2
import numpy as np
import torch


class SimSwapGenerator:
    def __init__(self, weights_dir, simswap_repo, device=None, crop_size=224):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # Set compute device to GPU if available and requested, otherwise default to CPU
        self.crop_size = crop_size  # Define the pixel dimensions for cropping face images

 # Add the external SimSwap repository path to the system path for module resolution
        if simswap_repo not in sys.path:
            sys.path.insert(0, simswap_repo)

  # Import the internal model creation function from the SimSwap repository
        try:
            from models.models import create_model  # SimSwap repo's own factory
        except ImportError as e:
            raise ImportError(
                "Could not import SimSwap's model code. Clone "
                "https://github.com/neuralchen/SimSwap and pass its path as "
                "--simswap-repo."
            ) from e

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "insightface is required for face detection/alignment. "
                "pip install insightface onnxruntime-gpu"
            ) from e

        self.face_app = FaceAnalysis(name="antelope", root=weights_dir)
        self.face_app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(320, 320))

        class _Opt:
            name = "people"
            gpu_ids = "0" if self.device == "cuda" else "-1"
            checkpoints_dir = weights_dir
            isTrain = False
            Arc_path = os.path.join(weights_dir, "arcface_checkpoint.tar")
            crop_size = crop_size

# Instantiate the core SimSwap model structure using the mock options object
        self.model = create_model(_Opt())

# Transition the model parameters into evaluation mode to disable layers like dropout
        self.model.eval()

    def _align(self, img_bgr):
        faces = self.face_app.get(img_bgr)  # Detect all human faces present within the input BGR image frame
        if not faces:
            return None, None

         # Select the single largest face found in the image based on bounding box area
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Normalize, crop, and generate the transformation matrix for the chosen face
        aligned, m = insightface_align(img_bgr, face, self.crop_size)
        return aligned, m

# Preprocesses an aligned BGR face crop, resizes it to 112x112, and extracts an L2-normalized identity embedding vector using the ArcFace network.
    def _identity_embedding(self, aligned_source_bgr):
        img = cv2.cvtColor(aligned_source_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.model.netArc(torch.nn.functional.interpolate(img, size=(112, 112)))
            latent = latent / latent.norm(dim=1, keepdim=True)
        return latent

# Detect and align faces from both the source image and target frame, returning None if a face is missing in either image.
    def swap(self, source_img_bgr, target_frame_bgr):
        src_aligned, _ = self._align(source_img_bgr)
        tgt_aligned, tgt_m = self._align(target_frame_bgr)
        if src_aligned is None or tgt_aligned is None:
            return None
 
 # Extract the specialized feature embedding vector from the source face
        latent = self._identity_embedding(src_aligned)

        tgt = cv2.cvtColor(tgt_aligned, cv2.COLOR_BGR2RGB)
        tgt = torch.from_numpy(tgt).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)
        tgt = tgt.unsqueeze(0).to(self.device)

# Feed target face and source embedding into the generator network without tracking gradients
        with torch.no_grad():
            swapped = self.model.netG(tgt, latent)
        swapped = swapped.squeeze(0).clamp(-1, 1).add(1).div(2).mul(255)
        swapped = swapped.permute(1, 2, 0).byte().cpu().numpy()
        swapped_bgr = cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR)

        return paste_back(swapped_bgr, target_frame_bgr, tgt_m)


# InsightFace to normalize, rotate, and crop a face based on facial landmarks, returning the aligned face and its transformation matrix.
def insightface_align(img_bgr, face, crop_size):
    from insightface.utils import face_align
    aligned, m = face_align.norm_crop2(img_bgr, face.kps, crop_size)
    return aligned, m

# Blend the swapped face patch back into the original frame using an inverted transformation matrix and a Gaussian-blurred edge mask.
def paste_back(swapped_crop_bgr, original_frame_bgr, m):
    h, w = original_frame_bgr.shape[:2]
    inv_m = cv2.invertAffineTransform(m)
    warped = cv2.warpAffine(swapped_crop_bgr, inv_m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    mask = np.ones(swapped_crop_bgr.shape[:2], dtype=np.float32)
    mask = cv2.warpAffine(mask, inv_m, (w, h))
    mask = cv2.GaussianBlur(mask, (15, 15), 0)[..., None]

    out = warped.astype(np.float32) * mask + original_frame_bgr.astype(np.float32) * (1 - mask)
    return out.astype(np.uint8)
