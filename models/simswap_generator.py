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
        _added_simswap_to_path = simswap_repo not in sys.path
        if _added_simswap_to_path:
            sys.path.insert(0, simswap_repo)

  # Import and instantiate SimSwap's own model, in one scope (see
  # _create_simswap_model below for why the import and the call must happen
  # together, not as two separate steps).

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "insightface is required for face detection/alignment. "
                "pip install insightface onnxruntime-gpu"
            ) from e

        self.face_app = FaceAnalysis(name="antelopev2", root=weights_dir)
        self.face_app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(320, 320))

        # SimSwap's real model.initialize() reads many more fields than a
        # small hand-built mock options object provides (confirmed: it
        # raised AttributeError on opt.resize_or_crop, and would keep raising
        # on further fields one at a time). Use SimSwap's OWN options class
        # instead, exactly as SimSwap's own predict.py does -- this guarantees
        # every field their code expects gets a real, author-intended default.
        gpu_ids_str = "0" if self.device == "cuda" else "-1"
        arc_path = os.path.join(weights_dir, "arcface_checkpoint.tar")

        opt = self._build_simswap_opt(
            simswap_repo, arc_path=arc_path,
            # SimSwap's own code builds its search path as
            # checkpoints_dir/<name>/<epoch>_net_<label>.pth. Step 4 extracts
            # weights to {weights_dir}/checkpoints/people/*.pth, so
            # checkpoints_dir must point at the "checkpoints" folder itself,
            # not weights_dir directly -- passing weights_dir alone made
            # SimSwap look one level too shallow and report the generator
            # checkpoint as missing, even though it's genuinely on disk.
            checkpoints_dir=os.path.join(weights_dir, "checkpoints"),
            crop_size=crop_size, gpu_ids=gpu_ids_str)

# Instantiate the core SimSwap model structure using the real parsed options object
        try:
            self.model = self._create_simswap_model(simswap_repo, opt)
        except ImportError as e:
            raise ImportError(
                "Could not import SimSwap's model code. Clone "
                "https://github.com/neuralchen/SimSwap and pass its path as "
                "--simswap-repo."
            ) from e

# Transition the model parameters into evaluation mode to disable layers like dropout
        self.model.eval()

        # SimSwap's own modules are already loaded into memory at this point;
        # sys.path only matters at import time, not for already-instantiated
        # objects, so it's safe to remove simswap_repo now. Left on sys.path
        # permanently, it would keep shadowing this project's own models/ and
        # options/ packages (if any) for the rest of the process -- confirmed
        # via a real reproduction where a later, unrelated
        # "from models.detector import build_model" broke because of this.
        if _added_simswap_to_path and simswap_repo in sys.path:
            sys.path.remove(simswap_repo)

    @staticmethod
    def _build_simswap_opt(simswap_repo, arc_path, checkpoints_dir, crop_size, gpu_ids):
        """
        Builds a real, fully-populated SimSwap options object via SimSwap's
        own options.test_options.TestOptions -- the same class SimSwap's own
        predict.py uses -- rather than a hand-rolled mock missing fields.
        Same sys.modules-collision guard as _create_simswap_model, in
        case anything else in this process ever defines a top-level
        "options" package.
        """
        collision_cache = {
            name: mod for name, mod in sys.modules.items()
            if name == "options" or name.startswith("options.")
        }
        for name in list(collision_cache):
            del sys.modules[name]
        try:
            from options.test_options import TestOptions
            options = TestOptions()
            options.initialize()
            opt = options.parser.parse_args([
                "--name", "people",
                "--Arc_path", arc_path,
                "--checkpoints_dir", checkpoints_dir,
                "--crop_size", str(crop_size),
                "--gpu_ids", gpu_ids,
                "--no_simswaplogo",
            ])
        finally:
            for name in list(sys.modules):
                if name == "options" or name.startswith("options."):
                    del sys.modules[name]
            sys.modules.update(collision_cache)

        # TestOptions' own parse() normally does this gpu_ids string->list
        # conversion and isTrain flag (see SimSwap's predict.py); replicated
        # here since we call parser.parse_args() directly rather than their
        # full parse() wrapper, to avoid also inheriting parse()'s own
        # argv-based CLI parsing (which would collide with OUR notebook's argv).
        if isinstance(opt.gpu_ids, str):
            str_ids = opt.gpu_ids.split(",")
            opt.gpu_ids = [int(i) for i in str_ids if int(i) >= 0]
        opt.isTrain = False
        return opt

    @staticmethod
    def _create_simswap_model(simswap_repo, opt):
        """
        Imports SimSwap's create_model AND calls it, in the same
        sys.modules-eviction scope. This project's own models/ package
        (where this very file lives) collides BY NAME with SimSwap's
        models/ package. A first attempt fixed the import itself (evict,
        import, restore) but that alone isn't enough: SimSwap's own
        create_model() does "from .fs_model import fsModel" -- a RELATIVE
        import -- internally. A relative import resolves against whatever
        "models" package is active in sys.modules at the moment
        create_model() actually RUNS, not at the moment it was imported. If
        the eviction scope already restored this project's own "models"
        package before create_model() is called (confirmed via a real
        reproduction matching the exact observed error,
        "ModuleNotFoundError: No module named 'models.fs_model'"), the
        relative import fails because it's now looking inside the wrong
        package. Fix: do the import AND the call inside the same scope, only
        restoring afterward.
        """
        project_models_cache = {
            name: mod for name, mod in sys.modules.items()
            if name == "models" or name.startswith("models.")
        }
        for name in list(project_models_cache):
            del sys.modules[name]
        try:
            from models.models import create_model
            model = create_model(opt)
        finally:
            for name in list(sys.modules):
                if name == "models" or name.startswith("models."):
                    del sys.modules[name]
            sys.modules.update(project_models_cache)
        return model

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
        # SimSwap's own preprocessing for netArc (their "transformer_Arcface")
        # uses standard ImageNet mean/std normalization, NOT the simple
        # symmetric [-1,1] normalization this used before -- confirmed
        # against SimSwap's own real source code. Feeding netArc a tensor
        # normalized the wrong way produces a numerically valid but
        # semantically wrong identity vector (a real 512-d, correctly
        # L2-normalized embedding that is nonetheless garbage from the
        # network's actual trained perspective), which corrupts netG's
        # identity conditioning -- this was the actual cause of the mostly-
        # zero, spatially-collapsed output observed in testing.
        img = cv2.cvtColor(aligned_source_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float().div(255)
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
        img = (img - mean) / std
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

        # SimSwap's own target-image preprocessing (their "_totensor"
        # function, confirmed from their actual test_wholeimage_swapsingle.py
        # source) only scales to [0, 1] -- no further shift/scale. This
        # previously forced [-1, 1] instead, which is a real, meaningful
        # distribution mismatch from what netG was trained on; confirmed as
        # the actual cause of the near-total output collapse seen in testing
        # (the ArcFace/identity-embedding fix alone did not resolve it).
        tgt = cv2.cvtColor(tgt_aligned, cv2.COLOR_BGR2RGB)
        tgt = torch.from_numpy(tgt).permute(2, 0, 1).float().div(255)
        tgt = tgt.unsqueeze(0).to(self.device)

# Feed target face and source embedding into the generator network without tracking gradients
        with torch.no_grad():
            swapped = self.model.netG(tgt, latent)
        # netG's output range is [0, 1], matching its [0, 1] input convention
        # (confirmed empirically: raw output min/max sat almost exactly in
        # [0, 1] with zero clamped/collapsed pixels once the input-side [0,1]
        # fix was applied -- not [-1, 1], which was the prior, incorrect
        # assumption here and produced a visible blue/desaturated cast).
        swapped = swapped.squeeze(0).clamp(0, 1).mul(255)
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
