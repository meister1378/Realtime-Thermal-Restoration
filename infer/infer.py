"""실시간 추론 파이프라인: raw16 -> [선택: real-NUC] -> PPFN-small-direct
(single forward pass) -> [선택: 렌즈 왜곡 보정].

학습 코드(train/train.py)와는 완전히 분리되어 있고, 저장된 체크포인트(.pth)
하나로만 연결된다 -- 이 파일은 train.py를 import하지 않고, train.py도 이
파일을 import하지 않는다. 둘이 공유하는 건 model/model.py(아키텍처)와
prompts/(필수 prompt 임베딩)뿐이다.

출력에 적용하는 왜곡 보정은 모델 출력 이후에만 적용한다 -- 모델은 왜곡된
픽셀 격자 그대로 학습/검증됐으므로, 입력을 먼저 펴버리면 학습 분포를 벗어난다.
기하 보정은 복원과 함께 학습할 대상이 아니라 닫힌 형태(closed-form)의 별도
후처리 단계로 둔다.
"""
import os
import sys

import cv2
import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'model'))
sys.path.insert(0, _HERE)
from data_io import (  # noqa: E402
    build_undistort_remap, cam_distortion, cam_intrinsics_matrix, load_camchain, normalize_16bit_to_unit,
)
from model import Restormer_PPFN  # noqa: E402

_PROMPTS_DIR = os.path.join(_HERE, '..', 'prompts')
_CKPT_DIR = os.path.join(_HERE, '..', 'checkpoints')


class RealtimeRestorer:
    """PPFN-small-direct를 한 번 로드하고, restore_frame(raw16)을 호출할
    때마다 (선택) real-NUC -> PPFN 1회 forward -> (선택) 렌즈 왜곡보정을
    거친 [0,1] float32 이미지를 반환한다.

    camchain_path / cam_name: 지정하면 매 restore_frame() 출력에
    build_undistort_remap으로 왜곡 보정을 적용한다(프레임 크기별로 캐시).
    nuc_gain / nuc_offset: 지정하면(2-point NUC 캘리브레이션 결과, raw
    프레임과 같은 shape) 기본 percentile-stretch 정규화 대신 이걸 먼저
    적용한다. 둘 다 선택 사항이고 서로 독립적이다.
    """

    def __init__(self, ckpt_path, device='cuda', dim=24, num_blocks=(2, 3, 3, 4), num_refinement_blocks=2,
                 prompts_dir=None, camchain_path=None, cam_name='cam1', nuc_gain=None, nuc_offset=None):
        self.device = device
        self.model = Restormer_PPFN(
            inp_channels=1, out_channels=1, dim=dim, num_blocks=list(num_blocks),
            num_refinement_blocks=num_refinement_blocks, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        ).to(device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.eval()

        prompts_dir = prompts_dir or _PROMPTS_DIR
        type_npl = np.load(os.path.join(prompts_dir, 'type7.npy')).astype(np.float32)
        multi_npl = np.load(os.path.join(prompts_dir, 'multi2.npy')).astype(np.float32)
        self.tp_noise = torch.from_numpy(type_npl[2:3]).to(device)   # TYPE_DIC['noise']
        self.mp_composited = torch.from_numpy(multi_npl[1:2]).to(device)  # SCE_DIC['composited']

        self.camchain_path = camchain_path
        self.cam_name = cam_name
        self._remap_cache = {}
        self.nuc_gain = nuc_gain
        self.nuc_offset = nuc_offset

    @classmethod
    def load_nuc(cls, nuc_h5_path):
        """2-point NUC 캘리브레이션 파일(gain/offset dataset)을 읽어
        __init__에 바로 넘길 수 있는 (gain, offset)을 반환한다."""
        with h5py.File(nuc_h5_path, 'r') as f:
            return f['gain'][:].astype(np.float32), f['offset'][:].astype(np.float32)

    def _get_remap(self, width, height):
        key = (width, height)
        if key not in self._remap_cache:
            camchain = load_camchain(self.camchain_path)
            K = cam_intrinsics_matrix(camchain, self.cam_name)
            dist = cam_distortion(camchain, self.cam_name)
            self._remap_cache[key] = build_undistort_remap(K, dist, width, height)
        return self._remap_cache[key]

    @torch.no_grad()
    def _run_ppfn(self, x01):
        x = torch.from_numpy(x01.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        out = self.model([x, self.tp_noise, self.mp_composited]).clamp(0, 1)
        return out[0, 0].cpu().numpy()

    def restore_frame(self, raw16):
        h, w = raw16.shape
        if self.nuc_gain is not None:
            nuc16 = (raw16.astype(np.float32) - self.nuc_offset) / self.nuc_gain
            lo, hi = np.percentile(nuc16, [1, 99])
            input01 = np.clip((nuc16 - lo) / (hi - lo + 1e-8), 0, 1)
        else:
            input01 = (normalize_16bit_to_unit(raw16).astype(np.float32) + 1) / 2

        restored = self._run_ppfn(input01)

        if self.camchain_path is not None:
            map1, map2 = self._get_remap(w, h)
            restored = cv2.remap(restored, map1, map2, cv2.INTER_LINEAR)
        return restored


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='16-bit raw 프레임 한 장을 복원해서 저장한다.')
    p.add_argument('input', help='16-bit grayscale 이미지 경로')
    p.add_argument('output', help='복원 결과(8-bit PNG) 저장 경로')
    p.add_argument('--ckpt', default=os.path.join(_CKPT_DIR, 'ppfn_small_direct_final.pth'))
    p.add_argument('--camchain', default=None, help='(선택) 렌즈 왜곡보정용 camchain.yaml')
    p.add_argument('--nuc-h5', default=None, help='(선택) 2-point NUC 캘리브레이션 h5 (gain/offset)')
    args = p.parse_args()

    raw16 = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if raw16 is None or raw16.dtype != np.uint16:
        raise SystemExit(f'{args.input} did not load as a 16-bit image (got {None if raw16 is None else raw16.dtype})')

    nuc_gain, nuc_offset = (RealtimeRestorer.load_nuc(args.nuc_h5) if args.nuc_h5 else (None, None))
    restorer = RealtimeRestorer(args.ckpt, camchain_path=args.camchain, nuc_gain=nuc_gain, nuc_offset=nuc_offset)
    out01 = restorer.restore_frame(raw16)
    cv2.imwrite(args.output, (np.clip(out01, 0, 1) * 255).astype(np.uint8))
    print(f'wrote {args.output}')
