# Copyright (c) 2026, Obed Jamir and Angshuman Paul.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Builds and runs the FedMoSA job in the NVFlare simulator, or exports it for POC / production.

Resource paths are site-local. Every client reads its data folder and SAM checkpoint from the environment
variables FEDMOSA_DATA_DIR and FEDMOSA_SAM_CKPT, and the server reads FEDMOSA_SAM_CKPT for the initial model.
For a simulator run this script sets them from --data-dir and --sam-ckpt. For an exported job each site sets
them before it starts; the paths may differ between sites. The training script receives no paths as arguments,
so paths with spaces are safe on every NVFlare version."""

import argparse
import os

from src.fedmosa_controller import FedMoSAController
from src.model import MoSA

from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner

HERE = os.path.dirname(os.path.abspath(__file__))


DATA_ENV, CKPT_ENV = "FEDMOSA_DATA_DIR", "FEDMOSA_SAM_CKPT"


def site_checkpoint(checkpoint=None):
    """The SAM checkpoint of this site: FEDMOSA_SAM_CKPT if set, else `checkpoint`; must exist when given."""
    path = os.environ.get(CKPT_ENV) or checkpoint
    if path and not os.path.isfile(path):
        raise FileNotFoundError(f"SAM checkpoint '{path}' not found on this site (set {CKPT_ENV} to a local path)")
    return path or None


class InitialModel(MoSA):
    """Server-side initial model; only the trainable (exchanged) parameters are put in the global model.
    The checkpoint is resolved on the machine that builds the model (FEDMOSA_SAM_CKPT, see above)."""

    def __init__(self, checkpoint=None, modalities=("MR", "CT"), reduction=16, decoder_init="random"):
        super().__init__(site_checkpoint(checkpoint), modalities, reduction, decoder_init)
        self.checkpoint = checkpoint  # exported as given; each server resolves its own path

    def state_dict(self, *args, **kwargs):
        return {k: v for k, v in super().state_dict(*args, **kwargs).items() if k in self.trainable_state()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="", help="dataset folder with slices_*.pkl (system/prepare_data.py); "
                    "required for a simulator run, exported jobs read FEDMOSA_DATA_DIR on each site")
    ap.add_argument("--dataset-id", default="0")
    ap.add_argument("--sites", default="site-1,site-2,site-3")
    ap.add_argument("--modalities", default="MR,CT")
    ap.add_argument("--sam-ckpt", default="", help="SAM ViT-B checkpoint (sam_vit_b_01ec64.pth or MedSAM)")
    ap.add_argument(
        "--decoder-init",
        default="random",
        choices=["random", "pretrained"],
        help="mask decoder: trained from a random initialization (default) or from the SAM checkpoint",
    )
    ap.add_argument("--num-rounds", type=int, default=50)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam-proto", type=float, default=0.01)
    ap.add_argument("--workspace", default="/tmp/nvflare/fedmosa")
    ap.add_argument("--gpu", default="")
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="simulator threads (default: one per site, so each client keeps its process and optimizer state)",
    )
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    sites = [s for s in args.sites.split(",") if s]
    mods = [m for m in args.modalities.split(",") if m]
    for name, values in (("--sites", sites), ("--modalities", mods)):
        if not values or any(not v.strip() or any(c.isspace() for c in v) for v in values):
            raise ValueError(f"{name}: comma-separated names without whitespace, got '{getattr(args, name[2:])}'")
    if args.num_rounds < 1 or args.local_epochs < 1 or args.batch_size < 1:
        raise ValueError("--num-rounds, --local-epochs and --batch-size must be positive")
    # site-local resources (see the module docstring); for a simulator run every site uses these paths
    if not args.export_only:
        if not args.data_dir:
            raise ValueError("--data-dir is required for a simulator run")
        os.environ[DATA_ENV] = os.path.abspath(args.data_dir)
        if args.sam_ckpt:
            os.environ[CKPT_ENV] = os.path.abspath(args.sam_ckpt)
    # every configured site must join: the job does not start with fewer sites (a modality could be missing)
    job = FedJob(name="fedmosa", min_clients=len(sites), mandatory_clients=sites)
    ckpt = os.path.abspath(args.sam_ckpt) if args.sam_ckpt else None
    comp_ids = job.to_server(PTModel(InitialModel(ckpt, mods, decoder_init=args.decoder_init)))
    job.to_server(
        FedMoSAController(
            num_clients=len(sites),
            num_rounds=args.num_rounds,
            modalities=mods,
            persistor_id=comp_ids["persistor_id"],
        ),
        id="fedmosa",
    )

    script_args = [
        "--dataset-id", args.dataset_id,
        "--sites", args.sites,
        "--modalities", args.modalities,
        "--local-epochs", str(args.local_epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--lam-proto", str(args.lam_proto),
        "--decoder-init", args.decoder_init,
    ]  # fmt: skip
    if any(any(c.isspace() for c in a) for a in script_args):
        raise ValueError(f"script arguments must not contain whitespace: {script_args}")
    job.to_clients(os.path.join(HERE, "src"))  # ship src/ (model, data, client script) with the client app
    # no paths and no whitespace in the arguments, so the runner's whitespace splitting keeps every boundary
    job.to_clients(ScriptRunner(script="src/client.py", script_args=" ".join(script_args)))

    if args.export_only:
        job.export_job(os.path.join(args.workspace, "jobs"))
    else:
        job.simulator_run(args.workspace, clients=sites, threads=args.threads or len(sites), gpu=args.gpu or None)


if __name__ == "__main__":
    main()
