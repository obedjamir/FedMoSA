from flcore.clients.clientMSA import clientMSA
from flcore.clients.clientMoSA import clientMoSA
from flcore.servers.serverbase import Server
import torch
import os
from ..trainmodel.models import *
def load_brats_model(
    model,
    ckpt_path,
    num_new_experts,
    device="cpu",
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    new_state = {}

    for k, v in ckpt.items():

        # --------------------------------------------------
        # (1) SAM mask decoder
        # --------------------------------------------------
        if k.startswith("sam.mask_decoder."):
            new_state[k] = v
            continue

        # --------------------------------------------------
        # (2) Shared adapters (load as-is, per block)
        # --------------------------------------------------
        if (
            ".shared_adapter_attn.adapter." in k
            or ".shared_adapter_mlp.adapter." in k
        ):
            new_state[k] = v
            continue

        # --------------------------------------------------
        # (3) Conditional adapters: expand expert[0]
        # --------------------------------------------------
        if ".experts." in k:
            # example:
            # sam.image_encoder.blocks.0.adapter_attn.experts.0.down.weight
            prefix, rest = k.split(".experts.")
            expert_idx, suffix = rest.split(".", 1)

            if expert_idx == "0":
                for new_idx in range(num_new_experts):
                    new_key = f"{prefix}.experts.{new_idx}.{suffix}"
                    new_state[new_key] = v.clone()
            continue

    # --------------------------------------------------
    # Load into model
    # --------------------------------------------------
    missing, unexpected = model.load_state_dict(
        new_state, strict=False
    )

    model.to(device)

    print("✓ Loaded SAM decoder")
    print("✓ Loaded shared adapters (per block)")
    print(f"✓ Expanded conditional expert[0] → {num_new_experts} experts")
    print("Unexpected keys:", unexpected)

class FedSAM(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, {self.args.model_name}, num_clients:{args.num_clients}"

        if args.model_name=="mosa":
            clientObj = clientMoSA
        else:
            clientObj = clientMSA

        self.message_hp_dash = self.message_hp.replace(", ", "-")
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp_dash}-{args.goal}-{self.times}-alt.h5")
        self.recovered = False

        self.set_clients(args, clientObj)
        print("Finished creating server and clients.")


    def train(self):
        if self.args.prev_round > 0 and not self.recovered:
            print("Loading Previous Checkpoint...")
            for client in self.clients:
                client.current_round = self.args.prev_round
                state_dict = torch.load(f"FedSAM/{self.args.dataset[0:9]}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}.pth", map_location="cpu")
                client.model.load_state_dict(state_dict)
                client.model.to(self.args.device)
                client.load_train_data()

            if self.args.model_name == "mosa":
                self.receive_models_mosa()
                self.aggregate_parameters_mosa()
                for client in self.clients:
                    client.global_shared_prototypes = self.global_shared_prototypes

            mean_test_acc, best_mean_test_acc = self.evaluate()
            self.save_results(fn=self.hist_result_fn, reset=False)
            self.recovered = True

        # Load Model for BraTS
        #for client in self.clients:
        #    load_model_special(
        #    model=client.model,
        #    ckpt_path=f"FedSAM/liver_seg-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}.pth",
        #    num_new_experts=2,
        #    device=self.args.device,
        #    )

        for i in range(self.global_rounds):
            if i >= self.args.prev_round:
                print(f"\n------------- Round number: [{i+1:3d}/{self.global_rounds}]-------------")
                print(f"==> Training for {len(self.clients)} clients...", flush=True)

                for client in self.clients:
                    client.current_round = i
                    client.train()

                if self.args.model_name == "mosa":
                    self.receive_models_mosa()
                    self.aggregate_parameters_mosa()
                else:
                    self.receive_models()
                    self.aggregate_parameters()

                print("==> Evaluating personalized models...", flush=True)
                self.send_models(i)

                # === Capture the returned accuracies ===
                mean_test_acc, best_mean_test_acc = self.evaluate()
                self.save_results(fn=self.hist_result_fn)
                # === End of capture ===

                # === Add model saving and early stopping logic ===
                if mean_test_acc >= self.best_mean_test_acc:
                    print(f"New best personalized accuracy: {mean_test_acc * 100:.2f}% (Round {i+1})")
                    self.non_improve_rounds = 0

                    for client in self.clients:
                        torch.save(client.model.state_dict(), f"FedSAM/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}.pth")
                else:
                    self.non_improve_rounds += 1
                    print(f"No improvement in personalized accuracy for {self.non_improve_rounds} consecutive round(s).", flush=True)

                for client in self.clients:
                    torch.save(client.model.state_dict(), f"FedSAM/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}-ckpt.pth")
                
        print("==> Evaluating model Accuracy...", flush=True)
        for client in self.clients:
            client.model.load_state_dict(torch.load(f"FedSAM/{self.args.dataset[0:9]}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}.pth", map_location="cpu"))
        self.evaluate(val=False)

        self.save_results(fn=self.hist_result_fn)

