import torch
import os
import numpy as np
import h5py
import copy
import random
import logging
import shutil
default_max_workers = os.cpu_count()
print(f"Default number of threads: {default_max_workers}")

class Server(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)

        self.num_clients = args.num_clients
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.best_mean_test_acc = -1.0
        self.clients = []

        self.uploaded_weights = []
        self.uploaded_ids = [i for i in range(args.num_clients)]
        self.uploaded_models = []
        self.global_shared_prototypes  = None

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_test_loss = []
        self.rs_train_loss = []
        self.clients_test_accs = []
        self.clients_test_aucs = []
        self.clients_test_loss = []

        self.times = times

        self.set_seed(32)
        self.set_path(args)

        dir_alpha = 0.3

        self.actual_dataset = f"{self.dataset}-{self.num_clients}clients_alpha{dir_alpha:.1f}"
        logger_fn = os.path.join(args.log_dir, f"{args.algorithm}-{self.actual_dataset}.log")
        self.set_logger(save=True, fn=logger_fn)

        self.non_improve_rounds = 0
        self.patience = 5

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def set_logger(self, save=False, fn=None):
        if save:
            fn = "testlog.log" if fn == None else fn
            logging.basicConfig(
                filename=fn,
                filemode="a",
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                level=logging.DEBUG
            )
        else:
            logging.basicConfig(
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                level=logging.DEBUG
            )

    def set_path(self, args):
        self.hist_dir = args.hist_dir
        self.log_dir = args.log_dir
        self.ckpt_dir = args.ckpt_dir
        if not os.path.exists(args.hist_dir):
            os.makedirs(args.hist_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

    def reset_directory(self, dir_path):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    def set_clients(self, args, clientObj):
        dataset_id = str(self.times)

        for i in range(self.num_clients):
            client = clientObj(args, id=i, dataset_id=dataset_id, shared_model=self.global_model)
            self.clients.append(client)

    def send_models(self, round):
        for client in self.clients:
            client.set_parameters(self.global_model)
            client.global_shared_prototypes = self.global_shared_prototypes

    def receive_models(self):
        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []
        self.sample_counts = []
        for client in self.clients:
            self.sample_counts.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)
        for i, samples in enumerate(self.sample_counts):
            self.uploaded_weights.append(samples / tot_samples)

    def receive_models_mosa(self):
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.sample_counts = []
        self.uploaded_prototypes = []

        tot_samples = 0

        # ---- collect client info ----
        for client in self.clients:
            mod_counts = client.mod_sample_count.copy()
            self.sample_counts.append(mod_counts)

            client_total = sum(mod_counts.values())
            tot_samples += client_total

            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

            self.uploaded_prototypes.append(
                client.model.get_all_shared_prototypes()
            )
            client.model.flush_all_shared_prototypes()

        # ---- compute aggregation weights ----
        for mod_counts in self.sample_counts:
            client_total = sum(mod_counts.values())
            self.uploaded_weights.append(client_total / tot_samples)

    def add_parameters(self, w, client_model):
        server_params = dict(self.global_model.named_parameters())
        client_params = dict(client_model.named_parameters())

        for name in server_params:
            if name in client_params:
                if server_params[name].shape == client_params[name].shape:
                    server_params[name].data += client_params[name].data * w


    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        self.global_model = copy.deepcopy(self.uploaded_models[0])

        global_params = dict(self.global_model.named_parameters())

        common_keys = set(global_params.keys())
        for client_model in self.uploaded_models:
            common_keys &= set(dict(client_model.named_parameters()).keys())

        for k in common_keys:
            global_params[k].data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def compute_modality_weights(self):
        # modality-specific weights: one list per modality
        self.modality_weights = {m: [] for m in self.args.mod_list}
        self.shared_weights = []

        raw_modalities = []      # |M_k|
        raw_shared = []          # |D_k|
        raw_counts = {m: [] for m in self.args.mod_list}

        # --------------------------------------------------
        # collect raw statistics
        # --------------------------------------------------
        for mod_counts in self.sample_counts:
            total = sum(mod_counts.values())
            raw_shared.append(total)

            # number of active modalities at client
            num_modalities = sum(int(mod_counts[m] > 0) for m in self.args.mod_list)
            raw_modalities.append(num_modalities)

            for m in self.args.mod_list:
                raw_counts[m].append(mod_counts[m])

        # --------------------------------------------------
        # modality-specific weights (per modality)
        # --------------------------------------------------
        for m in self.args.mod_list:
            sum_m = sum(raw_counts[m])
            for i in range(len(self.sample_counts)):
                self.modality_weights[m].append(
                    raw_counts[m][i] / sum_m if sum_m > 0 else 0.0
                )

        # --------------------------------------------------
        # shared aggregation weights: |M_k| * |D_k|
        # --------------------------------------------------
        raw_shared_weights = [
            raw_shared[i] * raw_modalities[i]
            for i in range(len(raw_shared))
        ]

        sum_shared_weights = sum(raw_shared_weights)

        self.shared_weights = [
            w / sum_shared_weights if sum_shared_weights > 0 else 0.0
            for w in raw_shared_weights
        ]

        # sanity check
        assert abs(sum(self.shared_weights) - 1.0) < 1e-6

    def add_parameters_modality_aware(
        self,
        client_model,
        w_shared,
        w_mod,     # e.g. {"CT": 0.3, "MR": 0.7}
        w_data,
    ):
        print(f"data weight: {w_data}")
        print(f"shared weight: {w_shared}")

        for (name_s, server_param), (name_c, client_param) in zip(
            self.global_model.named_parameters(),
            client_model.named_parameters(),
        ):
            assert name_s == name_c

            # ---------------- shared adapters ----------------
            if "shared_adapter" in name_s:
                server_param.data += client_param.data * w_shared

            # ---------------- modality-specific experts ----------------
            elif "adapter_attn.experts" in name_s or "adapter_mlp.experts" in name_s:
                applied = False

                for mod_idx, mod_name in enumerate(self.args.mod_list):
                    if f"experts.{mod_idx}" in name_s:
                        w = w_mod.get(mod_name, 0.0)
                        server_param.data += client_param.data * w
                        applied = True
                        break

                if not applied:
                    server_param.data += client_param.data * w_shared

            # ---------------- fallback ----------------
            else:
                server_param.data += client_param.data * w_shared

    def aggregate_shared_prototypes(self):
        num_clients = len(self.uploaded_prototypes)
        assert num_clients > 0

        num_blocks = len(self.uploaded_prototypes[0])
        global_shared_prototypes = []

        for blk_idx in range(num_blocks):
            attn_sum, mlp_sum = None, None

            for client_idx in range(num_clients):
                w = self.shared_weights[client_idx]
                proto_blk = self.uploaded_prototypes[client_idx][blk_idx]

                if proto_blk["attn_shared"] is not None:
                    attn_sum = (
                        proto_blk["attn_shared"] * w
                        if attn_sum is None
                        else attn_sum + proto_blk["attn_shared"] * w
                    )

                if proto_blk["mlp_shared"] is not None:
                    mlp_sum = (
                        proto_blk["mlp_shared"] * w
                        if mlp_sum is None
                        else mlp_sum + proto_blk["mlp_shared"] * w
                    )

            global_shared_prototypes.append({
                "attn_shared": attn_sum,
                "mlp_shared": mlp_sum,
            })

        self.global_shared_prototypes = global_shared_prototypes

    def aggregate_parameters_mosa(self):
        assert len(self.uploaded_models) > 0

        # initialize global model
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        # compute modality-aware weights
        self.compute_modality_weights()

        # --------------------------------------------------
        # aggregate model parameters
        # --------------------------------------------------
        for i, client_model in enumerate(self.uploaded_models):
            # per-client modality weights dict
            w_mod = {
                m: self.modality_weights[m][i]
                for m in self.args.mod_list
            }

            self.add_parameters_modality_aware(
                client_model,
                w_shared=self.shared_weights[i],
                w_mod=w_mod,
                w_data=self.uploaded_weights[i],
            )

        # --------------------------------------------------
        # aggregate shared prototypes
        # --------------------------------------------------
        self.aggregate_shared_prototypes()

    def reset_records(self):
        self.best_mean_test_acc = 0.0
        self.clients_test_accs = []
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_test_loss = []

    def train_new_clients(self, epochs=20):
        self.global_model = self.global_model.to(self.device)
        self.clients = self.new_clients
        self.reset_records()
        for c in self.clients:
            c.model = copy.deepcopy(self.global_model)
        self.evaluate()
        for epoch_idx in range(epochs):
            for c in self.clients:
                c.standard_train()
            print(f"==> New clients epoch: [{epoch_idx+1:2d}/{epochs}] | Evaluating local models...", flush=True)
            self.evaluate()
        print(f"==> Best mean global accuracy: {self.best_mean_test_acc*100:.2f}%", flush=True)
        self.save_results(fn=self.hist_result_fn)
        message_res = f"\tnew_clients_test_acc:{self.best_mean_test_acc:.6f}"
        logging.info(self.message_hp + message_res)

    def save_global_model(self, model_path=None, state=None):
        if model_path is None:
            model_path = os.path.join("models", self.dataset)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        if state is None:
            torch.save({"global_model": self.global_model.cpu().state_dict()}, model_path)
        else:
            torch.save(state, model_path)

    def save_results(self, fn=None, reset=True):
        if fn is None:
            algo = self.dataset + "_" + self.algorithm
            result_path = self.hist_dir

        if len(self.rs_test_acc):
            if fn is None:
                algo = algo + "_" + self.goal + "_" + str(self.times+1)
                file_path = os.path.join(result_path, "{}.h5".format(algo))
            else:
                file_path = fn
            print("File path: " + file_path)

            if not reset and os.path.exists(file_path):
                # Load existing history
                with h5py.File(file_path, 'r') as hf:
                    prev_rs_test_acc = hf['rs_test_acc'][:]
                    prev_rs_test_auc = hf['rs_test_auc'][:]
                    prev_rs_test_loss = hf['rs_test_loss'][:]
                    prev_clients_test_accs = hf['clients_test_accs'][:]
                    prev_clients_test_aucs = hf['clients_test_aucs'][:]
                    prev_clients_test_loss = hf['clients_test_loss'][:]

                # Append new history to old
                self.rs_test_acc = np.concatenate([prev_rs_test_acc, self.rs_test_acc]).tolist()
                self.rs_test_auc = np.concatenate([prev_rs_test_auc, self.rs_test_auc]).tolist()
                self.rs_test_loss = np.concatenate([prev_rs_test_loss, self.rs_test_loss]).tolist()
                self.clients_test_accs = np.concatenate([prev_clients_test_accs, self.clients_test_accs]).tolist()
                self.clients_test_aucs = np.concatenate([prev_clients_test_aucs, self.clients_test_aucs]).tolist()
                self.clients_test_loss = np.concatenate([prev_clients_test_loss, self.clients_test_loss]).tolist()
                print("Previous History Loaded...")

            # Save (overwrite with full history)
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_test_loss', data=self.rs_test_loss)
                hf.create_dataset('clients_test_accs', data=self.clients_test_accs)
                hf.create_dataset('clients_test_aucs', data=self.clients_test_aucs)
                hf.create_dataset('clients_test_loss', data=self.clients_test_loss)

    def test_metrics(self, temp_model=None, val=True):
        """ A personalized evaluation scheme (test_acc's do not average based on num_samples) """
        test_accs, test_aucs, test_losses, test_nums = [], [], [], []
        ids = [c.id for c in self.clients]
        filtered_ids = ids
        for c in self.clients:
            if c.id in filtered_ids:
                test_acc, test_auc, test_loss, test_num = c.test_metrics(temp_model, val=val, ood=True)
                test_accs.append(test_acc)
                test_aucs.append(test_auc)
                test_losses.append(test_loss)
                test_nums.append(test_num)
        return ids, test_accs, test_aucs, test_losses, test_nums

    def evaluate(self, temp_model=None, mode="personalized", val=True):
        ids, test_accs, test_aucs, test_losses, test_nums = self.test_metrics(temp_model, val=val)
        self.clients_test_accs.append(copy.deepcopy(test_accs))
        self.clients_test_aucs.append(copy.deepcopy(test_aucs))
        self.clients_test_loss.append(copy.deepcopy(test_losses))
        if mode == "personalized":
            mean_test_acc, mean_test_auc, mean_test_loss = np.mean(test_accs), np.mean(test_aucs), np.mean(test_losses)
        elif mode == "global":
            mean_test_acc, mean_test_auc, mean_test_loss = np.average(test_accs, weights=test_nums), np.average(test_aucs, weights=test_nums), np.average(test_losses, weights=test_nums)
        else:
            raise NotImplementedError

        print(test_accs)
        print(test_aucs)
        self.best_mean_test_acc = max(mean_test_acc, self.best_mean_test_acc)
        self.rs_test_acc.append(mean_test_acc)
        self.rs_test_auc.append(mean_test_auc)
        self.rs_test_loss.append(mean_test_loss)
        if val:
            print(f"==> val_loss: {mean_test_loss:.5f} | mean_val_IoU: {mean_test_acc*100:.2f}% | best_IoU: {self.best_mean_test_acc*100:.2f}%\n")
        else:
            print(f"==> test_loss: {mean_test_loss:.5f} | mean_test_IoU: {mean_test_acc*100:.2f}% | best_IoU: {self.best_mean_test_acc*100:.2f}%\n")
        return mean_test_acc, self.best_mean_test_acc
