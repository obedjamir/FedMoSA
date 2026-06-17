from flcore.clients.clientnnUNET import clientnnUNET
from flcore.servers.serverbase import Server
import torch
import os
from flcore.servers.FFE import FederatedFingerprintAggregator

class FednnUNET(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.message_hp = f"{args.algorithm}, lr:{args.local_learning_rate:.5f}, {self.args.model_name}, num_clients:{args.num_clients}"
        clientObj = clientnnUNET
        self.message_hp_dash = self.message_hp.replace(", ", "-")
        self.hist_result_fn = os.path.join(args.hist_dir, f"{self.actual_dataset}-{self.message_hp_dash}-{args.goal}-{self.times}-alt.h5")
        self.recovered = False

        self.set_clients(args, clientObj)
        if self.args.afa==0:
            self.FFE = FederatedFingerprintAggregator()
            for client in self.clients:
                assert client.fingerprint is not None
                self.FFE.add_client_fingerprint(client.fingerprint)

            self.global_fingerprint = self.FFE.aggregate()
            for client in self.clients:
                client.initialize_model(self.global_fingerprint)
            print("Finished creating server and clients.")


    def train(self):
        '''
        if self.args.prev_round > 0 and not self.recovered:
            print("Loading Previous Checkpoint...")
            for client in self.clients:
                client.current_round = self.args.prev_round
                state_dict = torch.load(f"FednnUNET/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}-ckpt.pth", map_location="cpu")
                client.model.load_state_dict(state_dict)
                client.model.to(self.args.device)

            mean_test_acc, best_mean_test_acc = self.evaluate()
            self.save_results(fn=self.hist_result_fn, reset=False)
            self.recovered = True

        for i in range(self.global_rounds):
            if i >= self.args.prev_round:
                print(f"\n------------- Round number: [{i+1:3d}/{self.global_rounds}]-------------")
                print(f"==> Training for {len(self.clients)} clients...", flush=True)

                for client in self.clients:
                    client.current_round = i
                    client.train()

                #self.receive_models()
                #self.aggregate_parameters()

                print("==> Evaluating personalized models...", flush=True)
                #self.send_models(i)

                # === Capture the returned accuracies ===
                mean_test_acc, best_mean_test_acc = self.evaluate()
                self.save_results(fn=self.hist_result_fn)
                # === End of capture ===

                # === Add model saving and early stopping logic ===
                if mean_test_acc >= self.best_mean_test_acc:
                    print(f"New best personalized accuracy: {mean_test_acc * 100:.2f}% (Round {i+1})")
                    self.non_improve_rounds = 0

                    for client in self.clients:
                        torch.save(client.model.state_dict(), f"FednnUNET/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}.pth")
                        torch.save(client.model.state_dict(), f"FednnUNET/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}-ckpt.pth")
                else:
                    self.non_improve_rounds += 1
                    print(f"No improvement in personalized accuracy for {self.non_improve_rounds} consecutive round(s).", flush=True)
                    for client in self.clients:
                        torch.save(client.model.state_dict(), f"FednnUNET/{self.args.dataset}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}-ckpt.pth")
        '''
        print("==> Evaluating model Accuracy...", flush=True)
        for client in self.clients:
            client.model.load_state_dict(torch.load(f"FednnUNET/{self.args.dataset[0:9]}-{self.args.model_name}-num_clients:{self.args.num_clients}-{client.id}-{self.args.goal}-{self.times}.pth", map_location="cpu"))
        self.evaluate(val=False)

        #self.save_results(fn=self.hist_result_fn)

