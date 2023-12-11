import argparse
import logging

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from load_data import *
from models import *
from semi_utils import bootstrapping, boot_update_triple
from utils import *
import json


class Experiment:
    def __init__(self, args):
        self.mapping_ins_emb = None
        self.enh_ins_emb = None
        self.rel_embeddings = None
        self.ins_embeddings = None
        self.save = args.save
        self.save_prefix = "%s_%s" % (args.data_dir.split("/")[-1], args.log)

        self.hiddens = list(map(int, args.hiddens.split(",")))
        self.heads = list(map(int, args.heads.split(",")))

        self.args = args
        self.args.encoder = args.encoder.lower()
        self.args.decoder = args.decoder.lower().split(",")
        self.args.sampling = args.sampling.split(",")
        self.args.k = list(map(int, args.k.split(",")))
        self.args.margin = [float(x) if "-" not in x else list(map(float, x.split("-"))) for x in
                            args.margin.split(",")]
        self.args.alpha = list(map(float, args.alpha.split(",")))
        assert len(self.args.decoder) >= 1
        assert len(self.args.decoder) == len(self.args.sampling) and \
            len(self.args.sampling) == len(self.args.k) and \
            len(self.args.k) == len(self.args.alpha)

        self.cached_sample = {}
        self.best_result = ()

    def evaluate(self, it, test, ins_emb, mapping_emb=None):
        """
        Evaluate the model on test set.
        Args:
            it (int): An iteration counter, typically used for logging and tensorboard summaries.
            test (array): A 2D numpy array of test data, where each row represents a (left, right) entity pair.
            ins_emb (Tensor): Embeddings for instances (entities).
            mapping_emb (Tensor, optional): An alternative set of embeddings to be used for the left entities.
            If not provided, `ins_emb` is used for both left and right entities.

        Returns:
            A tuple containing the following metrics for both l2r and r2l tasks
                - Accuracy at different top-k ranks (as a numpy array).
                - Mean rank.
                - Mean reciprocal rank.
        """

        def log_results(direction, ranks, accuracy, mean, mrr, time_taken):
            logger.info(
                f"{direction}: acc of top {ranks} = {accuracy.tolist()}, mr = {mean:.3f}, mrr = {mrr:.3f}, time = {time_taken:.4f} s ")

        start_time = time.time()
        ranks_to_check = [1, 3, 5, 10]
        embeddings_left = mapping_emb[test[:, 0]
                                      ] if mapping_emb is not None else ins_emb[test[:, 0]]
        embeddings_right = ins_emb[test[:, 1]]

        dist_metric = self.args.test_dist
        sim_distance = - sim(embeddings_left, embeddings_right, metric=dist_metric, normalize=True,
                             csls_k=self.args.csls)
        if self.args.rerank:
            dist_sort_indices = np.argsort(
                np.argsort(sim_distance, axis=1), axis=1)
            transposed_indices = np.argsort(
                np.argsort(sim_distance.T, axis=1), axis=1)
            sim_distance = dist_sort_indices + transposed_indices.T

        split_tasks = div_list(np.array(range(len(test))), 10)
        process_pool = multiprocessing.Pool(processes=len(split_tasks))
        results = []
        for task in split_tasks:
            results.append(process_pool.apply_async(multi_cal_rank, (
                task, sim_distance[task, :], sim_distance[:, task], ranks_to_check, self.args)))
        process_pool.close()
        process_pool.join()

        left_to_right_acc, right_to_left_acc = np.zeros(
            len(ranks_to_check)), np.zeros(len(ranks_to_check))
        left_to_right_mean, right_to_left_mean = 0., 0.
        left_to_right_mrr, right_to_left_mrr = 0., 0.
        for result in results:
            l2r_acc, l2r_mean, l2r_mrr, r2l_acc, r2l_mean, r2l_mrr = result.get()
            left_to_right_acc += l2r_acc
            left_to_right_mean += l2r_mean
            left_to_right_mrr += l2r_mrr
            right_to_left_acc += r2l_acc
            right_to_left_mean += r2l_mean
            right_to_left_mrr += r2l_mrr

        total_tests = len(test)
        left_to_right_mean /= total_tests
        right_to_left_mean /= total_tests
        left_to_right_mrr /= total_tests
        right_to_left_mrr /= total_tests
        left_to_right_acc /= total_tests
        right_to_left_acc /= total_tests

        elapsed_time = time.time() - start_time
        log_results('l2r', ranks_to_check, left_to_right_acc,
                    left_to_right_mean, left_to_right_mrr, elapsed_time)
        log_results('r2l', ranks_to_check, right_to_left_acc,
                    right_to_left_mean, right_to_left_mrr, elapsed_time)

        for index, rank in enumerate(ranks_to_check):
            writer.add_scalar(
                f"l2r_HitsAt{rank}", left_to_right_acc[index], it)
            writer.add_scalar(
                f"r2l_HitsAt{rank}", right_to_left_acc[index], it)
        writer.add_scalar("l2r_MeanRank", left_to_right_mean, it)
        writer.add_scalar("l2r_MeanReciprocalRank", left_to_right_mrr, it)
        writer.add_scalar("r2l_MeanRank", right_to_left_mean, it)
        writer.add_scalar("r2l_MeanReciprocalRank", right_to_left_mrr, it)

        return left_to_right_acc, left_to_right_mean, left_to_right_mrr, right_to_left_acc, right_to_left_mean, right_to_left_mrr

    def init_emb(self):
        """
        Initialize embeddings of entities and relations.
        """
        scaling_entity, scaling_relation = 1, 1
        if not self.args.encoder:
            decoder_type = self.args.decoder[0]
            if decoder_type == "rotate":
                scaling_relation /= 2
            elif decoder_type == "hake":
                scaling_relation = (scaling_relation / 2) * 3
            elif decoder_type == "transh":
                scaling_relation *= 2
            elif decoder_type == "transr":
                scaling_relation = self.hiddens[0] + 1

        self.ins_embeddings = nn.Embedding(
            d.ins_num, self.hiddens[0] * scaling_entity).to(device)
        self.rel_embeddings = nn.Embedding(d.rel_num, int(
            self.hiddens[0] * scaling_relation)).to(device)

        if decoder_type in ["rotate", "hake"]:
            entity_range = (
                self.args.margin[0] + 2.0) / (self.hiddens[0] * scaling_entity)
            relation_range = (
                self.args.margin[0] + 2.0) / (self.hiddens[0] * scaling_relation)
            nn.init.uniform_(self.ins_embeddings.weight, -
                             entity_range, entity_range)
            nn.init.uniform_(self.rel_embeddings.weight, -
                             relation_range, relation_range)
            if decoder_type == "hake":
                mid_dim = self.hiddens[0] // 2
                nn.init.ones_(
                    self.rel_embeddings.weight[:, mid_dim:2 * mid_dim])
                nn.init.zeros_(
                    self.rel_embeddings.weight[:, 2 * mid_dim:3 * mid_dim])
        else:
            nn.init.xavier_normal_(self.ins_embeddings.weight)
            nn.init.xavier_normal_(self.rel_embeddings.weight)

        if decoder_type in ["alignea", "mtranse_align", "transedge"]:
            self.ins_embeddings.weight.data = F.normalize(
                self.ins_embeddings.weight, p=2, dim=1)
            self.rel_embeddings.weight.data = F.normalize(
                self.rel_embeddings.weight, p=2, dim=1)
        elif decoder_type == "transr":
            self.ins_embeddings.weight.data = torch.from_numpy(
                np.load(f"{self.args.pre}_ins.npy")).to(device)
            self.rel_embeddings.weight[:, :self.hiddens[0]].data = torch.from_numpy(
                np.load(f"{self.args.pre}_rel.npy")).to(device)

        self.enh_ins_emb = self.ins_embeddings.weight.detach().cpu().numpy()
        self.mapping_ins_emb = None

    def train_and_eval(self):
        # Initialize embeddings
        self.init_emb()

        # Set up the graph encoder if specified
        graph_encoder = None
        if self.args.encoder:
            graph_encoder = Encoder(self.args.encoder, self.hiddens, self.heads + [1], activation=F.elu,
                                    feat_drop=self.args.feat_drop, attn_drop=self.args.attn_drop, negative_slope=0.2,
                                    bias=False).to(device)
            logger.info(graph_encoder)

        # Initialize knowledge decoders
        knowledge_decoder = []
        for idx, decoder_name in enumerate(self.args.decoder):
            knowledge_decoder.append(Decoder(decoder_name, params={
                "e_num": d.ins_num,
                "r_num": d.rel_num,
                "dim": self.hiddens[-1],
                "feat_drop": self.args.feat_drop,
                "train_dist": self.args.train_dist,
                "sampling": self.args.sampling[idx],
                "k": self.args.k[idx],
                "margin": self.args.margin[idx],
                "alpha": self.args.alpha[idx],
                "boot": self.args.bootstrap,
                # Additional parameters can be passed to Decoder
            }).to(device))
        logger.info(knowledge_decoder)

        # Prepare optimization parameters
        params = nn.ParameterList(
            [self.ins_embeddings.weight, self.rel_embeddings.weight] + [p for k_d in knowledge_decoder for p in
                                                                        list(k_d.parameters())] + (
                list(graph_encoder.parameters()) if self.args.encoder else []))
        opt = optim.Adagrad(params, lr=self.args.lr, weight_decay=self.args.wd)

        # Learning rate scheduler
        if self.args.dr:
            scheduler = optim.lr_scheduler.ExponentialLR(opt, self.args.dr)
        logger.info(params)
        logger.info(opt)

        # Training loop
        for it in range(0, self.args.epoch):
            for idx, k_d in enumerate(knowledge_decoder):
                # Skip training for certain conditions
                if k_d.name == "align" and not d.ill_train_idx:
                    continue
                t_ = time.time()

                # Train the model for one epoch
                if k_d.print_name.startswith("["):
                    loss = self.train_1_epoch(it, opt, None, k_d, d.ins_G_edges_idx, d.triple_idx, d.ill_train_idx,
                                              [d.kg1_ins_ids,
                                                  d.kg2_ins_ids], d.boot_triple_idx, d.boot_pair_dix,
                                              self.ins_embeddings.weight, self.rel_embeddings.weight)
                else:
                    loss = self.train_1_epoch(it, opt, graph_encoder, k_d, d.ins_G_edges_idx, d.triple_idx,
                                              d.ill_train_idx, [
                                                  d.kg1_ins_ids, d.kg2_ins_ids], d.boot_triple_idx,
                                              d.boot_pair_dix, self.ins_embeddings.weight, self.rel_embeddings.weight)

                # Update mapping embeddings
                if hasattr(k_d, "mapping"):
                    self.mapping_ins_emb = k_d.mapping(
                        self.ins_embeddings.weight).cpu().detach().numpy()
                loss_name = "loss_" + \
                    k_d.print_name.replace("[", "_").replace("]", "_")
                writer.add_scalar(loss_name, loss, it)
                logger.info(
                    f"epoch: {it}\t{loss_name}: {loss:.8f}\ttime: {int(time.time() - t_)}s")

                # Adjust learning rate if needed
                if self.args.dr:
                    scheduler.step()

                # Evaluate the model periodically
                if (it + 1) % self.args.check == 0:
                    logger.info("Start validating...")
                    with torch.no_grad():
                        # Combine embeddings if needed
                        if graph_encoder and graph_encoder.name == "naea":
                            beta = self.args.margin[-1]
                            emb = beta * self.enh_ins_emb + (
                                1 - beta) * self.ins_embeddings.weight.cpu().detach().numpy()
                        else:
                            emb = self.enh_ins_emb

                        # Perform evaluation
                        result = self.evaluate(it, d.ill_val_idx if d.ill_val_idx else d.ill_test_idx, emb,
                                               self.mapping_ins_emb)

                    # Check for early stopping
                    if self.args.early and self.best_result and result[0][0] < self.best_result[0][0]:
                        logger.info("Early stop, best result:")
                        # Log best results
                        acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l = self.best_result
                        logger.info(
                            f"l2r: acc of top {[1, 3, 5, 10]} = {acc_l2r}, mr = {mean_l2r:.3f}, mrr = {mrr_l2r:.3f}")
                        logger.info(
                            f"r2l: acc of top {[1, 3, 5, 10]} = {acc_r2l}, mr = {mean_r2l:.3f}, mrr = {mrr_r2l:.3f}")
                        break
                    self.best_result = result

                # Bootstrapping for entity alignment
                if self.args.bootstrap and it >= self.args.start_bp and (it + 1) % self.args.update == 0:
                    with torch.no_grad():
                        if graph_encoder and graph_encoder.name == "naea":
                            beta = self.args.margin[-1]
                            emb = beta * self.enh_ins_emb + (
                                1 - beta) * self.ins_embeddings.weight.cpu().detach().numpy()
                        else:
                            emb = self.enh_ins_emb

                    # Update bootstrapping parameters
                    d.labeled_alignment, A, B = bootstrapping(
                        ref_sim_mat=sim(emb[d.ill_test_idx[:, 0]], emb[d.ill_test_idx[:, 1]],
                                        metric=self.args.test_dist,
                                        normalize=True, csls_k=0),
                        ref_ent1=d.ill_test_idx[:, 0].tolist(),
                        ref_ent2=d.ill_test_idx[:, 1].tolist(),
                        labeled_alignment=d.labeled_alignment, th=self.args.threshold, top_k=10, is_edit=False)
                    if d.labeled_alignment:
                        d.boot_triple_idx = boot_update_triple(
                            A, B, d.triple_idx)
                        d.boot_pair_dix = [(A[i], B[i]) for i in range(len(A))]
                        logger.info(
                            f"Bootstrapping: + {len(A)} ills, {len(d.boot_triple_idx)} triples.")

        # Save embeddings
        if self.save:
            if not os.path.exists(self.save):
                os.makedirs(self.save)
            time_str = f"{self.save_prefix}_{time.strftime('%Y%m%d-%H%M', time.gmtime())}"
            if graph_encoder:
                with torch.no_grad():
                    graph_encoder.eval()
                    edges = torch.LongTensor(d.ins_G_edges_idx).to(device)
                    enh_emb = graph_encoder(edges, self.ins_embeddings.weight)
                    np.save(f"{self.save}/{time_str}_enh_ins.npy",
                            enh_emb.cpu().detach().numpy())
            np.save(f"{self.save}/{time_str}_ins.npy",
                    self.ins_embeddings.weight.cpu().detach().numpy())
            np.save(f"{self.save}/{time_str}_rel.npy",
                    self.rel_embeddings.weight.cpu().detach().numpy())
            logger.info("Embeddings saved!")

    def train_1_epoch(self, it, opt, encoder, decoder, edges, triples, ills, ids, boot_triples, boot_pairs, ins_emb,
                      rel_emb):
        """
        Train the model for one epoch.
        """
        # Set encoder and decoder to training mode
        if encoder:
            encoder.train()
        decoder.train()

        # Initialize list to store losses
        losses = []

        # Update the cached sample if necessary
        if "pos_" + decoder.print_name not in self.cached_sample or it % self.args.update == 0:
            if decoder.name in ["align", "mtranse_align", "n_r_align"]:
                # Cache positive samples for alignment models
                self.cached_sample[
                    "pos_" + decoder.print_name] = ills.tolist() + boot_pairs if decoder.boot else ills.tolist()
                self.cached_sample["pos_" + decoder.print_name] = np.array(
                    self.cached_sample["pos_" + decoder.print_name])
            else:
                # Cache positive samples for other models
                self.cached_sample["pos_" + decoder.print_name] = triples + \
                    boot_triples if decoder.boot else triples
            np.random.shuffle(self.cached_sample["pos_" + decoder.print_name])

        # Get the training batch
        train = self.cached_sample["pos_" + decoder.print_name]
        train_batch_size = len(
            train) if self.args.train_batch_size == -1 else self.args.train_batch_size

        # Iterate over batches
        for i in range(0, len(train), train_batch_size):
            pos_batch = train[i:i + train_batch_size]

            # Update negative samples if necessary
            if (decoder.print_name + str(
                    i) not in self.cached_sample or it % self.args.update == 0) and decoder.sampling_method:
                self.cached_sample[decoder.print_name + str(i)] = decoder.sampling_method(pos_batch, triples, ills, ids,
                                                                                          decoder.k, params={
                                                                                              "emb": self.enh_ins_emb,
                                                                                              "metric": self.args.test_dist,
                                                                                          })

            # Prepare positive and negative samples
            if decoder.sampling_method:
                neg_batch = self.cached_sample[decoder.print_name + str(i)]
                neg = torch.LongTensor(neg_batch).to(device)
                pos = torch.LongTensor(pos_batch).repeat(decoder.k * 2, 1).to(device) if neg.size(0) > len(
                    pos_batch) * decoder.k else torch.LongTensor(pos_batch).to(device)
            else:
                pos = torch.LongTensor(pos_batch).to(device)

            # Forward pass through the encoder if present
            if encoder:
                use_edges = torch.LongTensor(edges).to(device)
                enh_emb = encoder(
                    use_edges, ins_emb, rel_emb[d.r_ij_idx] if encoder.name == "naea" else None)
            else:
                enh_emb = ins_emb

            # Update enhanced instance embeddings
            self.enh_ins_emb = enh_emb[
                0].cpu().detach().numpy() if encoder and encoder.name == "naea" else enh_emb.cpu().detach().numpy()

            # Adjust relation embeddings for specific decoders
            if decoder.name == "n_r_align":
                rel_emb = ins_emb

            # Calculate loss and backpropagate
            opt.zero_grad()
            if decoder.sampling_method:
                pos_score = decoder(enh_emb, rel_emb, pos)
                neg_score = decoder(enh_emb, rel_emb, neg)
                target = torch.ones(neg_score.size()).to(device)
                loss = decoder.loss(pos_score, neg_score,
                                    target) * decoder.alpha
            else:
                loss = decoder(enh_emb, rel_emb, pos) * decoder.alpha
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Return the average loss
        return np.mean(losses)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/DBP15K/zh_en", required=False,
                        help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--rate", type=float, default=0.3,
                        help="training set rate")
    parser.add_argument("--val", type=float, default=0.0,
                        help="valid set rate")
    parser.add_argument("--save", default="",
                        help="the output dictionary of the model and embedding")
    parser.add_argument("--pre", default="",
                        help="pre-train embedding dir (only use in transr)")
    parser.add_argument("--cuda", action="store_true",
                        default=True, help="whether to use cuda or not")
    parser.add_argument("--log", type=str, default="tensorboard_log",
                        nargs="?", help="where to save the log")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--epoch", type=int, default=1000,
                        help="number of epochs to train")
    parser.add_argument("--check", type=int, default=5, help="check point")
    parser.add_argument("--update", type=int, default=5,
                        help="number of epoch for updating negtive samples")
    parser.add_argument("--train_batch_size", type=int,
                        default=-1, help="train batch_size (-1 means all in)")
    parser.add_argument("--early", action="store_true", default=False,
                        help="whether to use early stop")  # Early stop when the Hits@1 score begins to drop on the validation sets, checked every 10 epochs.
    parser.add_argument("--share", action="store_true",
                        default=False, help="whether to share ill emb")
    parser.add_argument("--swap", action="store_true",
                        default=False, help="whether to swap ill in triple")

    parser.add_argument("--bootstrap", action="store_true",
                        default=False, help="whether to use bootstrap")
    parser.add_argument("--start_bp", type=int, default=9,
                        help="epoch of starting bootstrapping")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="threshold of bootstrap alignment")

    parser.add_argument("--encoder", type=str, default="GCN-Align",
                        nargs="?", help="which encoder to use: . max = 1")
    parser.add_argument("--hiddens", type=str, default="100,100,100",
                        help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
    parser.add_argument("--heads", type=str, default="1,1",
                        help="heads in each gat layer, splitted with comma")
    parser.add_argument("--attn_drop", type=float, default=0,
                        help="dropout rate for gat layers")

    parser.add_argument("--decoder", type=str, default="Align",
                        nargs="?", help="which decoder to use: . min = 1")
    parser.add_argument("--sampling", type=str, default="N",
                        help="negtive sampling method for each decoder")
    parser.add_argument("--k", type=str, default="25",
                        help="negtive sampling number for each decoder")
    parser.add_argument("--margin", type=str, default="1",
                        help="margin for each margin based ranking loss (or params for other loss function)")
    parser.add_argument("--alpha", type=str, default="1",
                        help="weight for each margin based ranking loss")
    parser.add_argument("--feat_drop", type=float, default=0,
                        help="dropout rate for layers")

    parser.add_argument("--lr", type=float, default=0.005,
                        help="initial learning rate")
    parser.add_argument("--wd", type=float, default=0,
                        help="weight decay (L2 loss on parameters)")
    parser.add_argument("--dr", type=float, default=0, help="decay rate of lr")

    parser.add_argument("--train_dist", type=str, default="euclidean",
                        help="distance function used in train (inner, cosine, euclidean, manhattan)")
    parser.add_argument("--test_dist", type=str, default="euclidean",
                        help="distance function used in test (inner, cosine, euclidean, manhattan)")

    parser.add_argument("--csls", type=int, default=0,
                        help="whether to use csls in test (0 means not using)")
    parser.add_argument("--rerank", action="store_true",
                        default=False, help="whether to use rerank in test")
    return parser.parse_args()


def update_config_from_json(args, config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


if __name__ == '__main__':

    args = parse_arguments()
    config_file = 'training_configuration/MTransE_zh_en.json'
    args = update_config_from_json(args, config_file)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    writer = SummaryWriter("_runs/%s_%s" %
                           (args.data_dir.split("/")[-1], args.log))
    logger.info(args)

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load Data
    d = AlignmentData(data_dir=args.data_dir, rate=args.rate, share=args.share, swap=args.swap, val=args.val,
                      with_r=args.encoder.lower() == "naea")
    logger.info(d)

    experiment = Experiment(args=args)

    t_total = time.time()
    experiment.train_and_eval()
    logger.info("optimization finished!")
    logger.info("total time elapsed: {:.4f} s".format(time.time() - t_total))
