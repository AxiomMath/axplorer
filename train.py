import argparse
import os
import time
from logging import getLogger

import numpy as np
import torch

from src.datasets import CharDataset, InfiniteDataLoader, load_initial_data, update_datasets
from src.envs import ENVS, build_env
from src.envs.environment import do_stats
from src.evaluator import sample_and_score
from src.models.model import Transformer
from src.trainer import reload_model_optimizer, train
from src.utils import bool_flag, force_release_memory, initialize_exp, log_resources, write_important_metrics

logger = getLogger()


def get_parser():
    parser = argparse.ArgumentParser("A simple Axplorer loop for different maths problems")

    parser.add_argument("--gensize", type=int, default=100000, help="Number of generate initial values")
    parser.add_argument("--max_epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=50000, help="number of training steps.")
    parser.add_argument("--num_samples_from_model", type=int, default=500000, help="sample the specified number from the model in each loop")
    parser.add_argument("--pop_size", type=int, default=200000, help="Total maximum number of examples at each epoch")
    parser.add_argument("--ntest", type=int, default=1000, help="Size of test set")
    parser.add_argument("--env_name", type=str, default="square", help="Math problem to be addressed")
    ENVS[parser.parse_known_args()[0].env_name].register_args(parser)

    parser.add_argument("--process_pool", type=bool_flag, default="true", help="use process_pool to generate and score initial data")
    parser.add_argument("--always_search", type=bool_flag, default="true", help="if True, use local search for all examples generated")
    parser.add_argument("--redeem_only", type=bool_flag, default="false", help="if True, save invalid examples only")
    parser.add_argument("--new_proportion", type=float, default=0.0, help="proportion of new samples in test set")

    parser.add_argument("--num_workers", type=int, default=8, help="number of data workers for both train/test")
    parser.add_argument("--num_eval_steps", type=int, default=500, help="number of step between each evaluation during training.")
    parser.add_argument("--seed", type=int, default=-1, help="seed")
    # sampling
    parser.add_argument("--top_k", type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument("--n_layer", type=int, default=4, help="number of layers")
    parser.add_argument("--n_head", type=int, default=8, help="number of heads (in a transformer)")
    parser.add_argument("--n_embd", type=int, default=256, help="number of feature channels in the model")
    parser.add_argument("--no_positional", type=bool_flag, default="false", help="no positional embedding")
    parser.add_argument("--max_len", type=int, default=500, help="Block size, maximum length of sequences")

    # optimization
    parser.add_argument("--batch_size", type=int, default=32, help="batch size during optimization")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    # evaluation against known "good sequences"
    parser.add_argument("--gen_batch_size", type=int, default=1000, help="generation batch size")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--temp_span", type=int, default=0, help="temperature span")
    parser.add_argument("--inc_temp", type=float, default=0.0, help="temperature")
    parser.add_argument("--keep_only_unique", type=bool_flag, default="true", help="keep only unique data")
    parser.add_argument("--save_best", type=bool_flag, default="false", help="save best model based on test loss")

    # path and ports
    parser.add_argument("--dump_path", type=str, default="checkpoint", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--cpu", type=bool_flag, default="false", help="run on cpu only")
    parser.add_argument("--data_generation_only", type=bool_flag, default="false", help="only generate data and exit")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.exp_id == "" and os.environ.get("MODAL_EXP_ID") is None:
        os.environ["MODAL_EXP_ID"] = time.strftime("%Y_%m_%d_%H_%M_%S")
        args.exp_id = os.environ["MODAL_EXP_ID"]

    args.device = "cpu" if args.cpu else ("mps" if torch.backends.mps.is_available() else "cuda")
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "mps":
        torch.mps.manual_seed(args.seed)

    fused = True if args.device in ["cuda", "mps"] else False

    logger = initialize_exp(args)
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)

    if args.seed < 0:
        args.seed = np.random.randint(1_000_000_000)
    logger.info(f"seed: {args.seed}")

    env = build_env(args)

    classname = env.data_class

    # system inits
    torch.manual_seed(args.seed)

    args.vocab_size = len(env.tokenizer.itos)

    args.block_size = args.max_len + 2
    stoi = env.tokenizer.stoi
    itos = env.tokenizer.itos

    # Initialize transformer
    model = Transformer(args, stoi["PAD"], stoi["EOS"])
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8, fused=fused)
    reload_model_optimizer(args, model, optimizer)

    train_set, test_set = load_initial_data(args, classname)
    if args.data_generation_only:
        logger.info("Data generation only mode. Exiting...")
        exit(0)
    train_data_path = os.path.join(args.dump_path, "train_data.pkl")
    test_data_path = os.path.join(args.dump_path, "test_data.pkl")

    # log initial stats
    metrics = do_stats(-1, data=train_set)
    temperature = args.temperature
    # Loop of Axplorer
    best_loss = None
    epoch_file = os.path.join(args.dump_path, "epoch.txt")
    if os.path.isfile(epoch_file):
        with open(epoch_file, "r") as f:
            n_epoch = int(f.read())
    else:
        n_epoch = 0
    temp_file = os.path.join(args.dump_path, "temperature.txt")
    if os.path.isfile(temp_file):
        with open(temp_file, "r") as f:
            temperature = float(f.read())
    else:
        temperature = args.temperature

    metric_file = os.path.join(args.dump_path, "metrics.txt")
    write_important_metrics(metrics, n_epoch, metric_file, command=args.command)

    for epoch in range(n_epoch, args.max_epochs):
        logger.info(f"==== Starting Epoch {n_epoch} =====")
        log_resources(f"Epoch {epoch} START")

        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()

        # tokenize
        train_words = [env.tokenizer.encode(d) for d in train_set]
        test_words = [env.tokenizer.encode(d) for d in test_set]
        # data loaders
        train_dataset = CharDataset(train_words, args.max_len, stoi)
        test_dataset = CharDataset(test_words, args.max_len, stoi)
        force_release_memory()

        if args.device == "cuda":
            logger.info(
                f"Memory allocated: {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB"
            )
        elif args.device == "mps":
            logger.info(
                f"Memory allocated: {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB, reserved: {torch.mps.driver_allocated_memory()/(1024*1024):.2f}MB"
            )

        batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=args.device == "cuda", num_workers=0)
        try:
            best_loss = train(model, args, batch_loader, optimizer, test_dataset, current_best_loss=best_loss)
        finally:
            batch_loader.close()
            del batch_loader
        log_resources(f"Epoch {epoch} AFTER_TRAIN")
        force_release_memory()

        logger.info(f"Sample with temperature {temperature} to {temperature+0.1*args.temp_span}")
        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()

        new_data = sample_and_score(model, args, stoi, itos, env, temperature, args.temp_span)
        log_resources(f"Epoch {epoch} AFTER_SAMPLE")

        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()

        # Possible to add another generation method here and mix it before taking the best
        train_set, test_set, inc_temp = update_datasets(args, new_data, train_set, test_set, train_data_path, test_data_path)
        log_resources(f"Epoch {epoch} AFTER_UPDATE_DATASETS")
        force_release_memory()

        # Possible to add another generation method here and mix it before taking the best
        if inc_temp and args.inc_temp > 0.0:
            temperature += args.inc_temp

        metrics = do_stats(-1, data=train_set)

        n_epoch += 1
        with open(epoch_file, "w") as f:
            f.write(str(n_epoch))
        with open(temp_file, "w") as f:
            f.write(str(temperature))

        write_important_metrics(metrics, n_epoch, metric_file)
