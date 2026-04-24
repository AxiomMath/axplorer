import os
import time
from logging import getLogger

import torch

from src.models.model import evaluate

logger = getLogger()


def reload_model_optimizer(args, model, optimizer):
    model_path = os.path.join(args.dump_path, "model.pt")
    optimizer_path = os.path.join(args.dump_path, "optimizer.pt")
    if os.path.isfile(model_path):
        logger.info("resuming from existing model")
        if args.device == "cuda":
            reloaded = torch.load(model_path)
        else:
            reloaded = torch.load(model_path, map_location=torch.device(args.device))
        model.load_state_dict(reloaded)
    if os.path.isfile(optimizer_path):
        logger.info("resuming from existing optimizer")
        if args.device == "cuda":
            reloaded = torch.load(optimizer_path)
        else:
            reloaded = torch.load(optimizer_path, map_location=torch.device(args.device))
        optimizer.load_state_dict(reloaded)


def train(model, args, loader, optim, test_dataset, current_best_loss=None):
    best_loss = current_best_loss or float("inf")
    curr_loss = 0
    for step in range(args.max_steps):

        if step % 100 == 0:
            t0 = time.time()
        batch = loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch[0], batch[1]

        _, loss, _ = model(X, Y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        curr_loss += loss.item()

        if args.device == "mps" and (step + 1) % 500 == 0:
            torch.mps.empty_cache()

        # logging
        if (step + 1) % 100 == 0:
            t1 = time.time()
            logger.info(f"step {step + 1} | loss {loss.item():.4f} | steps time {(t1-t0)*1000:.2f}ms")
        if (step + 1) % args.num_eval_steps == 0:
            train_loss = curr_loss / args.num_eval_steps
            test_loss = evaluate(model, test_dataset, args.device, batch_size=100, max_batches=10)
            logger.info(f"step {step + 1} train loss: {train_loss} test loss: {test_loss}")
            if args.save_best and test_loss < best_loss:
                model_path = os.path.join(args.dump_path, "model.pt")
                optimizer_path = os.path.join(args.dump_path, "optimizer.pt")
                torch.save(model.state_dict(), model_path)
                torch.save(optim.state_dict(), optimizer_path)
                logger.info(f"test loss {test_loss} is the best so far, saved model to {model_path}")
                best_loss = test_loss
            curr_loss = 0

    if not args.save_best:
        model_path = os.path.join(args.dump_path, "model.pt")
        optimizer_path = os.path.join(args.dump_path, "optimizer.pt")
        torch.save(model.state_dict(), model_path)
        torch.save(optim.state_dict(), optimizer_path)

    return best_loss
