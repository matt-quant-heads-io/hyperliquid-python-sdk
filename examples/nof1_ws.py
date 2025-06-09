import os
from collections import deque
import yaml
import threading
import time
import json
from collections import deque
from queue import Queue
import torch
import numpy as np

import example_utils

from hyperliquid.utils import constants
import numpy as np
from numba import njit
from model import model


def _load_model_and_normalization_vals(path_to_conf="/home/ubuntu/hyperliquid-python-sdk/configs/test_model.yaml"):
    with open(path_to_conf, "r") as f:
        config = yaml.safe_load(f)
        
    checkpoint = torch.load(config["path_to_saved_model"], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    normalization_dict = checkpoint['normalization_values']

    return model, normalization_dict


model, normalization_dict = _load_model_and_normalization_vals() 

# --- Shared Buffers ---
MODEL_SEQ_LENGTH = 3
state_buffer = deque(maxlen=MODEL_SEQ_LENGTH) # ring buffer
LATEST_OB = None

#TODO: Read this in from model state dict
BID_PX_NORM_MEAN, BID_PX_NORM_STD = normalization_dict['bid_px']['mean'], normalization_dict['bid_px']['std'] 
ASK_PX_NORM_MEAN, ASK_PX_NORM_STD = normalization_dict['ask_px']['mean'], normalization_dict['ask_px']['std']
BID_SZ_NORM_MEAN, BID_SZ_NORM_STD = normalization_dict['bid_sz']['mean'], normalization_dict['bid_sz']['std']
ASK_SZ_NORM_MEAN, ASK_SZ_NORM_STD = normalization_dict['ask_sz']['mean'], normalization_dict['ask_sz']['std']
STATE_SHAPE = 83
MODEL_INPUT_SHAPE = (MODEL_SEQ_LENGTH, STATE_SHAPE)
position_and_passives = [0.0, 0.0, 0.0]
closed_trades = []   # fully executed trades (entry + exit)
trade_tracker_lock = threading.Lock()


@njit
def compute_features(prices):
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    return returns


def log_trade(prediction, action, metadata):
    instrument = metadata.get("symbol", "UNKNOWN")
    price = float(metadata.get("price", 0.0))
    timestamp = time.time()

    trade_event = {
        "timestamp": timestamp,
        "price": price,
        "prediction": prediction.tolist(),
        "action": action,
        "instrument": instrument
    }

    with trade_tracker_lock:
        if action in [1, 2]:  # Entry trade
            open_positions[instrument] = trade_event
        elif action == 3:  # Exit trade
            entry = open_positions.pop(instrument, None)
            if entry:
                # Pair entry and exit
                full_trade = {
                    "instrument": instrument,
                    "entry": entry,
                    "exit": trade_event,
                    "pnl": (price - entry["price"]) if entry["action"] == 1 else (entry["price"] - price)
                }
                closed_trades.append(full_trade)

    with open("./trades.log", "a") as f:
        f.write(json.dumps(trade_event) + "\n")


def normalize_next_orderbook(msg, depth=20):
    curr_bids, curr_asks = [[0.0,0.0]]*depth, [[0.0,0.0]]*depth

    bids_slice_len, asks_slice_len = min(depth, len(msg['data']['levels'][0])), min(depth, len(msg['data']['levels'][1]))
    
    curr_bids[:bids_slice_len] = [[float(d['sz']), float(d['px'])] for d in msg['data']['levels'][0]][:bids_slice_len]
    curr_asks[:asks_slice_len] = [[float(d['px']), float(d['sz'])] for d in msg['data']['levels'][1]][:asks_slice_len]

    norm_curr_ob = np.concatenate([list(np.array([[(b[0] - BID_SZ_NORM_MEAN)/BID_SZ_NORM_STD, (b[1] - BID_PX_NORM_MEAN)/BID_PX_NORM_STD]+[(a[0] - ASK_PX_NORM_MEAN)/ASK_PX_NORM_STD, (a[1] - ASK_SZ_NORM_MEAN)/ASK_SZ_NORM_STD] for b,a in zip(curr_bids, curr_asks)]).ravel()), position_and_passives])
    curr_ob = np.concatenate([list(np.array([a+b for b,a in zip(curr_bids, curr_asks)]).ravel()), position_and_passives])

    return norm_curr_ob, curr_ob


# --- Ingestion Thread ---
def websocket_listener(info):
    def handle_msg(msg):
        try:
            if msg['channel'] == 'l2Book':
                normalized_ob, LATEST_OB = normalize_next_orderbook(msg)
                state_buffer.appendleft(
                    normalized_ob
                )
        except Exception as e:
            print(f"Error parsing: {e}")

    # Subscriptions
    info.subscribe({"type": "l2Book", "coin": "BERA"}, handle_msg)

    while True:
        time.sleep(1)  # keep thread alive


def flush_closed_trades():
    with trade_tracker_lock:
        if closed_trades:
            with open("closed_trades.jsonl", "a") as f:
                for t in closed_trades:
                    f.write(json.dumps(t) + "\n")
            closed_trades.clear()


def _determine_if_action_is_valid(action):
    if position_and_passives[0] == 0:
        return action




def run_inference():
    while True:
        try:
            if len(state_buffer) >= MODEL_SEQ_LENGTH:
                tensor_input = torch.tensor(list(state_buffer), dtype=torch.float32)
                
                with torch.no_grad():
                    prediction, _, _ = model(tensor_input.unsqueeze(0))  # Assume output is logits or action probabilities

                # Example action logic: take argmax
                action = torch.argmax(prediction, dim=1).item()

                # TODO: Process whether action is valid and then process current positions etc.
                action = _determine_if_action_is_valid(action)

                # TODO: Send orders here depending on action
                

                # TODO: Update position_and_passives with passives and position + PnL etc.

                if action != 0: # i.e. do nothing 
                    # TODO: process metadata access latest OB via LATEST_OB
                    metadata = {
                        # "price": list(state_buffer)[0][0][1] if action in [2] else list(state_buffer)[0][0][1],
                        # "source": latest_msg.get("type"),
                        # "symbol": latest_msg.get("coin", "ETH")
                        # "action"
                    }
                    # log_trade(prediction, action, metadata)

                print(f"Inference: {prediction} -> Action: {action}")

        except Exception as e:
            print(f"Error in inference: {e}")

        time.sleep(0.05)


# --- Bootstrapping ---
def start_pipeline(info):
    threading.Thread(target=websocket_listener, args=(info,), daemon=True).start()
    threading.Thread(target=run_inference, daemon=True).start()
    # TODO: add a listener here for order updates / user events


def main():
    address, info, _ = example_utils.setup(constants.TESTNET_API_URL)
    start_pipeline(info)


if __name__ == "__main__":
    main()
