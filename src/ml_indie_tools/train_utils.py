import logging
import math
from datetime import datetime
from datetime import timezone
import time
import numpy as np
import json
import queue
import threading
import asyncio


class TrainUtils:
    def __init__(self, indra_server_profile_name=None, username=None, password=None):
        self.log = logging.getLogger(__name__)
        self.indra_server_profile_name = indra_server_profile_name
        self.username = username
        self.password = password
        self.indra_active = False
        self.icl = None
        self.train_session_active = False
        self.session_id = None

    async def init_indra(self):
        self.indra_active = False
        self.icl = None
        self.session_id = None
        if self.indra_server_profile_name is not None:
            try:
                from indralib.indra_client import IndraClient
            except Exception as e:
                self.log.error(
                    f"indralib is required to use the indra_server_profile: {e}"
                )
        else:
            self.log.debug("No indra_server_profile_name provided")
            return False
        if self.indra_server_profile_name is not None:
            self.icl = IndraClient(
                verbose=False, profile_name=self.indra_server_profile_name
            )
        if self.icl is None:
            logging.error(
                f"Could not create Indrajala client with profile: {self.indra_server_profile_name}"
            )
            self.indra_server_profile_name = None
            return False
        ws = await self.icl.init_connection(verbose=True)
        if ws is None:
            logging.error(
                f"Could not connect to Indrajala with profile: {self.indra_server_profile_name}"
            )
            return False
        else:
            self.indra_active = True
            self.log.info(
                f"Connected to Indrajala server with profile: {self.indra_server_profile_name}"
            )
            if self.username is not None and self.password is not None:
                self.session_id = await self.icl.login_wait(
                    self.username, self.password
                )
                if self.session_id is None:
                    self.log.error(
                        f"Could not log in to Indrajala as user: {self.username}"
                    )
                    self.indra_active = False
                    return False
                else:
                    self.log.info(
                        f"Logged in to Indrajala as user: {self.username}, session id: {self.session_id}"
                    )
                    return True
            else:
                self.log.error(
                    f"No username and/or password provided, cannot log in to Indrajala profile: {self.indra_server_profile_name}"
                )
                return False

    @staticmethod
    def progress_bar_string(progress, max_progress, bar_length=20):
        """Create a Unicode progress bar string

        This creates a string of length bar_length with a Unicode progress bar using
        fractional Unicode block characters. The returned string is always of constant
        length and is suitable for printing to a terminal or notebook.

        This pretty much obsoletes the `tqdm` or similar package for simple progress bars.

        :param progress: current progress
        :param max_progress: maximum progress
        :param bar_length: length of the progress bar
        :return: Unicode progress bar string of length `bar_length`
        """
        progress_frac = progress / max_progress
        num_blocks = int(bar_length * progress_frac)
        rem = bar_length * progress_frac - num_blocks
        blocks = " ▏▎▍▌▋▊▉█"
        remainder_index = int(rem * len(blocks))
        bar = blocks[-1] * num_blocks
        if remainder_index > 0:
            bar += blocks[remainder_index]
        bar += " " * (bar_length - len(bar))
        return bar

    def train_session_start(
        self,
        model_name,
        model_description,
        model_version,
        model_params,
        indra_subdomain=None,
        mean_window=20,
        status_string_size=80,
    ):
        if self.train_session_active is True:
            self.log.warning(
                "Training session already active, closing existing session"
            )
            self.train_session_end()

        self.model_name = model_name
        self.model_description = model_description
        self.model_version = model_version
        self.model_params = model_params
        if indra_subdomain is None:
            subdomain = f"{model_name}/{model_version}".replace(" ", "_")
            self.log.warning(f"No Indrajala subdomain set, using {subdomain}")
        self.indra_subdomain = indra_subdomain
        self.model_loss_history = []
        self.losses = np.array([])
        self.norms = np.array([])
        self.mean_window = mean_window
        self.status_string_size = status_string_size
        self.last_tick = None
        self.last_iter = None
        self.train_session_active = True

        if self.indra_server_profile_name is not None:
            self.indra_queue = queue.Queue()
            self.indra_thread_running = True
            self.sync_indra = threading.Thread(
                target=self.sync_logger_worker,
                name="_sync_logger_worker",
                args=[],
                daemon=True,
            )
            self.sync_indra.start()

    def sync_logger_worker(self):
        self.log.debug("Starting indra thread")
        asyncio.run(self.async_logger_worker())
        self.log.debug("Stopped indra thread, async_logger_worker() returned")

    async def async_logger_worker(self):
        if self.indra_active is False:
            indra_inited = await self.init_indra()
            if indra_inited is False:
                self.log.error(
                    "Could not initialize Indrajala, stopping indra async thread"
                )
                self.indra_thread_running = False
                return
        while self.indra_thread_running is True:
            try:
                rec = self.indra_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            await self.register_train_state(rec)
            self.indra_queue.task_done()
        await self.icl.close_connection()
        self.log.debug("Stopped indra thread (async_logger_worker)")

    def train_session_end(self):
        self.indra_thread_running = False
        if not self.train_session_active:
            self.log.error(
                "No active training session: use train_session_start() first"
            )
            return
        self.train_session_active = False

    async def indra_report(self, record):
        if not self.indra_active:
            self.log.error("Indrajala not active")
            return
        from indralib.indra_event import IndraEvent

        event = IndraEvent()
        event.domain = f"$event/ml/model/train/{self.indra_subdomain}/record"
        event.from_id = "python/ml_indie_tools"
        event.to_scope = "public"
        event.data_type = "json/ml/trainrecord"
        event.data = json.dumps(record)
        event.auth_hash = self.session_id
        await self.icl.send_event(event)

        event = IndraEvent()
        event.domain = f"$event/ml/model/train/{self.indra_subdomain}/loss"
        event.from_id = "python/ml_indie_tools"
        event.to_scope = "public"
        event.data_type = "number/float"
        event.data = json.dumps(record["mean_loss"])
        event.auth_hash = self.session_id
        await self.icl.send_event(event)

    def train_state(self, record):

        mandatory_fields = ["epoch", "batch", "num_batches", "loss"]
        # optional_fields = ["val_loss", "learning_rate", "gradient_norm"]

        if self.train_session_active is False:
            self.log.error(
                "No active training session at train_state(): use train_session_start() first"
            )
            return "n/a", None
        for field in mandatory_fields:
            if field not in record:
                self.log.error(f"Missing mandatory field '{field}' in record")
                return "n/a", None

        # Calculate perplexity, accuracy from loss:
        record["perplexity"] = math.exp(record["loss"])
        record["accuracy"] = 1 - record["loss"]
        if "val_loss" in record:
            record["val_perplexity"] = math.exp(record["val_loss"])
            record["val_accuracy"] = 1 - record["val_loss"]

        self.losses = np.append(self.losses, record["loss"])
        record["mean_loss"] = np.mean(self.losses[-self.mean_window :])
        if "gradient_norm" in record:
            self.norms = np.append(self.norms, record["gradient_norm"])
            record["mean_gradient_norm"] = np.mean(self.norms[-self.mean_window :])
        t = time.time()
        if self.last_tick is not None and self.last_iter is not None:
            dt = t - self.last_tick
            di = record["batch"] - self.last_iter
            if di > 0:
                dt = dt / di
                record["Sec/It"] = dt
        self.last_tick = t
        self.last_iter = record["batch"]

        record["timestamp"] = (datetime.now(timezone.utc).isoformat(),)

        self.model_loss_history.append(record)

        if self.indra_active:
            self.indra_queue.put(record)

        pbar = self.progress_bar_string(record["batch"], record["num_batches"])

        batch_state = (
            f"Ep: {record['epoch']:.02f} Bat: {record['batch']}/{record['num_batches']}"
        )
        len_bs = 28
        if len(batch_state) < len_bs:
            batch_state = batch_state + " " * (len_bs - len(batch_state))
        status_string = f"{batch_state} ⦊{pbar}⦉ loss: {record['mean_loss']:.4f}"
        if "learning_rate" in record:
            status_string += f" lr: {record['learning_rate']:.6f}"
        if "gradient_norm" in record:
            status_string += f" grad_norm: {record['gradient_norm']:.3f}"
        if "Sec/It" in record:
            status_string += f" Sec/It: {record['Sec/It']:.3f}"
        if len(status_string) > self.status_string_size:
            status_string = status_string[: self.status_string_size]
        if len(status_string) < self.status_string_size:
            status_string = status_string + " " * (
                self.status_string_size - len(status_string)
            )

        return status_string, record

    async def register_train_state(self, record):
        if self.indra_active:
            await self.indra_report(record)
