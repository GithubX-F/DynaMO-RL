# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import copy
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
import tensordict
from tensordict import TensorDict
from typing import Type, Dict, List, Union, Optional
from .rl_dataset import RLHFDataset




def repeat(
    batch,
    non_tensor_batch,
    meta_info,
    repeat_times: Union[int, List[int]] = 2,
    interleave=True,
):
    """
    Repeat the batch data a specified number of times.

    Args:
        repeat_times (int): Number of times to repeat the data.
        interleave (bool): Whether to interleave the repeated data.

    Returns:
        DataProto: A new DataProto with repeated data.
    """
    if isinstance(repeat_times, int):
        if batch is not None:
            if interleave:
                # Interleave the data
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0)
                    for key, tensor in batch.items()
                }
            else:
                # Stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0)
                    .expand(repeat_times, *tensor.shape)
                    .reshape(-1, *tensor.shape[1:])
                    for key, tensor in batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(
                    val, (repeat_times,) + (1,) * (val.ndim - 1)
                )

        return DataProto(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=meta_info,
        )
    else:
        assert len(repeat_times) == batch.batch_size[0]
        repeated_tensors = {}
        for key, tensor in batch.items():
            tensor_list = []
            for n, item in zip(repeat_times, tensor):
                expanded_item = item.unsqueeze(0).expand(n, *item.shape)
                tensor_list.append(expanded_item)
            repeated_tensors[key] = torch.cat(tensor_list, dim=0)

        repeated_batch = TensorDict(
            source=repeated_tensors,
            batch_size=(sum(repeat_times),),
        )

        repeated_non_tensor_batch = {}
        for key, val in non_tensor_batch.items():
            total_size = sum(repeat_times)
            repeated_val = np.empty(total_size, dtype=object)
            current_idx = 0
            for n, item in zip(repeat_times, val):
                for i in range(n):
                    repeated_val[current_idx + i] = copy.deepcopy(item)
                current_idx += n
            repeated_non_tensor_batch[key] = repeated_val

        return DataProto(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=meta_info,
        )
    

def get_rollout_n_per_prompt(
    sample_counts: np.ndarray, 
    priority: np.ndarray,
    total_rollout_n: int,
    rollout_n_min: int = 4,
    rollout_n_max: int = 24,
    initial_budget: int = 8, 
) -> List[int]:
    """
    Dynamically allocate rollout_n based on priority while strictly adhering to min/max constraints.

    Improved strategy:
    1. Prioritize allocating exploration budget for prompts with sample_count < initial_budget
    2. Allocate remaining budget using the original "waterfall filling" method
    """
    n_prompts = len(priority)
    if n_prompts == 0:
        return []

    rollout_n_min = max(0, rollout_n_min)
    rollout_n_max = max(rollout_n_min, rollout_n_max)
    initial_budget = max(rollout_n_min, initial_budget)
    min_possible_total = n_prompts * rollout_n_min
    max_possible_total = n_prompts * rollout_n_max
    total_rollout_n = np.clip(total_rollout_n, min_possible_total, max_possible_total)

    exploration_needed = np.maximum(0, initial_budget - sample_counts)
    exploration_needed = np.minimum(exploration_needed, rollout_n_max)
    exploration_mask = exploration_needed > 0
    
    total_exploration_needed = exploration_needed.sum()
    
    rollout_counts = np.zeros(n_prompts, dtype=np.int64)
    if total_exploration_needed > 0:
        if total_exploration_needed <= total_rollout_n:
            rollout_counts[exploration_mask] = exploration_needed[exploration_mask]
            budget_remaining = total_rollout_n - total_exploration_needed
        else:
            scale_factor = total_rollout_n / total_exploration_needed
            scaled_exploration = (exploration_needed * scale_factor).astype(np.int64)
            scaled_exploration = np.maximum(scaled_exploration, rollout_n_min)
            while scaled_exploration.sum() > total_rollout_n:
                candidates = np.where(scaled_exploration > rollout_n_min)[0]
                if len(candidates) == 0:
                    break
                idx_to_reduce = candidates[np.argmax(scaled_exploration[candidates])]
                scaled_exploration[idx_to_reduce] -= 1
            
            rollout_counts = scaled_exploration
            budget_remaining = total_rollout_n - rollout_counts.sum()
    else:
        budget_remaining = total_rollout_n
    if budget_remaining > 0:
        min_needed = np.maximum(0, rollout_n_min - rollout_counts)
        total_min_needed = min_needed.sum()
        
        if total_min_needed > 0:
            if total_min_needed <= budget_remaining:
                rollout_counts += min_needed
                budget_remaining -= total_min_needed
            else:
                scale = budget_remaining / total_min_needed
                scaled_min = (min_needed * scale).astype(np.int64)
                rollout_counts += scaled_min
                budget_remaining -= scaled_min.sum()
    
    if budget_remaining == 0:
        return rollout_counts.tolist()

    current_priorities = priority.copy().astype(np.float64)
    current_priorities[current_priorities < 1e-8] = 0.0

    while budget_remaining > 0:
        eligible_mask = (rollout_counts < rollout_n_max) & (current_priorities > 1e-8)
        eligible_indices = np.where(eligible_mask)[0]

        if len(eligible_indices) == 0:
            fallback_indices = np.where(rollout_counts < rollout_n_max)[0]
            if len(fallback_indices) == 0:
                break 

            for i in range(budget_remaining):
                idx_to_add = fallback_indices[i % len(fallback_indices)]
                rollout_counts[idx_to_add] += 1
            
            budget_remaining = 0
            break

        eligible_priorities = current_priorities[eligible_indices]
        normalized_priorities = eligible_priorities / eligible_priorities.sum()
        ideal_add_float = budget_remaining * normalized_priorities

        ideal_add_int = np.floor(ideal_add_float).astype(np.int64)
        remainder = int(np.round(budget_remaining - ideal_add_int.sum()))

        if remainder > 0:
            fractional_parts = ideal_add_float - ideal_add_int
            remainder_indices = np.argsort(-fractional_parts)
            
            for i in range(remainder):
                idx_in_eligible = remainder_indices[i]
                ideal_add_int[idx_in_eligible] += 1

        capacity_left = rollout_n_max - rollout_counts[eligible_indices]
        actual_add = np.minimum(ideal_add_int, capacity_left)

        rollout_counts[eligible_indices] += actual_add
        budget_added_this_round = actual_add.sum()
        budget_remaining -= budget_added_this_round
    final_diff = total_rollout_n - rollout_counts.sum()
    if final_diff > 0:
        fallback_indices = np.where(rollout_counts < rollout_n_max)[0]
        if len(fallback_indices) > 0:
            for i in range(final_diff):
                idx_to_add = fallback_indices[i % len(fallback_indices)]
                rollout_counts[idx_to_add] += 1
    elif final_diff < 0:
        fallback_indices = np.where(rollout_counts > rollout_n_min)[0]
        if len(fallback_indices) > 0:
            for i in range(abs(final_diff)):
                idx_to_add = fallback_indices[i % len(fallback_indices)]
                rollout_counts[idx_to_add] -= 1

    assert rollout_counts.sum() == total_rollout_n
    return rollout_counts.tolist()


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager,
        ray_worker_group_cls=None,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            **kwargs,
        )

        self.rollout_n_min = self.config.actor_rollout_ref.rollout.get("n_low", 4)   
        self.rollout_n_max = self.config.actor_rollout_ref.rollout.get("n_high", 24) 
        self.rollout_n = self.config.actor_rollout_ref.rollout.n

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                                
                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
                        if self.use_reference_policy:
                            self.ref_policy_wg.start_profile()
                        if self.use_critic:
                            self.critic_wg.start_profile()
                        if self.use_rm:
                            self.rm_wg.start_profile()

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )


                if self.config.actor_rollout_ref.rollout.get("rollout_allocation", False):
                    rollout_n_per_prompt = get_rollout_n_per_prompt( 
                        batch_dict["sample_count"],
                        batch_dict["rank"],
                        self.rollout_n * len(batch_dict["rank"]),
                        self.rollout_n_min,
                        self.rollout_n_max,
                        initial_budget = 4,
                    )
                    gen_batch = repeat(
                        gen_batch.batch,
                        gen_batch.non_tensor_batch,
                        gen_batch.meta_info,
                        repeat_times=rollout_n_per_prompt,
                        interleave=True,
                    )
                else:
                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing


                index = [i for i in batch.non_tensor_batch["index"]]
                reward = reward_extra_infos_dict["acc"]
                self.train_dataset.update_prompt_stats(index, reward)

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)

                self.global_steps += 1


            self.train_dataset.update_rank()