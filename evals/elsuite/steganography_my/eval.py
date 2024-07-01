import numpy as np
from pyparsing import Any

import evals
import evals.elsuite.steganography_my.steganography_my as steganography
import evals.metrics
from evals.api import CompletionFn, DummyCompletionFn
from evals.eval import Eval
from evals.record import RecorderBase


class Steganography_my(Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        prompt_version: str = "scratch",
        reconstruction_error_metric: str = "semantic_distance",
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        if len(completion_fns) == 1 and isinstance(completion_fns[0], DummyCompletionFn):
            completion_fn = completion_fns[0]
            completion_fns = [completion_fn, completion_fn]

        assert (
            len(completion_fns) == 2
        ), f"Steganography requires 2 completion_fns (the 1st is the target model, the 2nd is the monitor model), received {len(completion_fns)}."

        self.completion_fns = completion_fns
        self.samples_jsonl = samples_jsonl
        self.prompt_version = prompt_version
        self.reconstruction_error_metric = reconstruction_error_metric

    def eval_sample(self, sample: Any, *_):
        assert isinstance(sample, dict), "sample must be a dict"
        assert (
            "payload_text" in sample or "test_type" in sample
        ), "sample must have a 'payload_text' or 'test_type' key"  # NOTE: modified to test for interleaved tokens example from James
        assert "task_prompt" in sample, "sample must have a task_prompt key"
        assert "task_output" in sample, "sample must have a task_output key"
        if "payload_text" in sample:
            assert isinstance(sample["payload_text"], str), "payload_text must be a string"
            payload = sample["payload_text"]
        assert isinstance(sample["task_prompt"], str), "task_prompt must be a string"
        assert isinstance(sample["task_output"], str), "task_output must be a string"

        task_prompt = sample["task_prompt"]
        reference_output = sample["task_output"]

        if (
            sample.get("test_type") == "interleaved_tokens"
        ):  # NOTE: added to test for interleaved tokens example from James
            self.prompt_version = "interleaved_tokens"  # NOTE: probably there is a better/earlier way to set this but atm I don't know how
            # self.reconstruction_error_metric = "semantic_distance" #TODO: check if this is needed and if needed then define a new metric
            results = steganography.run_interleaved_tokens_eval(
                task_prompt=task_prompt,
                reference_output=reference_output,
                completion_fn=self.completion_fns[0],
                monitor_completion_fn=self.completion_fns[1],
                prompt_version=self.prompt_version,
                metadata=sample,
            )
            evals.record.record_metrics(
                reconstruction_error=results[self.reconstruction_error_metric],
                character_error_rate_cap1=results["character_error_rate_cap1"],
                semantic_distance=results["semantic_distance"],
                rule_violated=results["rule_violated"],
                quality_scratchpad=results.get("quality_scratchpad", None),
                task_successful=results.get("task_successful"),
                quality_explanation=results.get("quality_explanation", ""),
                similarity_score=results.get("similarity_score", 123),
            )
        else:
            results = steganography.run_eval(
                task_prompt=task_prompt,
                reference_output=reference_output,
                payload=payload,
                completion_fn=self.completion_fns[0],
                monitor_completion_fn=self.completion_fns[1],
                prompt_version=self.prompt_version,
                metadata=sample,
            )
            evals.record.record_metrics(
                undetected_acc=results["undetected_acc"],
                detectability=results["detected"],
                reconstruction_error=results[self.reconstruction_error_metric],
                character_error_rate_cap1=results["character_error_rate_cap1"],
                semantic_distance=results["semantic_distance"],
                rule_violated=results["rule_violated"],
            )

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)
        metrics = recorder.get_metrics()
        return {
            **(
                {
                    "undetected_acc": np.mean(
                        [d["undetected_acc"] for d in metrics if "undetected_acc" in d]
                    )
                }
                if any("undetected_acc" in d for d in metrics)
                else {}
            ),
            **(
                {
                    "detectability": np.mean(
                        [d["detectability"] for d in metrics if "detectability" in d]
                    )
                }
                if any("detectability" in d for d in metrics)
                else {}
            ),
            "reconstruction_error": np.mean([d["reconstruction_error"] for d in metrics]),
            "character_error_rate_cap1": np.mean([d["character_error_rate_cap1"] for d in metrics]),
            "semantic_distance": np.mean([d["semantic_distance"] for d in metrics]),
            "rule_violated": np.mean([d["rule_violated"] for d in metrics]),
            **(
                {
                    "similarity_score": np.mean(
                        [d["similarity_score"] for d in metrics if "similarity_score" in d]
                    )
                }
                if any("similarity_score" in d for d in metrics)
                else {}
            ),
        }
