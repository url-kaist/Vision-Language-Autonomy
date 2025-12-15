import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Sequence, Literal, Union

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


TimePolicy = Literal["manual", "required"]


@dataclass
class RRConfig:
    # Default timeline used by set_time/step in manual mode
    timeline: str = "run_timeline"

    # Default dt used by step() in manual mode
    dt: float = 0.1

    # If true, write an .rrd file to disk
    save_rrd: bool = True

    # Time policy:
    # - "manual": time can be driven by set_time/step; log calls may omit times
    # - "required": every log call must provide times
    time_policy: TimePolicy = "manual"

    # Optional: add extra timelines you commonly want to set together
    # Example: ["ros", "task"]
    extra_timelines: Optional[List[str]] = None


class RRCore:
    def __init__(self, output_path, run_id: Optional[str] = None, cfg: Optional[RRConfig] = None):
        self.cfg = cfg or RRConfig()
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base = f"run/{self.run_id}"

        rr.init(f"{self.cfg.timeline}_{self.run_id}")
        if self.cfg.save_rrd:
            rr.save(output_path / f"run_{self.run_id}.rrd")

        # Internal time context per timeline
        self._time_ctx: Dict[str, float] = {}

        # Initialize manual timeline time context
        self._t = 0.0
        self._time_ctx[self.cfg.timeline] = self._t
        rr.set_time_seconds(self.cfg.timeline, self._t)

        # Shared camera origin for 2D views
        self.primary_camera_entity = f"{self.base}/camera"

        self._send_blueprint()

    # -------------------------
    # Blueprint
    # -------------------------
    def _send_blueprint(self):
        blueprint = rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(
                    name="3D",
                    contents=[
                        f"{self.base}/world/**",
                        f"{self.base}/sg/**",
                        f"{self.base}/vg/annotations/**",
                        f"{self.base}/nav/**",
                    ],
                ),
                rrb.TextDocumentView(name="Status", contents=[f"{self.base}/panel/status"]),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="RGB (global)",
                    origin=self.primary_camera_entity,
                    contents=[
                        f"{self.base}/vg/kfs/**/annotated_global",
                        f"{self.base}/vg/annotations/**",
                    ],
                ),
                rrb.Spatial2DView(
                    name="RGB (inference)",
                    origin=self.primary_camera_entity,
                    contents=[
                        f"{self.base}/vg/kfs/**/annotated_inference",
                        f"{self.base}/vg/annotations/**",
                    ],
                ),
                rrb.Spatial2DView(
                    name="Depth",
                    origin=self.primary_camera_entity,
                    contents=[f"{self.base}/vg/kfs/**/depth"],
                ),
            ),
            rrb.Vertical(
                rrb.TextDocumentView(name="Instruction", contents=[f"{self.base}/panel/instruction"]),
                rrb.TextDocumentView(name="Prompt", contents=[f"{self.base}/panel/prompt"]),
                rrb.TextDocumentView(name="Response", contents=[f"{self.base}/panel/response"]),
                rrb.TextDocumentView(name="Aggregation", contents=[f"{self.base}/panel/aggregation"]),
            ),
        )
        rr.send_blueprint(blueprint)

    # -------------------------
    # Time handling
    # -------------------------
    def set_time(self, t_seconds: float, timeline: Optional[str] = None):
        # Sets time on a timeline (manual mode helper).
        tl = timeline or self.cfg.timeline
        sec = float(t_seconds)
        self._time_ctx[tl] = sec
        rr.set_time_seconds(str(tl), sec)

    def step(self, dt: Optional[float] = None, timeline: Optional[str] = None):
        # Advances time on a timeline (manual mode helper).
        tl = timeline or self.cfg.timeline
        inc = float(self.cfg.dt if dt is None else dt)
        curr = float(self._time_ctx.get(tl, 0.0))
        self.set_time(curr + inc, timeline=tl)

    def set_times(self, times: Dict[str, float]):
        # Sets time on multiple timelines at once.
        for tl, sec in times.items():
            self.set_time(float(sec), timeline=str(tl))

    def _require_or_default_times(self, times: Optional[Dict[str, float]]) -> Dict[str, float]:
        # Returns the times to apply for a log call.
        # - required policy: must be provided
        # - manual policy: if not provided, use current internal time context (at least default timeline)
        if self.cfg.time_policy == "required":
            if times is None or len(times) == 0:
                raise ValueError("RRCore(time_policy='required') requires `times` for every log call.")
            return {str(k): float(v) for k, v in times.items()}

        # manual policy
        if times is not None and len(times) > 0:
            return {str(k): float(v) for k, v in times.items()}

        # Use current manual context
        default_tl = self.cfg.timeline
        if default_tl not in self._time_ctx:
            self._time_ctx[default_tl] = 0.0
        return {default_tl: float(self._time_ctx[default_tl])}

    def _apply_times_for_log(self, times: Optional[Dict[str, float]]):
        # Applies time context before rr.log calls so that logs are timestamped properly.
        resolved = self._require_or_default_times(times)
        for tl, sec in resolved.items():
            rr.set_time_seconds(str(tl), float(sec))
            self._time_ctx[str(tl)] = float(sec)

    # -------------------------
    # Core logging wrappers (time-aware)
    # -------------------------
    def log(self, entity: str, components: Union[Any, Sequence[Any]], *, times: Optional[Dict[str, float]] = None, timeless: bool = False):
        # Generic, time-aware rr.log wrapper.
        # If timeless=True, time context is irrelevant, but we still keep consistent behavior.
        if not timeless:
            self._apply_times_for_log(times)
        rr.log(entity, components, timeless=timeless)

    def log_text(self, entity: str, text: str, *, times: Optional[Dict[str, float]] = None, timeless: bool = False):
        self.log(entity, rr.TextDocument(text), times=times, timeless=timeless)

    def log_panel(self, name: str, text: str, *, times: Optional[Dict[str, float]] = None):
        # Panels are normal time-stamped logs.
        self.log_text(f"{self.base}/panel/{name}", text, times=times)

    def clear(self, namespace: Optional[str] = None, *, times: Optional[Dict[str, float]] = None):
        # Clear is usually time-stamped so you can scrub history; keep it non-timeless by default.
        self.log(namespace or self.base, rr.Clear(recursive=True), times=times, timeless=False)

    # Convenience helpers for common data types
    def log_image(self, entity: str, image: np.ndarray, *, times: Optional[Dict[str, float]] = None, jpeg_quality: int = 95):
        self.log(entity, rr.Image(image).compress(jpeg_quality=jpeg_quality), times=times)

    def log_depth(self, entity: str, depth: np.ndarray, *, times: Optional[Dict[str, float]] = None, meter: float = 1.0):
        self.log(entity, rr.DepthImage(depth, meter=meter), times=times)

    def log_points3d(self, entity: str, points: np.ndarray, *, times: Optional[Dict[str, float]] = None,
                     colors: Optional[List[int]] = None, radii: Optional[float] = None):
        kwargs = {}
        if colors is not None:
            kwargs["colors"] = colors
        if radii is not None:
            kwargs["radii"] = radii
        self.log(entity, rr.Points3D(points, **kwargs), times=times)


class VGPlugin:
    def __init__(self, core: RRCore):
        self.core = core
        self.base = f"{core.base}/vg"
        self._qid = 0

    def next_qid(self) -> int:
        self._qid += 1
        return self._qid

    # -------------------------
    # High-level task/panel logs
    # -------------------------
    def log_task(self, action: str, target_name: str, candidate_names: list, reference_names: list, *,
                 times: Optional[Dict[str, float]] = None):
        payload = {
            "action": action,
            "target_name": target_name,
            "candidate_names": candidate_names,
            "reference_names": reference_names,
        }
        self.core.log_panel("instruction", json.dumps(payload, indent=2), times=times)

    def log_status(self, status: str, node_active: Optional[bool] = None, *, times: Optional[Dict[str, float]] = None):
        text = f"status: {status}"
        if node_active is not None:
            text += f"\nnode_active: {node_active}"
        self.core.log_panel("status", text, times=times)

    # -------------------------
    # Keyframe selection & images
    # -------------------------
    def log_selected_keyframes(self, kf_ids, etype=None, gid=None, eids=None, *, times: Optional[Dict[str, float]] = None):
        payload = {
            "etype": etype,
            "gid": gid,
            "eids": list(eids) if eids is not None else None,
            "kf_ids": list(kf_ids),
        }
        self.core.log_text(f"{self.base}/selection", json.dumps(payload, indent=2), times=times)

    def log_kf_images(self, kf_id: int, rgb=None, annotated_global=None, annotated_inference=None, depth=None, *,
                      times: Optional[Dict[str, float]] = None):
        if rgb is not None:
            self.core.log_image(f"{self.base}/kfs/{kf_id}/rgb", rgb, times=times)
        if annotated_global is not None:
            self.core.log_image(f"{self.base}/kfs/{kf_id}/annotated_global", annotated_global, times=times)
        if annotated_inference is not None:
            self.core.log_image(f"{self.base}/kfs/{kf_id}/annotated_inference", annotated_inference, times=times)
        if depth is not None:
            self.core.log_depth(f"{self.base}/kfs/{kf_id}/depth", depth, times=times, meter=1.0)

    # -------------------------
    # Query / response logs
    # -------------------------
    def log_query(self, qid: int, prompt: str, options: dict = None, *, times: Optional[Dict[str, float]] = None):
        self.core.log_text(f"{self.base}/query/{qid}/prompt", prompt, times=times)
        if options is not None:
            self.core.log_text(f"{self.base}/query/{qid}/options", json.dumps(options, indent=2), times=times)
        self.core.log_panel("prompt", prompt, times=times)

    def log_response(self, qid: int, response_text: str, parsed: dict = None, latency_s: float = None, *,
                     times: Optional[Dict[str, float]] = None):
        self.core.log_text(f"{self.base}/query/{qid}/response_raw", response_text[:5000], times=times)

        if parsed is not None:
            parsed_txt = json.dumps(parsed, indent=2)
            self.core.log_text(f"{self.base}/query/{qid}/response_parsed", parsed_txt, times=times)
            self.core.log_panel("response", parsed_txt, times=times)
        else:
            self.core.log_panel("response", response_text[:5000], times=times)

        if latency_s is not None:
            self.core.log_text(f"{self.base}/query/{qid}/latency", f"{float(latency_s):.3f}s", times=times)

    def log_aggregation(self, snapshot: dict, *, times: Optional[Dict[str, float]] = None):
        text = json.dumps(snapshot, indent=2)
        self.core.log_text(f"{self.base}/agg/snapshot", text, times=times)
        self.core.log_panel("aggregation", text, times=times)

    def log_final_answer(self, answer_text: str, extra: dict = None, *, times: Optional[Dict[str, float]] = None):
        self.core.log_text(f"{self.base}/answer/final", answer_text, times=times)
        if extra is not None:
            self.core.log_text(f"{self.base}/answer/meta", json.dumps(extra, indent=2), times=times)

    # -------------------------
    # Optional: explicit begin/end event markers for latency visualization
    # -------------------------
    def log_event(self, name: str, payload: Optional[dict] = None, *, times: Optional[Dict[str, float]] = None):
        obj = {"event": name}
        if payload is not None:
            obj["payload"] = payload
        self.core.log_text(f"{self.base}/events/{name}", json.dumps(obj, indent=2), times=times)
