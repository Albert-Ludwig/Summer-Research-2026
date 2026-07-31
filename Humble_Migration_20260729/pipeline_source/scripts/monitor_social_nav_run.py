#!/usr/bin/env python3
"""Record synchronized SocialNav runtime topics until Ctrl+C, then summarize them."""

import json
import math
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message


OUTPUT_ROOT = Path("/tmp/social_nav_monitor")
NONZERO_EPS = 1.0e-4
TRAJECTORY_PATTERN = re.compile(r"predicted|projected|trajectory|traj|path|marker", re.I)

REQUIRED_TOPICS = {
    "/goal_pose": "geometry_msgs/msg/PoseStamped",
    "/social_nav_diffusion/policy_debug": "std_msgs/msg/String",
    "/cpr_j100_0001/cmd_vel": "geometry_msgs/msg/TwistStamped",
    "/cpr_j100_0001/platform/cmd_vel": "geometry_msgs/msg/TwistStamped",
    "/cpr_j100_0001/platform/odom/filtered": "nav_msgs/msg/Odometry",
    "/social_nav_diffusion/goal_path": "nav_msgs/msg/Path",
}


def wall_iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).astimezone().isoformat(timespec="microseconds")


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def is_nonzero(*values):
    return any(value is not None and abs(value) > NONZERO_EPS for value in values)


def message_stamp(message):
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    return {"sec": sec, "nanosec": nanosec, "seconds": sec + nanosec * 1.0e-9}


def topic_filename(topic):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "__", topic.strip("/"))
    return (cleaned or "root") + ".jsonl"


def parse_policy_debug(text):
    """Parse key=value fields without modifying or truncating the original string."""
    result = {}
    matches = list(re.finditer(r"(?:^|, )([A-Za-z0-9_]+)=", text))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[start:end]
        if value.endswith(", "):
            value = value[:-2]
        result[match.group(1)] = value
    return result


def twist_values(message):
    twist = getattr(message, "twist", message)
    nested = getattr(twist, "twist", None)
    if nested is not None:
        twist = nested
    return float(twist.linear.x), float(twist.angular.z)


def first_after(samples, start_time, predicate):
    for sample in samples:
        if sample["wall"] >= start_time and predicate(sample):
            return sample
    return None


def signed_magnitude_result(a, b):
    if a is None or b is None:
        return "unknown"
    if abs(a) <= NONZERO_EPS and abs(b) <= NONZERO_EPS:
        return "agree (both zero)"
    signs_agree = a * b > 0.0
    denominator = max(abs(a), abs(b), NONZERO_EPS)
    close = abs(a - b) <= max(0.05, 0.5 * denominator)
    if signs_agree and close:
        return "approximately agree"
    if signs_agree:
        return "sign agrees; magnitude differs"
    return "sign differs"


def detect_holds(samples, minimum_duration=0.5):
    holds = []
    start = None
    previous = None
    for sample in samples:
        if not is_nonzero(sample["v"], sample["w"]):
            if start is not None and previous["wall"] - start["wall"] >= minimum_duration:
                holds.append((start, previous))
            start = None
            previous = sample
            continue
        same = (
            previous is not None
            and is_nonzero(previous["v"], previous["w"])
            and sample["wall"] - previous["wall"] <= 0.5
            and abs(sample["v"] - previous["v"]) <= 1.0e-4
            and abs(sample["w"] - previous["w"]) <= 1.0e-4
        )
        if not same:
            if start is not None and previous["wall"] - start["wall"] >= minimum_duration:
                holds.append((start, previous))
            start = sample
        previous = sample
    if start is not None and previous is not None and previous["wall"] - start["wall"] >= minimum_duration:
        holds.append((start, previous))
    return holds


class SocialNavMonitor(Node):
    def __init__(self):
        super().__init__("social_nav_run_monitor")
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = OUTPUT_ROOT / run_name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.started_wall = time.time()
        self.files = {}
        self.file_names = {}
        self.subscriptions_by_topic = {}
        self.topic_types = {}
        self.discovered_topic_types = {}
        self.warnings = []

        self.goals = []
        self.policy = []
        self.policy_cmd = []
        self.platform_cmd = []
        self.odom = []
        self.message_counts = {}

        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1000,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        for topic, type_name in REQUIRED_TOPICS.items():
            self.add_subscription(topic, type_name, required=True)

        self.discovery_timer = self.create_timer(2.0, self.discover_trajectory_topics)
        self.discover_trajectory_topics()
        self.write_manifest(final=False)

        print(f"Recording SocialNav data in: {self.run_dir}", flush=True)
        print("Click the RViz Nav2 Goal only after this message appears.", flush=True)
        print("Press Ctrl+C to stop recording and generate summary.md.", flush=True)

    def add_subscription(self, topic, type_name, required=False):
        if topic in self.subscriptions_by_topic:
            return
        try:
            message_type = get_message(type_name)
            subscription = self.create_subscription(
                message_type,
                topic,
                lambda msg, t=topic, ty=type_name: self.record_message(t, ty, msg),
                self.qos,
            )
        except Exception as exc:
            warning = f"Could not subscribe to {topic} [{type_name}]: {type(exc).__name__}: {exc}"
            self.warnings.append(warning)
            self.get_logger().warning(warning)
            return
        self.subscriptions_by_topic[topic] = subscription
        self.topic_types[topic] = type_name
        label = "required" if required else "discovered trajectory-related"
        self.get_logger().info(f"Subscribed ({label}): {topic} [{type_name}]")

    def discover_trajectory_topics(self):
        for topic, type_names in self.get_topic_names_and_types():
            if not TRAJECTORY_PATTERN.search(topic):
                continue
            current_types = list(type_names)
            if self.discovered_topic_types.get(topic) != current_types:
                self.discovered_topic_types[topic] = current_types
                discovery_record = {
                    "receive_wall_time": wall_iso(time.time()),
                    "topic": topic,
                    "types": current_types,
                    "search_pattern": "predicted|projected|trajectory|traj|path|marker",
                }
                with (self.run_dir / "trajectory_topic_discovery.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(discovery_record, ensure_ascii=False, separators=(",", ":")) + "\n")
            if topic in self.subscriptions_by_topic or not type_names:
                continue
            self.add_subscription(topic, type_names[0], required=False)

    def stream_for(self, topic):
        if topic not in self.files:
            filename = topic_filename(topic)
            self.file_names[topic] = filename
            self.files[topic] = (self.run_dir / filename).open("a", encoding="utf-8", buffering=1)
        return self.files[topic]

    def record_message(self, topic, type_name, message):
        received_wall = time.time()
        ros_stamp = message_stamp(message)
        try:
            contents = message_to_ordereddict(message)
        except Exception as exc:
            contents = {"conversion_error": f"{type(exc).__name__}: {exc}", "repr": repr(message)}

        record = {
            "receive_wall_time": wall_iso(received_wall),
            "receive_wall_unix_sec": received_wall,
            "ros_message_stamp": ros_stamp,
            "topic": topic,
            "type": type_name,
            "message": contents,
        }
        stream = self.stream_for(topic)
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=True, separators=(",", ":")) + "\n")
        stream.flush()
        self.message_counts[topic] = self.message_counts.get(topic, 0) + 1
        self.capture_summary_sample(topic, message, received_wall, ros_stamp)

    def capture_summary_sample(self, topic, message, wall, ros_stamp):
        ros_seconds = ros_stamp["seconds"] if ros_stamp is not None else None
        if topic == "/goal_pose":
            self.goals.append({
                "wall": wall,
                "ros": ros_seconds,
                "x": float(message.pose.position.x),
                "y": float(message.pose.position.y),
                "frame": message.header.frame_id,
            })
        elif topic == "/social_nav_diffusion/policy_debug":
            complete_text = str(message.data)
            fields = parse_policy_debug(complete_text)
            self.policy.append({"wall": wall, "ros": ros_seconds, "text": complete_text, "fields": fields})
        elif topic == "/cpr_j100_0001/cmd_vel":
            v, w = twist_values(message)
            self.policy_cmd.append({"wall": wall, "ros": ros_seconds, "v": v, "w": w})
        elif topic == "/cpr_j100_0001/platform/cmd_vel":
            v, w = twist_values(message)
            self.platform_cmd.append({"wall": wall, "ros": ros_seconds, "v": v, "w": w})
        elif topic == "/cpr_j100_0001/platform/odom/filtered":
            v, w = twist_values(message)
            self.odom.append({
                "wall": wall,
                "ros": ros_seconds,
                "v": v,
                "w": w,
                "x": float(message.pose.pose.position.x),
                "y": float(message.pose.pose.position.y),
            })

    def policy_values(self, sample):
        fields = sample["fields"]
        return {
            "raw_v": finite_number(fields.get("raw_model_v_before_conversion")),
            "raw_r_or_w": finite_number(fields.get("raw_model_r_or_w_before_conversion")),
            "converted_v": finite_number(fields.get("converted_cmd_linear")),
            "converted_w": finite_number(fields.get("converted_cmd_angular")),
            "final_v": finite_number(fields.get("final_cmd_linear")),
            "final_w": finite_number(fields.get("final_cmd_angular")),
            "action_type": fields.get("raw_action_type", "unknown"),
            "command_source": fields.get("command_source", "unknown"),
            "distance_to_goal": finite_number(fields.get("distance_to_goal")),
        }

    def first_policy_stage(self, goal_time, stage):
        for sample in self.policy:
            if sample["wall"] < goal_time:
                continue
            values = self.policy_values(sample)
            if stage == "model" and is_nonzero(values["raw_v"], values["raw_r_or_w"]):
                return {**sample, **values}
            if stage == "final" and is_nonzero(values["final_v"], values["final_w"]):
                return {**sample, **values}
        return None

    def format_stage(self, sample, fields):
        if sample is None:
            return "not observed"
        values = ", ".join(f"{name}={sample.get(name)!r}" for name in fields)
        return f"{wall_iso(sample['wall'])} ({values})"

    def inference_gaps(self, goal_time):
        samples = [sample for sample in self.policy if sample["wall"] >= goal_time]
        if len(samples) < 3:
            return [], None
        gaps = [b["wall"] - a["wall"] for a, b in zip(samples, samples[1:])]
        median_gap = statistics.median(gaps)
        threshold = max(1.5, median_gap * 1.75)
        found = [
            (samples[index], samples[index + 1], gap)
            for index, gap in enumerate(gaps)
            if gap > threshold
        ]
        return found, median_gap

    def overshoot_event(self, goal_time):
        distance_samples = []
        for sample in self.policy:
            if sample["wall"] < goal_time:
                continue
            distance = self.policy_values(sample)["distance_to_goal"]
            if distance is not None:
                distance_samples.append((sample["wall"], distance))
        if len(distance_samples) < 3:
            return None, "insufficient distance_to_goal samples"
        min_index = min(range(len(distance_samples)), key=lambda i: distance_samples[i][1])
        minimum = distance_samples[min_index]
        later = distance_samples[min_index + 1 :]
        if not later:
            return None, "minimum distance occurred at the end of the run"
        peak = max(later, key=lambda item: item[1])
        increase = peak[1] - minimum[1]
        if increase <= 0.10:
            return None, f"no >0.10 m post-minimum increase (largest increase={increase:.3f} m)"
        return {
            "minimum_time": minimum[0],
            "minimum_distance": minimum[1],
            "event_time": peak[0],
            "later_distance": peak[1],
            "increase": increase,
        }, "heuristic overshoot detected from policy distance_to_goal"

    def write_manifest(self, final):
        manifest = {
            "run_directory": str(self.run_dir),
            "started_wall_time": wall_iso(self.started_wall),
            "stopped_wall_time": wall_iso(time.time()) if final else None,
            "required_topics": REQUIRED_TOPICS,
            "subscribed_topic_types": self.topic_types,
            "topic_log_files": self.file_names,
            "message_counts": self.message_counts,
            "warnings": self.warnings,
            "complete": final,
        }
        with (self.run_dir / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")

    def write_summary(self):
        stopped_wall = time.time()
        goal = self.goals[0] if self.goals else None
        goal_time = goal["wall"] if goal else self.started_wall

        model = self.first_policy_stage(goal_time, "model")
        final = self.first_policy_stage(goal_time, "final")
        policy_cmd = first_after(self.policy_cmd, goal_time, lambda s: is_nonzero(s["v"], s["w"]))
        platform_cmd = first_after(self.platform_cmd, goal_time, lambda s: is_nonzero(s["v"], s["w"]))
        odom_motion = first_after(self.odom, goal_time, lambda s: is_nonzero(s["v"], s["w"]))

        stages = [
            ("Goal", goal),
            ("Predicted/model action", model),
            ("Converted/final command", final),
            ("/cpr_j100_0001/cmd_vel", policy_cmd),
            ("/cpr_j100_0001/platform/cmd_vel", platform_cmd),
            ("Filtered odom motion", odom_motion),
        ]

        gaps, median_gap = self.inference_gaps(goal_time)
        policy_holds = detect_holds([s for s in self.policy_cmd if s["wall"] >= goal_time])
        platform_holds = detect_holds([s for s in self.platform_cmd if s["wall"] >= goal_time])
        if goal is None:
            overshoot, overshoot_note = None, "no goal was observed"
        else:
            overshoot, overshoot_note = self.overshoot_event(goal_time)

        correlation = "not assessable"
        if overshoot is None:
            correlation = f"No correlation established: {overshoot_note}."
        else:
            event_time = overshoot["event_time"]
            hold_overlap = any(start["wall"] - 0.5 <= event_time <= end["wall"] + 0.5 for start, end in platform_holds)
            gap_overlap = any(start["wall"] <= event_time <= end["wall"] for start, end, _ in gaps)
            if hold_overlap or gap_overlap:
                parts = []
                if hold_overlap:
                    parts.append("a platform command-hold interval")
                if gap_overlap:
                    parts.append("a policy-debug receive gap")
                correlation = "Temporal overlap found with " + " and ".join(parts) + ". This is correlation, not proof of cause."
            else:
                correlation = "Heuristic overshoot was detected, but it did not overlap a detected hold or policy-debug receive gap."

        lines = [
            "# SocialNav Monitor Summary",
            "",
            f"- Run directory: `{self.run_dir}`",
            f"- Start: {wall_iso(self.started_wall)}",
            f"- Stop: {wall_iso(stopped_wall)}",
            f"- Duration: {stopped_wall - self.started_wall:.3f} s",
            "",
            "## First Events",
            "",
        ]
        if goal:
            lines.append(f"- First goal receive time: {wall_iso(goal['wall'])} (x={goal['x']:.3f}, y={goal['y']:.3f}, frame={goal['frame']!r})")
        else:
            lines.append("- First goal receive time: not observed")
        lines.extend([
            f"- First nonzero predicted/model action: {self.format_stage(model, ['raw_v', 'raw_r_or_w', 'action_type'])}",
            f"- First nonzero converted/final command: {self.format_stage(final, ['converted_v', 'converted_w', 'final_v', 'final_w'])}",
            f"- First nonzero `/cpr_j100_0001/cmd_vel`: {self.format_stage(policy_cmd, ['v', 'w'])}",
            f"- First nonzero `/cpr_j100_0001/platform/cmd_vel`: {self.format_stage(platform_cmd, ['v', 'w'])}",
            f"- First nonzero filtered odom velocity: {self.format_stage(odom_motion, ['v', 'w'])}",
            "",
            "## Stage Delays",
            "",
            "| From | To | Delay (s) |",
            "|---|---|---:|",
        ])
        for (from_name, before), (to_name, after) in zip(stages, stages[1:]):
            delay = "not available" if before is None or after is None else f"{after['wall'] - before['wall']:.6f}"
            lines.append(f"| {from_name} | {to_name} | {delay} |")

        lines.extend(["", "## Action And Velocity Comparison", ""])
        if model and final:
            lines.append(f"- Raw model linear vs converted linear: {signed_magnitude_result(model['raw_v'], final['converted_v'])}.")
            if "Rot" in str(model["action_type"]):
                lines.append(f"- Raw model angular vs converted angular: {signed_magnitude_result(model['raw_r_or_w'], final['converted_w'])}.")
            else:
                lines.append(f"- Raw second action field is `{model['action_type']}` dependent; direct angular sign comparison is not assumed.")
            lines.append(f"- Converted vs final linear: {signed_magnitude_result(final['converted_v'], final['final_v'])}.")
            lines.append(f"- Converted vs final angular: {signed_magnitude_result(final['converted_w'], final['final_w'])}.")
        else:
            lines.append("- Model-to-final comparison: unavailable because one or both stages were not observed.")
        if final and policy_cmd:
            lines.append(f"- Final vs policy cmd_vel linear: {signed_magnitude_result(final['final_v'], policy_cmd['v'])}.")
            lines.append(f"- Final vs policy cmd_vel angular: {signed_magnitude_result(final['final_w'], policy_cmd['w'])}.")
        if policy_cmd and platform_cmd:
            lines.append(f"- Policy vs platform cmd_vel linear: {signed_magnitude_result(policy_cmd['v'], platform_cmd['v'])}.")
            lines.append(f"- Policy vs platform cmd_vel angular: {signed_magnitude_result(policy_cmd['w'], platform_cmd['w'])}.")
        if platform_cmd and odom_motion:
            lines.append(f"- Platform command vs odom linear: {signed_magnitude_result(platform_cmd['v'], odom_motion['v'])}.")
            lines.append(f"- Platform command vs odom angular: {signed_magnitude_result(platform_cmd['w'], odom_motion['w'])}.")
        lines.append("- Comparisons use first nonzero samples and are approximate because topic callbacks are asynchronous.")

        lines.extend([
            "",
            "## Inference Gaps, Command Holds, And Overshoot",
            "",
            f"- Median policy_debug receive interval: {median_gap:.6f} s" if median_gap is not None else "- Median policy_debug receive interval: insufficient samples",
            f"- Large policy_debug receive gaps: {len(gaps)}",
            f"- Nonzero policy cmd_vel hold periods >= 0.5 s: {len(policy_holds)}",
            f"- Nonzero platform cmd_vel hold periods >= 0.5 s: {len(platform_holds)}",
            f"- Overshoot assessment: {overshoot_note}.",
            f"- Hold/gap correlation assessment: {correlation}",
            "- Policy-debug gaps are only a proxy for inference gaps because the debug topic is published by a separate 1 Hz timer.",
        ])
        if overshoot is not None:
            lines.append(
                f"- Overshoot heuristic details: minimum={overshoot['minimum_distance']:.3f} m at "
                f"{wall_iso(overshoot['minimum_time'])}; later={overshoot['later_distance']:.3f} m at "
                f"{wall_iso(overshoot['event_time'])}; increase={overshoot['increase']:.3f} m."
            )
        for index, (start, end, gap) in enumerate(gaps[:20], 1):
            lines.append(f"- Gap {index}: {wall_iso(start['wall'])} to {wall_iso(end['wall'])}, {gap:.3f} s")

        lines.extend(["", "## Topic Counts", ""])
        for topic in sorted(self.topic_types):
            lines.append(f"- `{topic}` [{self.topic_types[topic]}]: {self.message_counts.get(topic, 0)} messages")
        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)

        with (self.run_dir / "summary.md").open("w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")

    def finish(self):
        for stream in self.files.values():
            stream.flush()
            stream.close()
        self.write_summary()
        self.write_manifest(final=True)
        print(f"\nRecording stopped. Summary: {self.run_dir / 'summary.md'}", flush=True)


def main():
    rclpy.init()
    monitor = SocialNavMonitor()
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.finish()
        monitor.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
