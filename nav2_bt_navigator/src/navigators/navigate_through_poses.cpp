// Copyright (c) 2021 Samsung Research
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include <string>
#include <set>
#include <memory>
#include <limits>
#include <cmath>
#include <iomanip>
#include "nav2_bt_navigator/navigators/navigate_through_poses.hpp"

namespace nav2_bt_navigator
{

static inline double yawDeg(const geometry_msgs::msg::Quaternion &q)
{
  const double n = std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
  double x=q.x, y=q.y, z=q.z, w=q.w;
  if (n > 1e-12) { x/=n; y/=n; z/=n; w/=n; }
  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  return std::atan2(siny_cosp, cosy_cosp) * 180.0 / M_PI;
}

static inline void logPoseStamped(
  rclcpp::Logger logger, size_t idx, const geometry_msgs::msg::PoseStamped &ps)
{
  const auto &p = ps.pose.position;
  const double yaw_deg = yawDeg(ps.pose.orientation);
  const double st = rclcpp::Time(ps.header.stamp).seconds();
  RCLCPP_INFO(
    logger,
    "[NTP][pose][%03zu] stamp=%.3f frame='%s' x=%.3f y=%.3f z=%.3f yaw_deg=%.1f",
    idx, st, ps.header.frame_id.c_str(), p.x, p.y, p.z, yaw_deg);
}

bool
NavigateThroughPosesNavigator::configure(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent_node,
  std::shared_ptr<nav2_util::OdomSmoother> odom_smoother)
{
  start_time_ = rclcpp::Time(0);
  auto node = parent_node.lock();

  if (!node->has_parameter("goals_blackboard_id")) {
    node->declare_parameter("goals_blackboard_id", std::string("goals"));
  }

  goals_blackboard_id_ = node->get_parameter("goals_blackboard_id").as_string();

  if (!node->has_parameter("path_blackboard_id")) {
    node->declare_parameter("path_blackboard_id", std::string("path"));
  }

  path_blackboard_id_ = node->get_parameter("path_blackboard_id").as_string();

  // Odometry smoother object for getting current speed
  odom_smoother_ = odom_smoother;

  // ---- LOGS: confirm config wiring
  RCLCPP_INFO(
    logger_,
    "[NTP][configure] goals_bb_id='%s' path_bb_id='%s'",
    goals_blackboard_id_.c_str(), path_blackboard_id_.c_str());

  // Show default BT
  std::string default_bt = getDefaultBTFilepath(parent_node);
  RCLCPP_INFO(logger_, "[NTP][configure] default BT xml: %s", default_bt.c_str());

  // Optional prune params (off by default)
  if (!node->has_parameter("prune_behind_goals")) {
    node->declare_parameter("prune_behind_goals", true);
  }
  if (!node->has_parameter("prune_fwd_thresh_m")) {
    node->declare_parameter("prune_fwd_thresh_m", 0.25);
  }
  if (!node->has_parameter("prune_dist_thresh_m")) {
    node->declare_parameter("prune_dist_thresh_m", 0.75);
  }
  prune_behind_goals_ = node->get_parameter("prune_behind_goals").as_bool();
  prune_fwd_thresh_m_ = node->get_parameter("prune_fwd_thresh_m").as_double();
  prune_dist_thresh_m_ = node->get_parameter("prune_dist_thresh_m").as_double();

  RCLCPP_WARN(logger_,
    "[NTP][prune] enabled=%s fwd_thresh=%.2f dist_thresh=%.2f",
    prune_behind_goals_ ? "true" : "false",
    prune_fwd_thresh_m_, prune_dist_thresh_m_);

  prune_behind_goals_ = true;
  
  return true;
}

std::string
NavigateThroughPosesNavigator::getDefaultBTFilepath(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent_node)
{
  std::string default_bt_xml_filename;
  auto node = parent_node.lock();

  if (!node->has_parameter("default_nav_through_poses_bt_xml")) {
    std::string pkg_share_dir =
      ament_index_cpp::get_package_share_directory("nav2_bt_navigator");
    node->declare_parameter<std::string>(
      "default_nav_through_poses_bt_xml",
      pkg_share_dir +
      "/behavior_trees/navigate_through_poses_w_replanning_and_recovery.xml");
  }

  node->get_parameter("default_nav_through_poses_bt_xml", default_bt_xml_filename);

  return default_bt_xml_filename;
}

bool
NavigateThroughPosesNavigator::goalReceived(ActionT::Goal::ConstSharedPtr goal)
{
  const std::string current_bt = bt_action_server_->getCurrentBTFilename();
  const std::string default_bt = bt_action_server_->getDefaultBTFilename();

  // --- Decide which BT to use ---
  std::string chosen_bt;
  if (goal->behavior_tree.empty()) {
    // Keep running tree if one is already loaded; otherwise use default
    chosen_bt = !current_bt.empty() ? current_bt : default_bt;
    RCLCPP_INFO(
      logger_,
      "[NTP][goalReceived] empty behavior_tree → use '%s' (current='%s' default='%s')",
      chosen_bt.c_str(), current_bt.c_str(), default_bt.c_str());
  } else {
    chosen_bt = goal->behavior_tree;
    RCLCPP_INFO(
      logger_,
      "[NTP][goalReceived] requested BT: '%s' (current='%s')",
      chosen_bt.c_str(), current_bt.c_str());
  }

  // --- Load only if necessary ---
  if (current_bt.empty() || chosen_bt != current_bt) {
    if (!bt_action_server_->loadBehaviorTree(chosen_bt)) {
      RCLCPP_ERROR(logger_, "Failed to load BT XML: '%s'. Navigation canceled.", chosen_bt.c_str());
      return false;
    }
  }

  RCLCPP_INFO(
    logger_,
    "[NTP][goalReceived] t=%.3f poses=%zu BT: using='%s'",
    clock_->now().seconds(), goal->poses.size(),
    bt_action_server_->getCurrentBTFilename().c_str());

  // Log poses and install them
  for (size_t i = 0; i < goal->poses.size(); ++i) {
    logPoseStamped(logger_, i, goal->poses[i]);
  }
  initializeGoalPoses(goal);
  return true;
}

void
NavigateThroughPosesNavigator::goalCompleted(
  typename ActionT::Result::SharedPtr /*result*/,
  const nav2_behavior_tree::BtStatus /*final_bt_status*/)
{
}

void
NavigateThroughPosesNavigator::onLoop()
{
  using namespace nav2_util::geometry_utils;  // NOLINT

  // action server feedback (pose, duration of task,
  // number of recoveries, and distance remaining to goal, etc)
  auto feedback_msg = std::make_shared<ActionT::Feedback>();

  auto blackboard = bt_action_server_->getBlackboard();

  Goals goal_poses;
  blackboard->get<Goals>(goals_blackboard_id_, goal_poses);

  RCLCPP_INFO_THROTTLE(
    logger_, *clock_, 1000,
    "[NTP][loop] goals_remaining=%zu", goal_poses.size());

  if (goal_poses.size() == 0) {
    bt_action_server_->publishFeedback(feedback_msg);
    return;
  }

  geometry_msgs::msg::PoseStamped current_pose;
  
  if (!nav2_util::getCurrentPose(
        current_pose, *feedback_utils_.tf,
        feedback_utils_.global_frame, feedback_utils_.robot_frame,
        feedback_utils_.transform_tolerance)) {
    RCLCPP_WARN_THROTTLE(
      logger_, *clock_, 2000,
      "[NTP] TF getCurrentPose() failed; skipping prune/feedback this tick.");
    bt_action_server_->publishFeedback(feedback_msg);
    return;
  }

  // --- OPTIONAL PRUNE: drop any leading goals that are clearly behind the robot ---
  if (prune_behind_goals_) {
    // get robot yaw
    const auto &q = current_pose.pose.orientation;
    const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                                  1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    bool pruned = false;
    size_t dropped = 0;
    while (!goal_poses.empty()) {
      const auto &g = goal_poses.front().pose.position;
      const double dx = g.x - current_pose.pose.position.x;
      const double dy = g.y - current_pose.pose.position.y;
      const double fwd =  dx * cy + dy * sy;               // +fwd is ahead along robot heading
      const double dist = std::hypot(dx, dy);
      if (fwd < -prune_fwd_thresh_m_ && dist > prune_dist_thresh_m_) {
        ++dropped;
        pruned = true;
        RCLCPP_WARN(
          logger_,
          "[NTP][prune] dropping leading goal behind robot: fwd=%.2f< -%.2f, dist=%.2f>%.2f "
          "(remaining before drop: %zu)",
          fwd, prune_fwd_thresh_m_, dist, prune_dist_thresh_m_, goal_poses.size());
        goal_poses.erase(goal_poses.begin());
      } else {
        break;
      }
    }
    if (pruned) {
      blackboard->set<Goals>(goals_blackboard_id_, goal_poses);
      RCLCPP_INFO(logger_, "[NTP][prune] dropped %zu goals; %zu remain", dropped, goal_poses.size());
      // If we pruned all goals, exit early — BT will handle completion state on next tick
      if (goal_poses.empty()) {
        bt_action_server_->publishFeedback(feedback_msg);
        return;
      }
    }
  }

  RCLCPP_INFO_THROTTLE(
    logger_, *clock_, 1000,
    "[NTP][loop][robot] frame='%s' x=%.3f y=%.3f z=%.3f yaw=%.1fdeg",
    current_pose.header.frame_id.c_str(),
    current_pose.pose.position.x,
    current_pose.pose.position.y,
    current_pose.pose.position.z,
    yawDeg(current_pose.pose.orientation));

  try {
    // Get current path points
    nav_msgs::msg::Path current_path;
    blackboard->get<nav_msgs::msg::Path>(path_blackboard_id_, current_path);

    RCLCPP_INFO_THROTTLE(
      logger_, *clock_, 1500,
      "[NTP][loop][path] frame='%s' poses=%zu",
      current_path.header.frame_id.c_str(),
      current_path.poses.size());

    const size_t K = std::min<size_t>(current_path.poses.size(), 3);
    for (size_t i = 0; i < K; ++i) {
      RCLCPP_INFO_THROTTLE(
        logger_, *clock_, 1500,
        "  [path.%02zu] x=%.3f y=%.3f yaw=%.1fdeg",
        i,
        current_path.poses[i].pose.position.x,
        current_path.poses[i].pose.position.y,
        yawDeg(current_path.poses[i].pose.orientation));
    }

    // Find the closest pose to current pose on global path
    auto find_closest_pose_idx =
      [&current_pose, &current_path]() {
        size_t closest_pose_idx = 0;
        double curr_min_dist = std::numeric_limits<double>::max();
        for (size_t curr_idx = 0; curr_idx < current_path.poses.size(); ++curr_idx) {
          double curr_dist = nav2_util::geometry_utils::euclidean_distance(
            current_pose, current_path.poses[curr_idx]);
          if (curr_dist < curr_min_dist) {
            curr_min_dist = curr_dist;
            closest_pose_idx = curr_idx;
          }
        }
        return closest_pose_idx;
      };

    size_t closest_idx_dbg = (current_path.poses.empty() ? 0 : find_closest_pose_idx());
    if (!current_path.poses.empty()) {
      double d_to_closest = nav2_util::geometry_utils::euclidean_distance(
        current_pose, current_path.poses[closest_idx_dbg]);
      RCLCPP_INFO_THROTTLE(
        logger_, *clock_, 1000,
        "[NTP][loop][path] closest_idx=%zu d=%.2f m",
        closest_idx_dbg, d_to_closest);
    }

    // Calculate distance on the path
    double distance_remaining =
      nav2_util::geometry_utils::calculate_path_length(current_path, find_closest_pose_idx());

    // Default value for time remaining
    rclcpp::Duration estimated_time_remaining = rclcpp::Duration::from_seconds(0.0);

    // Get current speed
    geometry_msgs::msg::Twist current_odom = odom_smoother_->getTwist();
    double current_linear_speed = std::hypot(current_odom.linear.x, current_odom.linear.y);

    RCLCPP_INFO_THROTTLE(
      logger_, *clock_, 1000,
      "[NTP][loop][speed] vx=%.2f vy=%.2f v=%.2f m/s | dist_remaining=%.2f m",
      current_odom.linear.x, current_odom.linear.y, current_linear_speed, distance_remaining);

    // Calculate estimated time taken to goal if speed is higher than 1cm/s
    // and at least 10cm to go
    if ((std::abs(current_linear_speed) > 0.01) && (distance_remaining > 0.1)) {
      estimated_time_remaining =
        rclcpp::Duration::from_seconds(distance_remaining / std::abs(current_linear_speed));
    }

    feedback_msg->distance_remaining = distance_remaining;
    feedback_msg->estimated_time_remaining = estimated_time_remaining;

    // -------- ADDED: diagnostics for backward targeting hypothesis --------
    // 1) Distance to first goal
    const auto &G0 = goal_poses.front();
    double d_to_first_goal = nav2_util::geometry_utils::euclidean_distance(current_pose, G0);
    // 2) Forward/behind w.r.t robot heading for first GOAL
    const double ryaw_rad =
      yawDeg(current_pose.pose.orientation) * M_PI / 180.0;
    const double dxg = G0.pose.position.x - current_pose.pose.position.x;
    const double dyg = G0.pose.position.y - current_pose.pose.position.y;
    const double fwd_g0 =  std::cos(ryaw_rad)*dxg + std::sin(ryaw_rad)*dyg;  // >0 ahead
    const double lat_g0 = -std::sin(ryaw_rad)*dxg + std::cos(ryaw_rad)*dyg;

    // 3) First PATH segment yaw vs robot yaw
    double seg0_yaw_deg = 0.0, seg0_cos = 1.0;
    if (current_path.poses.size() >= 2) {
      const auto &A = current_path.poses[0].pose.position;
      const auto &B = current_path.poses[1].pose.position;
      seg0_yaw_deg = std::atan2(B.y - A.y, B.x - A.x) * 180.0 / M_PI;
      const double dyaw = (seg0_yaw_deg - yawDeg(current_pose.pose.orientation)) * M_PI / 180.0;
      seg0_cos = std::cos(dyaw);
    }

    RCLCPP_INFO_THROTTLE(
      logger_, *clock_, 800,
      "[NTP][loop][diag] d_to_first_goal=%.2f m | first_goal fwd=%.2f lat=%.2f | seg0_yaw=%.1f° cos(seg0,robot)=%.3f",
      d_to_first_goal, fwd_g0, lat_g0, seg0_yaw_deg, seg0_cos);

    if (fwd_g0 < 0.0) {
      RCLCPP_WARN_THROTTLE(
        logger_, *clock_, 1500,
        "[NTP][loop][diag] FIRST GOAL is behind the robot (fwd=%.2f).", fwd_g0);
    }
    if (current_path.poses.size() >= 2 && seg0_cos <= 0.0) {
      RCLCPP_WARN_THROTTLE(
        logger_, *clock_, 1500,
        "[NTP][loop][diag] First PATH segment points backward wrt robot (cos<=0).");
    }
  } catch (...) {
    // Ignore
  }

  int recovery_count = 0;
  blackboard->get<int>("number_recoveries", recovery_count);
  feedback_msg->number_of_recoveries = recovery_count;
  feedback_msg->current_pose = current_pose;
  feedback_msg->navigation_time = clock_->now() - start_time_;
  feedback_msg->number_of_poses_remaining = goal_poses.size();

  bt_action_server_->publishFeedback(feedback_msg);
}

void
NavigateThroughPosesNavigator::onPreempt(ActionT::Goal::ConstSharedPtr goal)
{
  RCLCPP_INFO(logger_, "Received goal preemption request");

  const std::string current_bt = bt_action_server_->getCurrentBTFilename();

  RCLCPP_INFO(
    logger_, "[NTP][preempt] pending poses=%zu bt='%s' @ t=%.3f",
    goal->poses.size(), goal->behavior_tree.c_str(), clock_->now().seconds());

  for (size_t i = 0; i < goal->poses.size(); ++i) {
    logPoseStamped(logger_, i, goal->poses[i]);
  }

  if (goal->behavior_tree.empty() || goal->behavior_tree == current_bt) {
    // Keep using the current tree; just accept the new poses
    initializeGoalPoses(bt_action_server_->acceptPendingGoal());
  } else {
    RCLCPP_WARN(
        logger_,
        "[NTP][preempt] rejected: requested BT '%s' != current '%s'. "
        "Cancel the current goal if you need to switch trees.",
        goal->behavior_tree.c_str(), current_bt.c_str());
    RCLCPP_WARN(
      logger_,
      "Preemption request was rejected since the requested BT XML file is not the same "
      "as the one that the current goal is executing. Preemption with a new BT is invalid "
      "since it would require cancellation of the previous goal instead of true preemption."
      "\nCancel the current goal and send a new action request if you want to use a "
      "different BT XML file. For now, continuing to track the last goal until completion.");
    bt_action_server_->terminatePendingGoal();
  }
}

void
NavigateThroughPosesNavigator::initializeGoalPoses(ActionT::Goal::ConstSharedPtr goal)
{
  if (goal->poses.size() > 0) {
    RCLCPP_INFO(
      logger_, "Begin navigating from current location through %zu poses to (%.2f, %.2f)",
      goal->poses.size(), goal->poses.back().pose.position.x, goal->poses.back().pose.position.y);
  }
  
  // --- ADDED: log the series as installed on the blackboard ---
  RCLCPP_INFO(
    logger_, "[NTP][initialize] installing poses=%zu @ t=%.3f",
    goal->poses.size(), clock_->now().seconds());
  for (size_t i = 0; i < goal->poses.size(); ++i) {
    logPoseStamped(logger_, i, goal->poses[i]);
  }
  // --- END LOGS ---
  
  // Reset state for new action feedback
  start_time_ = clock_->now();
  auto blackboard = bt_action_server_->getBlackboard();
  blackboard->set<int>("number_recoveries", 0);  // NOLINT

  // Update the goal pose on the blackboard
  blackboard->set<Goals>(goals_blackboard_id_, goal->poses);
}

}  // namespace nav2_bt_navigator
