// Copyright (c) 2020, Samsung Research America
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
// limitations under the License. Reserved.

#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <limits>

#include "Eigen/Core"
#include "nav2_smac_planner/smac_planner_hybrid.hpp"

// #define BENCHMARK_TESTING

namespace nav2_smac_planner
{

using namespace std::chrono;  // NOLINT
using rcl_interfaces::msg::ParameterType;
using std::placeholders::_1;

SmacPlannerHybrid::SmacPlannerHybrid()
: _a_star(nullptr),
  _collision_checker(nullptr, 1, nullptr),
  _smoother(nullptr),
  _costmap(nullptr),
  _costmap_downsampler(nullptr)
{
}

SmacPlannerHybrid::~SmacPlannerHybrid()
{
  RCLCPP_INFO(
    _logger, "Destroying plugin %s of type SmacPlannerHybrid",
    _name.c_str());
}

void SmacPlannerHybrid::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer>/*tf*/,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  _node = parent;
  auto node = parent.lock();
  _logger = node->get_logger();
  _clock = node->get_clock();
  _costmap = costmap_ros->getCostmap();
  _costmap_ros = costmap_ros;
  _name = name;
  _global_frame = costmap_ros->getGlobalFrameID();

  RCLCPP_INFO(_logger, "Configuring %s of type SmacPlannerHybrid", name.c_str());

  int angle_quantizations;
  double analytic_expansion_max_length_m;
  bool smooth_path;

  // General planner params
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".downsample_costmap", rclcpp::ParameterValue(false));
  node->get_parameter(name + ".downsample_costmap", _downsample_costmap);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".downsampling_factor", rclcpp::ParameterValue(1));
  node->get_parameter(name + ".downsampling_factor", _downsampling_factor);

  nav2_util::declare_parameter_if_not_declared(
    node, name + ".angle_quantization_bins", rclcpp::ParameterValue(72));
  node->get_parameter(name + ".angle_quantization_bins", angle_quantizations);
  _angle_bin_size = 2.0 * M_PI / angle_quantizations;
  _angle_quantizations = static_cast<unsigned int>(angle_quantizations);

  nav2_util::declare_parameter_if_not_declared(
    node, name + ".tolerance", rclcpp::ParameterValue(0.25));
  _tolerance = static_cast<float>(node->get_parameter(name + ".tolerance").as_double());
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".allow_unknown", rclcpp::ParameterValue(true));
  node->get_parameter(name + ".allow_unknown", _allow_unknown);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".max_iterations", rclcpp::ParameterValue(1000000));
  node->get_parameter(name + ".max_iterations", _max_iterations);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".max_on_approach_iterations", rclcpp::ParameterValue(1000));
  node->get_parameter(name + ".max_on_approach_iterations", _max_on_approach_iterations);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".smooth_path", rclcpp::ParameterValue(true));
  node->get_parameter(name + ".smooth_path", smooth_path);

  nav2_util::declare_parameter_if_not_declared(
    node, name + ".minimum_turning_radius", rclcpp::ParameterValue(0.4));
  node->get_parameter(name + ".minimum_turning_radius", _minimum_turning_radius_global_coords);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".cache_obstacle_heuristic", rclcpp::ParameterValue(false));
  node->get_parameter(name + ".cache_obstacle_heuristic", _search_info.cache_obstacle_heuristic);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".reverse_penalty", rclcpp::ParameterValue(2.0));
  node->get_parameter(name + ".reverse_penalty", _search_info.reverse_penalty);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".change_penalty", rclcpp::ParameterValue(0.0));
  node->get_parameter(name + ".change_penalty", _search_info.change_penalty);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".non_straight_penalty", rclcpp::ParameterValue(1.2));
  node->get_parameter(name + ".non_straight_penalty", _search_info.non_straight_penalty);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".cost_penalty", rclcpp::ParameterValue(2.0));
  node->get_parameter(name + ".cost_penalty", _search_info.cost_penalty);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".retrospective_penalty", rclcpp::ParameterValue(0.015));
  node->get_parameter(name + ".retrospective_penalty", _search_info.retrospective_penalty);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".analytic_expansion_ratio", rclcpp::ParameterValue(3.5));
  node->get_parameter(name + ".analytic_expansion_ratio", _search_info.analytic_expansion_ratio);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".analytic_expansion_max_length", rclcpp::ParameterValue(3.0));
  node->get_parameter(name + ".analytic_expansion_max_length", analytic_expansion_max_length_m);
  _search_info.analytic_expansion_max_length =
    analytic_expansion_max_length_m / _costmap->getResolution();

  nav2_util::declare_parameter_if_not_declared(
    node, name + ".max_planning_time", rclcpp::ParameterValue(5.0));
  node->get_parameter(name + ".max_planning_time", _max_planning_time);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".lookup_table_size", rclcpp::ParameterValue(20.0));
  node->get_parameter(name + ".lookup_table_size", _lookup_table_size);

  nav2_util::declare_parameter_if_not_declared(
    node, name + ".motion_model_for_search", rclcpp::ParameterValue(std::string("DUBIN")));
  node->get_parameter(name + ".motion_model_for_search", _motion_model_for_search);
  _motion_model = fromString(_motion_model_for_search);
  if (_motion_model == MotionModel::UNKNOWN) {
    RCLCPP_WARN(
      _logger,
      "Unable to get MotionModel search type. Given '%s', "
      "valid options are MOORE, VON_NEUMANN, DUBIN, REEDS_SHEPP, STATE_LATTICE.",
      _motion_model_for_search.c_str());
  }

  if (_max_on_approach_iterations <= 0) {
    RCLCPP_INFO(
      _logger, "On approach iteration selected as <= 0, "
      "disabling tolerance and on approach iterations.");
    _max_on_approach_iterations = std::numeric_limits<int>::max();
  }

  if (_max_iterations <= 0) {
    RCLCPP_INFO(
      _logger, "maximum iteration selected as <= 0, "
      "disabling maximum iterations.");
    _max_iterations = std::numeric_limits<int>::max();
  }

  RCLCPP_INFO(_logger,
    "[SMAC][Cfg] bins=%u (bin_size=%.3f rad = %.1f°)  tol=%.2f  allow_unknown=%s",
    _angle_quantizations, _angle_bin_size, _angle_bin_size * 180.0 / M_PI,
    _tolerance, _allow_unknown ? "Y" : "N");
  RCLCPP_INFO(_logger,
    "[SMAC][Cfg] penalties: reverse=%.3f change=%.3f non_straight=%.3f cost=%.3f retro=%.3f "
    "analytic_ratio=%.2f analytic_len=%.1f cells  min_turn_r=%.3f m",
    _search_info.reverse_penalty, _search_info.change_penalty,
    _search_info.non_straight_penalty, _search_info.cost_penalty,
    _search_info.retrospective_penalty, _search_info.analytic_expansion_ratio,
    _search_info.analytic_expansion_max_length, _minimum_turning_radius_global_coords);

  // convert to grid coordinates
  if (!_downsample_costmap) {
    _downsampling_factor = 1;
  }
  _search_info.minimum_turning_radius =
    _minimum_turning_radius_global_coords / (_costmap->getResolution() * _downsampling_factor);
  _lookup_table_dim =
    static_cast<float>(_lookup_table_size) /
    static_cast<float>(_costmap->getResolution() * _downsampling_factor);

  // Make sure its a whole number
  _lookup_table_dim = static_cast<float>(static_cast<int>(_lookup_table_dim));

  // Make sure its an odd number
  if (static_cast<int>(_lookup_table_dim) % 2 == 0) {
    RCLCPP_INFO(
      _logger,
      "Even sized heuristic lookup table size set %f, increasing size by 1 to make odd",
      _lookup_table_dim);
    _lookup_table_dim += 1.0;
  }

  // Initialize collision checker
  _collision_checker = GridCollisionChecker(_costmap, _angle_quantizations, node);
  _collision_checker.setFootprint(
    _costmap_ros->getRobotFootprint(),
    _costmap_ros->getUseRadius(),
    findCircumscribedCost(_costmap_ros));

  // Initialize A* template
  _a_star = std::make_unique<AStarAlgorithm<NodeHybrid>>(_motion_model, _search_info);
  _a_star->initialize(
    _allow_unknown,
    _max_iterations,
    _max_on_approach_iterations,
    _max_planning_time,
    _lookup_table_dim,
    _angle_quantizations);

  // Initialize path smoother
  if (smooth_path) {
    SmootherParams params;
    params.get(node, name);
    _smoother = std::make_unique<Smoother>(params);
    _smoother->initialize(_minimum_turning_radius_global_coords);
  }

  // Initialize costmap downsampler
  if (_downsample_costmap && _downsampling_factor > 1) {
    _costmap_downsampler = std::make_unique<CostmapDownsampler>();
    std::string topic_name = "downsampled_costmap";
    _costmap_downsampler->on_configure(
      node, _global_frame, topic_name, _costmap, _downsampling_factor);
  }

  const auto *cm = _costmap_ros->getCostmap();
  RCLCPP_INFO(_logger,
    "[SMAC][Map] frame='%s' origin(%.3f,%.3f) res=%.3f size=%ux%u downsample=%s factor=%d "
    "effective_res=%.3f",
    _global_frame.c_str(), cm->getOriginX(), cm->getOriginY(), cm->getResolution(),
    cm->getSizeInCellsX(), cm->getSizeInCellsY(),
    (_downsample_costmap && _downsampling_factor > 1) ? "Y" : "N",
    (_downsample_costmap ? _downsampling_factor : 1),
    cm->getResolution() * (_downsample_costmap ? _downsampling_factor : 1));

  _raw_plan_publisher = node->create_publisher<nav_msgs::msg::Path>("unsmoothed_plan", 1);

  RCLCPP_INFO(
    _logger, "Configured plugin %s of type SmacPlannerHybrid with "
    "maximum iterations %i, max on approach iterations %i, and %s. Tolerance %.2f."
    "Using motion model: %s.",
    _name.c_str(), _max_iterations, _max_on_approach_iterations,
    _allow_unknown ? "allowing unknown traversal" : "not allowing unknown traversal",
    _tolerance, toString(_motion_model).c_str());
}

void SmacPlannerHybrid::activate()
{
  RCLCPP_INFO(
    _logger, "Activating plugin %s of type SmacPlannerHybrid",
    _name.c_str());
  _raw_plan_publisher->on_activate();
  if (_costmap_downsampler) {
    _costmap_downsampler->on_activate();
  }
  auto node = _node.lock();
  // Add callback for dynamic parameters
  _dyn_params_handler = node->add_on_set_parameters_callback(
    std::bind(&SmacPlannerHybrid::dynamicParametersCallback, this, _1));
}

void SmacPlannerHybrid::deactivate()
{
  RCLCPP_INFO(
    _logger, "Deactivating plugin %s of type SmacPlannerHybrid",
    _name.c_str());
  _raw_plan_publisher->on_deactivate();
  if (_costmap_downsampler) {
    _costmap_downsampler->on_deactivate();
  }
  _dyn_params_handler.reset();
}

void SmacPlannerHybrid::cleanup()
{
  RCLCPP_INFO(
    _logger, "Cleaning up plugin %s of type SmacPlannerHybrid",
    _name.c_str());
  _a_star.reset();
  _smoother.reset();
  if (_costmap_downsampler) {
    _costmap_downsampler->on_cleanup();
    _costmap_downsampler.reset();
  }
  _raw_plan_publisher.reset();
}

nav_msgs::msg::Path SmacPlannerHybrid::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  std::lock_guard<std::mutex> lock_reinit(_mutex);
  steady_clock::time_point a = steady_clock::now();

  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*(_costmap->getMutex()));

  RCLCPP_INFO(_logger,
    "[SMAC][Costmap] origin(%.3f,%.3f) res=%.3f size=%ux%u",
    _costmap->getOriginX(), _costmap->getOriginY(),
    _costmap->getResolution(), _costmap->getSizeInCellsX(), _costmap->getSizeInCellsY());

  // Downsample costmap, if required
  nav2_costmap_2d::Costmap2D * costmap = _costmap;
  if (_costmap_downsampler) {
    costmap = _costmap_downsampler->downsample(_downsampling_factor);
    _collision_checker.setCostmap(costmap);
  }

  // [ADD DBG] DS result (effective resolution)
  const double eff_res = costmap->getResolution();
  RCLCPP_INFO(_logger,
    "[SMAC][DS] using_ds=%s factor=%d eff_res=%.3f",
    (_costmap_downsampler ? "Y" : "N"),
    (_costmap_downsampler ? _downsampling_factor : 1),
    eff_res);

  // [DBG][SMAC] Request summary
  const double t_now = _clock->now().seconds();
  const double res = costmap->getResolution();
  const unsigned int size_x = costmap->getSizeInCellsX();
  const unsigned int size_y = costmap->getSizeInCellsY();
  const bool using_ds = (_costmap_downsampler != nullptr && _downsampling_factor > 1);
  const int ds_factor = using_ds ? _downsampling_factor : 1;
  const double yaw_start = tf2::getYaw(start.pose.orientation);
  const double yaw_goal  = tf2::getYaw(goal.pose.orientation);
  RCLCPP_INFO(
    _logger,
    "[SMAC][PlanReq] t=%.3f frame='%s' res=%.3fm size=%ux%u ds=%s(f=%d) model=%s tol=%.2f "
    "max_t=%.2f iters(max=%d on_appr=%d) bins=%u",
    t_now, _global_frame.c_str(), res, size_x, size_y, using_ds ? "Y":"N", ds_factor,
    toString(_motion_model).c_str(), _tolerance, _max_planning_time, _max_iterations,
    _max_on_approach_iterations, _angle_quantizations);
  RCLCPP_INFO(
    _logger,
    "[SMAC][Start] W(%.3f, %.3f, z=%.3f) yaw=%.1f°  |  [SMAC][Goal] W(%.3f, %.3f, z=%.3f) yaw=%.1f°",
    start.pose.position.x, start.pose.position.y, start.pose.position.z, yaw_start * 180.0 / M_PI,
    goal.pose.position.x,  goal.pose.position.y,  goal.pose.position.z,  yaw_goal  * 180.0 / M_PI);

  // Set collision checker and costmap information
  _collision_checker.setFootprint(
    _costmap_ros->getRobotFootprint(),
    _costmap_ros->getUseRadius(),
    findCircumscribedCost(_costmap_ros));
  _a_star->setCollisionChecker(&_collision_checker);

  // Set starting point, in A* bin search coordinates
  unsigned int mx, my;
  if (!costmap->worldToMap(start.pose.position.x, start.pose.position.y, mx, my)) {
    RCLCPP_WARN(
      _logger,
      "[SMAC][PlanReq][OOB] START outside costmap: W(%.3f,%.3f) res=%.3f size=%ux%u origin=(%.3f,%.3f)",
      start.pose.position.x, start.pose.position.y, res, size_x, size_y,
      costmap->getOriginX(), costmap->getOriginY());
    throw std::runtime_error("Start pose is out of costmap!");
  }

  const unsigned char s_cost = costmap->getCost(mx, my);
  RCLCPP_INFO(_logger,
    "[SMAC][Start->Map] M(%u,%u) raw_yaw=%.1f° bin_size=%.1f°",
    mx, my, tf2::getYaw(start.pose.orientation) * 180.0 / M_PI,
    _angle_bin_size * 180.0 / M_PI);

  double orientation_bin = std::round(tf2::getYaw(start.pose.orientation) / _angle_bin_size);
  while (orientation_bin < 0.0) {
    orientation_bin += static_cast<float>(_angle_quantizations);
  }
  // This is needed to handle precision issues
  if (orientation_bin >= static_cast<float>(_angle_quantizations)) {
    orientation_bin -= static_cast<float>(_angle_quantizations);
  }

  const unsigned int start_mx = mx, start_my = my;
  const unsigned int start_bin = static_cast<unsigned int>(orientation_bin);

  _a_star->setStart(mx, my, static_cast<unsigned int>(orientation_bin));

  RCLCPP_INFO(
    _logger,
    "[SMAC][Start->Map] M(%u,%u) bin=%u (bin_size=%.4f rad, bins=%u)",
    start_mx, start_my, start_bin, _angle_bin_size, _angle_quantizations);

  // Set goal point, in A* bin search coordinates
  if (!costmap->worldToMap(goal.pose.position.x, goal.pose.position.y, mx, my)) {
    RCLCPP_WARN(
      _logger,
      "[SMAC][PlanReq][OOB] GOAL outside costmap: W(%.3f,%.3f) res=%.3f size=%ux%u origin=(%.3f,%.3f)",
      goal.pose.position.x, goal.pose.position.y, eff_res,
      costmap->getSizeInCellsX(), costmap->getSizeInCellsY(),
      costmap->getOriginX(), costmap->getOriginY());
    throw std::runtime_error("Goal pose is out of costmap!");
  }

  const unsigned char g_cost = costmap->getCost(mx, my);

  orientation_bin = std::round(tf2::getYaw(goal.pose.orientation) / _angle_bin_size);
  while (orientation_bin < 0.0) {
    orientation_bin += static_cast<float>(_angle_quantizations);
  }
  // This is needed to handle precision issues
  if (orientation_bin >= static_cast<float>(_angle_quantizations)) {
    orientation_bin -= static_cast<float>(_angle_quantizations);
  }

  const unsigned int goal_mx = mx, goal_my = my;
  const unsigned int goal_bin = static_cast<unsigned int>(orientation_bin);

  _a_star->setGoal(mx, my, static_cast<unsigned int>(orientation_bin));

  RCLCPP_INFO(
    _logger,
    "[SMAC][Goal ->Map] M(%u,%u) bin=%u",
    goal_mx, goal_my, goal_bin);

  if (start_mx == goal_mx && start_my == goal_my) {
    RCLCPP_WARN(_logger, "[SMAC][Chk] START and GOAL are in the same cell.");
  }
  if (s_cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
    RCLCPP_WARN(_logger, "[SMAC][Chk] START cell is LETHAL (%u).", static_cast<unsigned int>(s_cost));
  }
  if (g_cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
    RCLCPP_WARN(_logger, "[SMAC][Chk] GOAL cell is LETHAL (%u).", static_cast<unsigned int>(g_cost));
  }

  // Setup message
  nav_msgs::msg::Path plan;
  plan.header.stamp = _clock->now();
  plan.header.frame_id = _global_frame;
  geometry_msgs::msg::PoseStamped pose;
  pose.header = plan.header;
  pose.pose.position.z = 0.0;
  pose.pose.orientation.x = 0.0;
  pose.pose.orientation.y = 0.0;
  pose.pose.orientation.z = 0.0;
  pose.pose.orientation.w = 1.0;

  // Compute plan
  NodeHybrid::CoordinateVector path;
  int num_iterations = 0;
  std::string error;
  try {
    if (!_a_star->createPath(
        path, num_iterations, _tolerance / static_cast<float>(costmap->getResolution())))
    {
      if (num_iterations < _a_star->getMaxIterations()) {
        error = std::string("no valid path found");
      } else {
        error = std::string("exceeded maximum iterations");
      }
    }
  } catch (const std::runtime_error & e) {
    error = "invalid use: ";
    error += e.what();
  }

  if (!error.empty()) {
    RCLCPP_WARN(
      _logger,
      "[SMAC][PlanFail] %s: %s | start M(%u,%u) bin=%u  goal M(%u,%u) bin=%u  allow_unknown=%s min_turn_r=%.3f lr_dim=%.0f",
      _name.c_str(), error.c_str(),
      start_mx, start_my, start_bin, goal_mx, goal_my, goal_bin,
      _allow_unknown ? "Y" : "N",
      _search_info.minimum_turning_radius,
      _lookup_table_dim);
    return plan;
  }

  // Convert to world coordinates
  plan.poses.reserve(path.size());
  for (int i = path.size() - 1; i >= 0; --i) {
    pose.pose = getWorldCoords(path[i].x, path[i].y, costmap);
    pose.pose.orientation = getWorldOrientation(path[i].theta);
    plan.poses.push_back(pose);
  }

  if (!plan.poses.empty()) {
    const double yaw_start = tf2::getYaw(start.pose.orientation);
    const auto &p0 = plan.poses.front().pose.position;
    size_t behind_count = 0;
    double first_step = 0.0;
    if (plan.poses.size() >= 2) {
      const auto &p1 = plan.poses[1].pose.position;
      first_step = std::hypot(p1.x - p0.x, p1.y - p0.y);
      const double seg_yaw = std::atan2(p1.y - p0.y, p1.x - p0.x);
      const double cos_heading = std::cos(seg_yaw - yaw_start);
      RCLCPP_INFO(_logger,
        "[SMAC][InitSeg2] first_step=%.3f m  cos(seg,robot)=%.3f", first_step, cos_heading);
      if (cos_heading <= 0.0) {
        RCLCPP_WARN(_logger, "[SMAC][Diag] initial segment points backward wrt robot heading.");
      }
      if (first_step < eff_res * 0.5) {
        RCLCPP_WARN(_logger, "[SMAC][Diag] very short first step (%.3f m) — grid snap/quantization?", first_step);
      }
    }
    // Count “behind-me” points in first K
    const size_t K = std::min<size_t>(10, plan.poses.size());
    for (size_t i = 0; i < K; ++i) {
      const auto &p = plan.poses[i].pose.position;
      const double dx = p.x - start.pose.position.x;
      const double dy = p.y - start.pose.position.y;
      const double fwd =  dx * std::cos(yaw_start) + dy * std::sin(yaw_start);
      if (fwd < 0.0) behind_count++;
    }
    if (behind_count > 0) {
      RCLCPP_WARN(_logger,
        "[SMAC][Diag] %zu/%zu of first path points are BEHIND the robot.", behind_count, K);
    }
  }

  // [DBG][SMAC] Path stats + preview (pre-smoothing)
  auto path_len_m = 0.0;
  for (size_t i = 1; i < plan.poses.size(); ++i) {
    const auto & a = plan.poses[i-1].pose.position;
    const auto & b = plan.poses[i].pose.position;
    path_len_m += std::hypot(b.x - a.x, b.y - a.y);
  }
  const size_t N = plan.poses.size();
  const double avg_step = (N > 1) ? (path_len_m / (N - 1)) : 0.0;

  steady_clock::time_point b_dbg = steady_clock::now();
  duration<double> t_plan_dbg = duration_cast<duration<double>>(b_dbg - a);

  RCLCPP_INFO(
    _logger,
    "[SMAC][Path] poses=%zu len=%.2fm avg_step=%.2fm iters=%d plan_t=%.3fs",
    N, path_len_m, avg_step, num_iterations, t_plan_dbg.count());

  // Preview first few poses relative to START heading (ahead/behind)
  const size_t k = std::min<size_t>(6, (N > 0 ? N : 0));
  if (k > 0) {
    RCLCPP_INFO(_logger, "[SMAC][Preview:first %zu]", k);
    double prev_x = start.pose.position.x;
    double prev_y = start.pose.position.y;
    for (size_t i = 0; i < k; ++i) {
      const auto & p = plan.poses[i].pose;
      const double dx = p.position.x - start.pose.position.x;
      const double dy = p.position.y - start.pose.position.y;
      const double fwd =  dx * std::cos(yaw_start) + dy * std::sin(yaw_start);
      const double lat = -dx * std::sin(yaw_start) + dy * std::cos(yaw_start);
      const double step = std::hypot(p.position.x - prev_x, p.position.y - prev_y);
      const double yaw_i_deg = std::atan2(
        (i+1 < N ? plan.poses[i+1].pose.position.y - p.position.y : p.position.y - prev_y),
        (i+1 < N ? plan.poses[i+1].pose.position.x - p.position.x : p.position.x - prev_x)
      ) * 180.0 / M_PI;
      RCLCPP_INFO(
        _logger,
        "  [%02zu] (%.3f, %.3f) yaw≈%.1f°  fwd=%.2f lat=%.2f step=%.2f",
        i, p.position.x, p.position.y, yaw_i_deg, fwd, lat, step);
      prev_x = p.position.x; prev_y = p.position.y;
    }
  }

  // Publish raw path for debug
  if (_raw_plan_publisher->get_subscription_count() > 0) {
    _raw_plan_publisher->publish(plan);
  }

  // Find how much time we have left to do smoothing
  steady_clock::time_point b = steady_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(b - a);
  double time_remaining = _max_planning_time - static_cast<double>(time_span.count());

#ifdef BENCHMARK_TESTING
  std::cout << "It took " << time_span.count() * 1000 <<
    " milliseconds with " << num_iterations << " iterations." << std::endl;
#endif

  // Smooth plan
  if (_smoother && num_iterations > 1) {
    _smoother->smooth(plan, costmap, time_remaining);
  }

  // [DBG][SMAC] Post-smoothing stats
  {
    double len2 = 0.0;
    for (size_t i = 1; i < plan.poses.size(); ++i) {
      const auto & a = plan.poses[i-1].pose.position;
      const auto & b = plan.poses[i].pose.position;
      len2 += std::hypot(b.x - a.x, b.y - a.y);
    }
    // --- Begin extra instrumentation (after Preview block) ---
    if (!plan.poses.empty()) {
      // 1) How far did we snap the start to the grid cell center?
      double cx, cy;
      costmap->mapToWorld(start_mx, start_my, cx, cy);
      const double snap_dx = cx - start.pose.position.x;
      const double snap_dy = cy - start.pose.position.y;
      RCLCPP_INFO(
        _logger,
        "[SMAC][Quant] start->cell_center dx=%.03f dy=%.03f (res=%.2f)",
        snap_dx, snap_dy, res);

      // 2) Is the FIRST pose behind the robot along its heading?
      const auto &p0 = plan.poses.front().pose.position;
      const double dx0 = p0.x - start.pose.position.x;
      const double dy0 = p0.y - start.pose.position.y;
      const double fwd0 =  dx0 * std::cos(yaw_start) + dy0 * std::sin(yaw_start);
      const double lat0 = -dx0 * std::sin(yaw_start) + dy0 * std::cos(yaw_start);
      RCLCPP_INFO(
        _logger,
        "[SMAC][InitPose] fwd0=%.03f lat0=%.03f (first pose rel to start, +fwd=ahead)",
        fwd0, lat0);

      // 3) What’s the initial segment heading vs start heading?
      double seg_yaw = yaw_start;
      if (plan.poses.size() >= 2) {
        const auto &p1 = plan.poses[1].pose.position;
        seg_yaw = std::atan2(p1.y - p0.y, p1.x - p0.x);
      }
      auto wrapPi = [](double a){ while (a >  M_PI) a -= 2.0*M_PI; while (a < -M_PI) a += 2.0*M_PI; return a; };
      const double dtheta = wrapPi(seg_yaw - yaw_start);
      RCLCPP_INFO(
        _logger,
        "[SMAC][InitSeg] seg_yaw=%.1f° start_yaw=%.1f° dtheta=%.1f°",
        seg_yaw * 180.0 / M_PI, yaw_start * 180.0 / M_PI, dtheta * 180.0 / M_PI);
    }
    // --- End extra instrumentation ---
    RCLCPP_INFO(
      _logger,
      "[SMAC][Smooth] poses=%zu len=%.2fm Δlen=%.2fm (smoothed)",
      plan.poses.size(), len2, len2 /*post*/ - path_len_m /*pre*/);
  }

#ifdef BENCHMARK_TESTING
  steady_clock::time_point c = steady_clock::now();
  duration<double> time_span2 = duration_cast<duration<double>>(c - b);
  std::cout << "It took " << time_span2.count() * 1000 <<
    " milliseconds to smooth path." << std::endl;
#endif

  return plan;
}

rcl_interfaces::msg::SetParametersResult
SmacPlannerHybrid::dynamicParametersCallback(std::vector<rclcpp::Parameter> parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  std::lock_guard<std::mutex> lock_reinit(_mutex);

  bool reinit_collision_checker = false;
  bool reinit_a_star = false;
  bool reinit_downsampler = false;
  bool reinit_smoother = false;

  for (auto parameter : parameters) {
    const auto & type = parameter.get_type();
    const auto & name = parameter.get_name();

    if (type == ParameterType::PARAMETER_DOUBLE ||
        type == ParameterType::PARAMETER_BOOL   ||
        type == ParameterType::PARAMETER_INTEGER||
        type == ParameterType::PARAMETER_STRING) {
      RCLCPP_INFO(_logger, "[SMAC][Dyn] %s := %s",
        name.c_str(), parameter.value_to_string().c_str());
    }

    if (type == ParameterType::PARAMETER_DOUBLE) {
      if (name == _name + ".max_planning_time") {
        reinit_a_star = true;
        _max_planning_time = parameter.as_double();
      } else if (name == _name + ".tolerance") {
        _tolerance = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".lookup_table_size") {
        reinit_a_star = true;
        _lookup_table_size = parameter.as_double();
      } else if (name == _name + ".minimum_turning_radius") {
        reinit_a_star = true;
        if (_smoother) {
          reinit_smoother = true;
        }
        _minimum_turning_radius_global_coords = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".reverse_penalty") {
        reinit_a_star = true;
        _search_info.reverse_penalty = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".change_penalty") {
        reinit_a_star = true;
        _search_info.change_penalty = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".non_straight_penalty") {
        reinit_a_star = true;
        _search_info.non_straight_penalty = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".cost_penalty") {
        reinit_a_star = true;
        _search_info.cost_penalty = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".analytic_expansion_ratio") {
        reinit_a_star = true;
        _search_info.analytic_expansion_ratio = static_cast<float>(parameter.as_double());
      } else if (name == _name + ".analytic_expansion_max_length") {
        reinit_a_star = true;
        _search_info.analytic_expansion_max_length =
          static_cast<float>(parameter.as_double()) / _costmap->getResolution();
      }
    } else if (type == ParameterType::PARAMETER_BOOL) {
      if (name == _name + ".downsample_costmap") {
        reinit_downsampler = true;
        _downsample_costmap = parameter.as_bool();
      } else if (name == _name + ".allow_unknown") {
        reinit_a_star = true;
        _allow_unknown = parameter.as_bool();
      } else if (name == _name + ".cache_obstacle_heuristic") {
        reinit_a_star = true;
        _search_info.cache_obstacle_heuristic = parameter.as_bool();
      } else if (name == _name + ".smooth_path") {
        if (parameter.as_bool()) {
          reinit_smoother = true;
        } else {
          _smoother.reset();
        }
      }
    } else if (type == ParameterType::PARAMETER_INTEGER) {
      if (name == _name + ".downsampling_factor") {
        reinit_a_star = true;
        reinit_downsampler = true;
        _downsampling_factor = parameter.as_int();
      } else if (name == _name + ".max_iterations") {
        reinit_a_star = true;
        _max_iterations = parameter.as_int();
        if (_max_iterations <= 0) {
          RCLCPP_INFO(
            _logger, "maximum iteration selected as <= 0, "
            "disabling maximum iterations.");
          _max_iterations = std::numeric_limits<int>::max();
        }
      } else if (name == _name + ".max_on_approach_iterations") {
        reinit_a_star = true;
        _max_on_approach_iterations = parameter.as_int();
        if (_max_on_approach_iterations <= 0) {
          RCLCPP_INFO(
            _logger, "On approach iteration selected as <= 0, "
            "disabling tolerance and on approach iterations.");
          _max_on_approach_iterations = std::numeric_limits<int>::max();
        }
      } else if (name == _name + ".angle_quantization_bins") {
        reinit_collision_checker = true;
        reinit_a_star = true;
        int angle_quantizations = parameter.as_int();
        _angle_bin_size = 2.0 * M_PI / angle_quantizations;
        _angle_quantizations = static_cast<unsigned int>(angle_quantizations);
      }
    } else if (type == ParameterType::PARAMETER_STRING) {
      if (name == _name + ".motion_model_for_search") {
        reinit_a_star = true;
        _motion_model = fromString(parameter.as_string());
        if (_motion_model == MotionModel::UNKNOWN) {
          RCLCPP_WARN(
            _logger,
            "Unable to get MotionModel search type. Given '%s', "
            "valid options are MOORE, VON_NEUMANN, DUBIN, REEDS_SHEPP.",
            _motion_model_for_search.c_str());
        }
      }
    }
  }

  // Re-init if needed with mutex lock (to avoid re-init while creating a plan)
  if (reinit_a_star || reinit_downsampler || reinit_collision_checker || reinit_smoother) {
    // convert to grid coordinates
    if (!_downsample_costmap) {
      _downsampling_factor = 1;
    }
    _search_info.minimum_turning_radius =
      _minimum_turning_radius_global_coords / (_costmap->getResolution() * _downsampling_factor);
    _lookup_table_dim =
      static_cast<float>(_lookup_table_size) /
      static_cast<float>(_costmap->getResolution() * _downsampling_factor);

    // Make sure its a whole number
    _lookup_table_dim = static_cast<float>(static_cast<int>(_lookup_table_dim));

    // Make sure its an odd number
    if (static_cast<int>(_lookup_table_dim) % 2 == 0) {
      RCLCPP_INFO(
        _logger,
        "Even sized heuristic lookup table size set %f, increasing size by 1 to make odd",
        _lookup_table_dim);
      _lookup_table_dim += 1.0;
    }

    auto node = _node.lock();

    // Re-Initialize A* template
    if (reinit_a_star) {
      _a_star = std::make_unique<AStarAlgorithm<NodeHybrid>>(_motion_model, _search_info);
      _a_star->initialize(
        _allow_unknown,
        _max_iterations,
        _max_on_approach_iterations,
        _max_planning_time,
        _lookup_table_dim,
        _angle_quantizations);
    }

    // Re-Initialize costmap downsampler
    if (reinit_downsampler) {
      if (_downsample_costmap && _downsampling_factor > 1) {
        std::string topic_name = "downsampled_costmap";
        _costmap_downsampler = std::make_unique<CostmapDownsampler>();
        _costmap_downsampler->on_configure(
          node, _global_frame, topic_name, _costmap, _downsampling_factor);
      }
    }

    // Re-Initialize collision checker
    if (reinit_collision_checker) {
      _collision_checker = GridCollisionChecker(_costmap, _angle_quantizations, node);
      _collision_checker.setFootprint(
        _costmap_ros->getRobotFootprint(),
        _costmap_ros->getUseRadius(),
        findCircumscribedCost(_costmap_ros));
    }

    // Re-Initialize smoother
    if (reinit_smoother) {
      SmootherParams params;
      params.get(node, _name);
      _smoother = std::make_unique<Smoother>(params);
      _smoother->initialize(_minimum_turning_radius_global_coords);
    }

  if (reinit_a_star || reinit_downsampler || reinit_collision_checker || reinit_smoother) {
    RCLCPP_INFO(_logger,
      "[SMAC][Dyn] reinit: A*=%s  DS=%s  CollChk=%s  Smooth=%s  -> "
      "lookup_dim=%.0f bins=%u min_turn_r_cells=%.1f",
      reinit_a_star ? "Y":"N",
      reinit_downsampler ? "Y":"N",
      reinit_collision_checker ? "Y":"N",
      reinit_smoother ? "Y":"N",
      _lookup_table_dim, _angle_quantizations, _search_info.minimum_turning_radius);
    }

  }
  result.successful = true;
  return result;
}

}  // namespace nav2_smac_planner

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_smac_planner::SmacPlannerHybrid, nav2_core::GlobalPlanner)
