#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "pcl_conversions/pcl_conversions.h"

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <Eigen/Dense>
#include <fstream>
#include <cmath>

class OdomNode : public rclcpp::Node {
public:
  OdomNode() : Node("lidar_imu_odometry") {
    using std::placeholders::_1;
    auto qos = rclcpp::SensorDataQoS();

    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/amr/lidar", qos, std::bind(&OdomNode::lidarCallback, this, _1));
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/livox/amr/imu", qos, std::bind(&OdomNode::imuCallback, this, _1));

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom_est", 10);

    csv_.open("/root/ros2_ws/traj.csv");
    csv_ << "time,x,y,yaw\n";

    global_pose_ = Eigen::Matrix4f::Identity();
    RCLCPP_INFO(this->get_logger(), "✅ ICP-based Lidar-IMU odometry initialized.");
  }

private:
  // ---------- IMU yaw integration ----------
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    double wz = msg->angular_velocity.z;
    double t = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    if (last_imu_time_ > 0) {
      double dt = t - last_imu_time_;
      if (std::abs(wz) < 5.0) {   // ignore spikes
        if (std::abs(wz) < 0.01 && init_count_ < 500) {
          bias_ += wz;
          init_count_++;
        }
        double wz_corr = wz - (bias_ / std::max(1, init_count_));
        yaw_ += wz_corr * dt;
      }
    }
    last_imu_time_ = t;
  }

  // ---------- LIDAR + ICP alignment ----------
  void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    // Limit to near-horizontal slice
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-0.3, 0.3);
    pass.filter(*cloud);

    // Downsample
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.1f, 0.1f, 0.1f);
    vg.filter(*cloud);

    // First frame → just store it
    if (!has_last_) {
      last_cloud_ = *cloud;
      has_last_ = true;
      return;
    }

    // --- ICP registration (last → current) ---
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(last_cloud_.makeShared());
    icp.setInputTarget(cloud);
    icp.setMaximumIterations(50);
    icp.setMaxCorrespondenceDistance(0.3);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1e-8);

    pcl::PointCloud<pcl::PointXYZ> aligned;
    icp.align(aligned);

    if (icp.hasConverged()) {
      Eigen::Matrix4f rel = icp.getFinalTransformation().inverse();

      // --- Fuse yaw: 90 % ICP + 10 % IMU ---
      double icp_yaw = std::atan2(rel(1,0), rel(0,0));
      double fused_yaw = 0.9 * icp_yaw + 0.1 * yaw_;
      Eigen::Matrix3f fusedRot = Eigen::AngleAxisf(fused_yaw, Eigen::Vector3f::UnitZ()).toRotationMatrix();
      rel.block<3,3>(0,0) = fusedRot;

      // --- Accumulate global pose ---
      global_pose_ = global_pose_ * rel;

      // --- Publish odometry ---
      nav_msgs::msg::Odometry odom;
      odom.header.stamp = msg->header.stamp;
      odom.header.frame_id = "odom";
      odom.child_frame_id = "base_link";
      odom.pose.pose.position.x = global_pose_(0,3);
      odom.pose.pose.position.y = global_pose_(1,3);
      odom.pose.pose.position.z = 0.0;

      Eigen::Quaternionf q(global_pose_.block<3,3>(0,0));
      odom.pose.pose.orientation.x = q.x();
      odom.pose.pose.orientation.y = q.y();
      odom.pose.pose.orientation.z = q.z();
      odom.pose.pose.orientation.w = q.w();
      odom_pub_->publish(odom);

      // --- Log & save trajectory ---
      double t = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
      csv_ << t << "," << global_pose_(0,3) << "," << global_pose_(1,3)
           << "," << fused_yaw << "\n";

      RCLCPP_INFO(this->get_logger(),
        "x=%.2f y=%.2f yaw=%.2f° fitness=%.4f",
        global_pose_(0,3), global_pose_(1,3),
        fused_yaw * 180.0 / M_PI, icp.getFitnessScore());
    } else {
      RCLCPP_WARN(this->get_logger(), "⚠️  ICP did not converge.");
    }

    last_cloud_ = *cloud;
  }

  // ---------- Members ----------
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

  pcl::PointCloud<pcl::PointXYZ> last_cloud_;
  bool has_last_ = false;

  double yaw_ = 0.0;
  double last_imu_time_ = 0.0;
  double bias_ = 0.0;
  int init_count_ = 0;

  Eigen::Matrix4f global_pose_;
  std::ofstream csv_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OdomNode>());
  rclcpp::shutdown();
  return 0;
}


// #include "rclcpp/rclcpp.hpp"
// #include "sensor_msgs/msg/imu.hpp"
// #include "sensor_msgs/msg/point_cloud2.hpp"
// #include "nav_msgs/msg/odometry.hpp"
// #include "pcl_conversions/pcl_conversions.h"

// #include <pcl/filters/passthrough.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/registration/icp.h>
// #include <Eigen/Dense>
// #include <fstream>
// #include <cmath>
// #include <iomanip>

// class OdomNode : public rclcpp::Node {
// public:
//   OdomNode() : Node("lidar_imu_odometry") {
//     using std::placeholders::_1;
//     auto qos = rclcpp::SensorDataQoS();

//     lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
//         "/livox/amr/lidar", qos, std::bind(&OdomNode::lidarCallback, this, _1));
//     imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
//         "/livox/amr/imu", qos, std::bind(&OdomNode::imuCallback, this, _1));

//     odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom_est", 10);

//     csv_.open("/root/ros2_ws/traj.csv");
//     csv_ << "time,x,y,yaw\n";

//     global_pose_ = Eigen::Matrix4f::Identity();
//     RCLCPP_INFO(this->get_logger(), "✅ ICP-based Lidar-IMU odometry initialized.");
//   }

// private:
//   // ---------- IMU yaw integration ----------
//   void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
//     double wz = msg->angular_velocity.z;
//     double t = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
//     if (last_imu_time_ > 0) {
//       double dt = t - last_imu_time_;
//       if (std::abs(wz) < 5.0) {   // ignore spikes
//         if (std::abs(wz) < 0.01 && init_count_ < 500) {
//           bias_ += wz;
//           init_count_++;
//         }
//         double wz_corr = wz - (bias_ / std::max(1, init_count_));
//         yaw_ += wz_corr * dt;
//       }
//     }
//     last_imu_time_ = t;
//   }

//   // ---------- LIDAR + ICP alignment ----------
//   void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::fromROSMsg(*msg, *cloud);

//     // Limit to near-horizontal slice
//     pcl::PassThrough<pcl::PointXYZ> pass;
//     pass.setInputCloud(cloud);
//     pass.setFilterFieldName("z");
//     pass.setFilterLimits(-0.3, 0.3);
//     pass.filter(*cloud);

//     // Downsample
//     pcl::VoxelGrid<pcl::PointXYZ> vg;
//     vg.setInputCloud(cloud);
//     vg.setLeafSize(0.1f, 0.1f, 0.1f);
//     vg.filter(*cloud);

//     // First frame → just store it
//     if (!has_last_) {
//       last_cloud_ = *cloud;
//       has_last_ = true;
//       return;
//     }

//     // --- ICP registration (last → current) ---
//     pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
//     icp.setInputSource(last_cloud_.makeShared());
//     icp.setInputTarget(cloud);
//     icp.setMaximumIterations(50);
//     icp.setMaxCorrespondenceDistance(0.5);
//     icp.setTransformationEpsilon(1e-6);
//     icp.setEuclideanFitnessEpsilon(1e-6);

//     pcl::PointCloud<pcl::PointXYZ> aligned;
//     icp.align(aligned);

//     if (icp.hasConverged()) {
//       // ✅ Fix 1: remove .inverse() (correct transform direction)
//       Eigen::Matrix4f rel = icp.getFinalTransformation();

//       // --- Fuse yaw safely ---
//       auto wrapAngle = [](double a) {
//         while (a > M_PI)  a -= 2*M_PI;
//         while (a < -M_PI) a += 2*M_PI;
//         return a;
//       };

//       double icp_yaw = std::atan2(rel(1,0), rel(0,0));
//       double fused_yaw = wrapAngle(0.9 * icp_yaw + 0.1 * wrapAngle(yaw_));

//       Eigen::Matrix3f fusedRot =
//           Eigen::AngleAxisf(fused_yaw, Eigen::Vector3f::UnitZ()).toRotationMatrix();
//       rel.block<3,3>(0,0) = fusedRot;

//       // --- Accumulate global pose ---
//       global_pose_ = global_pose_ * rel;

//       // --- Publish odometry ---
//       nav_msgs::msg::Odometry odom;
//       odom.header.stamp = msg->header.stamp;
//       odom.header.frame_id = "odom";
//       odom.child_frame_id = "base_link";
//       odom.pose.pose.position.x = global_pose_(0,3);
//       odom.pose.pose.position.y = global_pose_(1,3);
//       odom.pose.pose.position.z = 0.0;

//       Eigen::Quaternionf q(global_pose_.block<3,3>(0,0));
//       odom.pose.pose.orientation.x = q.x();
//       odom.pose.pose.orientation.y = q.y();
//       odom.pose.pose.orientation.z = q.z();
//       odom.pose.pose.orientation.w = q.w();
//       odom_pub_->publish(odom);

//       // ✅ Fix 3: use node clock to log time consistently
//       double t = this->now().seconds();
//       csv_ << std::fixed << std::setprecision(6)
//            << t << "," << global_pose_(0,3) << "," << global_pose_(1,3)
//            << "," << fused_yaw << "\n";

//       RCLCPP_INFO(this->get_logger(),
//         "x=%.2f y=%.2f yaw=%.2f° fitness=%.4f",
//         global_pose_(0,3), global_pose_(1,3),
//         fused_yaw * 180.0 / M_PI, icp.getFitnessScore());
//     } else {
//       RCLCPP_WARN(this->get_logger(), "⚠️  ICP did not converge.");
//     }

//     last_cloud_ = *cloud;
//   }

//   // ---------- Members ----------
//   rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
//   rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
//   rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

//   pcl::PointCloud<pcl::PointXYZ> last_cloud_;
//   bool has_last_ = false;

//   double yaw_ = 0.0;
//   double last_imu_time_ = 0.0;
//   double bias_ = 0.0;
//   int init_count_ = 0;

//   Eigen::Matrix4f global_pose_;
//   std::ofstream csv_;
// };

// int main(int argc, char **argv) {
//   rclcpp::init(argc, argv);
//   rclcpp::spin(std::make_shared<OdomNode>());
//   rclcpp::shutdown();
//   return 0;
// }
