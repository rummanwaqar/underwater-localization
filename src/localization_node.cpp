#include <au_localization/localization_ros.h>
#include <ros/ros.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "localization_node");
  ros::NodeHandle nh;
  ros::NodeHandle privateNh("~");

  LocalizationRos localizationNode(nh, privateNh);

  ros::spin();
  return 0;
}
