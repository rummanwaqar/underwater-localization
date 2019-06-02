cmake_minimum_required(VERSION 2.8.3)
project(au_localization)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra")

find_package(catkin REQUIRED COMPONENTS
    au_core
    nav_msgs
    roscpp
    sensor_msgs
    tf2
    tf2_geometry_msgs
    tf2_ros
)

find_package(Eigen3 REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS
        au_core
        nav_msgs
        sensor_msgs
        tf2
        tf2_geometry_msgs
        tf2_ros
)

include_directories(
    include/
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

add_executable(localization_node
    src/localization.cpp
    src/localization_ros.cpp
    src/localization_node.cpp
)
add_dependencies(localization_node
    au_core_generate_messages_cpp
    ${catkin_EXPORTED_TARGETS}
    ${EIGEN3_LIBRARIES}
)
target_link_libraries(localization_node
    utils
    ${catkin_LIBRARIES}
)

#############
##  Tests  ##
#############

if(CATKIN_ENABLE_TESTING)
    catkin_add_gtest(localization_tests
        test/localization_tests.cpp
        src/localization.cpp
    )
    target_link_libraries(localization_tests
        ${catkin_LIBRARIES}
    )
    catkin_add_gtest(ukf_tests
        test/ukf_tests.cpp
    )
    catkin_add_gtest(ukf_helpers_tests
        test/ukf_helpers_tests.cpp
    )
endif()

#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(TARGETS localization_node
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
