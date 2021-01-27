cmake_minimum_required(VERSION 3.10)

project(AlgorithmCheatSheet)

set(PROJECT_NAME "AlgorithmCheatSheet")
set(PROJECT_VERSION 1.0)


# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

file(GLOB HEADER_FILES *.h)


set(SOURCE_FILES ${SOURCE_FILES})
set(${CMAKE_PROJECT_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(${CMAKE_PROJECT_NAME}
    main.cpp
    ${HEADER_FILES}
)



#Testing
enable_testing()
#find_package(GTest REQUIRED)
#include_directories(${GTEST_INCLUDE_DIRS})
#add_executable( runUnitTests test.cpp)

#target_link_libraries(runUnitTests ${GTEST_BOTH_LIBRARIES})
#add_test( runUnitTests runUnitTests )