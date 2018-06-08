# CMake file for sba's demo program
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src /home/mjmarin/libs/ORB_SLAM2/include)
LINK_LIBRARIES(${PROJECT_NAME})

ADD_EXECUTABLE(fbow_create_voc_step0 fbow_create_voc_step0.cpp)
ADD_EXECUTABLE(fbow_create_voc_step1 fbow_create_voc_step1.cpp)
#ADD_EXECUTABLE(fbow_create_voc_step1_opencv fbow_create_voc_step1_opencv.cpp)
ADD_EXECUTABLE(fbow_transform fbow_transform.cpp)
ADD_EXECUTABLE(image_matching image_matching.cpp)

#INSTALL(TARGETS  slammarker      RUNTIME DESTINATION bin)
