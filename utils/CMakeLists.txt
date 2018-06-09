# CMake file for sba's demo program
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src /home/mjmarin/libs/ORB_SLAM2/include)
LINK_LIBRARIES(${PROJECT_NAME})

ADD_EXECUTABLE(fbow_create_voc_step0 fbow_create_voc_step0.cpp dirreader.h)
ADD_EXECUTABLE(fbow_create_voc_step0_list fbow_create_voc_step0_list.cpp)
ADD_EXECUTABLE(fbow_create_voc_step1 fbow_create_voc_step1.cpp)
#ADD_EXECUTABLE(fbow_create_voc_step1_opencv fbow_create_voc_step1_opencv.cpp)
ADD_EXECUTABLE(fbow_transform fbow_transform.cpp)
ADD_EXECUTABLE(image_matching image_matching.cpp)
INSTALL(TARGETS  fbow_create_voc_step0    fbow_create_voc_step1 fbow_transform RUNTIME DESTINATION bin)
