# CMake file for sba's demo program
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src )
LINK_LIBRARIES(${PROJECT_NAME})
 

ADD_EXECUTABLE(test_dbow2VSfbow test_dbow2VSfbow.cpp dbow2/ScoringObject.cpp dbow2/BowVector.h  dbow2/FClass.h  dbow2/FeatureVector.h   dbow2/FORB.h  dbow2/ScoringObject.h  dbow2/TemplatedVocabulary.h )
ADD_EXECUTABLE(test_cpu_x86   test_cpu_x86.cpp   )
#ADD_EXECUTABLE(test_flann test_flann.cpp  )
#ADD_EXECUTABLE(dbow2fbow dbow2fbow.cpp    dbow2/ScoringObject.cpp)

#find_package(Flann )
#if(${FLANN_FOUND})
#    include_directories(${FLANN_INCLUDE_DIRS})
#MESSAGE(STATUS "FLANN_LIBRARIES=${FLANN_LIBRARIES}")
#ADD_EXECUTABLE(test_knn   test_knn.cpp)
#target_link_libraries(test_knn  ${FLANN_LIBRARIES})
#ENDIF()
