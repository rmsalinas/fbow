# CMake file for sba's demo program
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src )
LINK_LIBRARIES(${PROJECT_NAME})
 

ADD_EXECUTABLE(fbow_test_dbow2VSfbow test_dbow2VSfbow.cpp dbow2/ScoringObject.cpp dbow2/BowVector.h  dbow2/FClass.h  dbow2/FeatureVector.h   dbow2/FORB.h  dbow2/ScoringObject.h  dbow2/TemplatedVocabulary.h )
ADD_EXECUTABLE(fbow_test_cpu_x86   test_cpu_x86.cpp   )
INSTALL(TARGETS  fbow_test_cpu_x86      RUNTIME DESTINATION bin)

