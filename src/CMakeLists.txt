ADD_DEFINITIONS(-DNOMINMAX)

SET(  srcs  fbow.cpp  vocabulary_creator.cpp )
SET(  hdrs  fbow.h fbow_exports.h vocabulary_creator.h cpu.h)

SET (THIS_LIBNAME ${EXTRALIBNAME}fbow)


ADD_LIBRARY(${THIS_LIBNAME} ${srcs} ${hdrs})

SET_TARGET_PROPERTIES(${THIS_LIBNAME} PROPERTIES 				# create *nix style library versions + symbolic links
        DEFINE_SYMBOL FBOW_DSO_EXPORTS
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_SOVERSION}
        CLEAN_DIRECT_OUTPUT 1 					# allow creating static and shared libs without conflicts
        OUTPUT_NAME "${THIS_LIBNAME}${PROJECT_DLLVERSION}" 	# avoid conflicts between library and binary target names
)
TARGET_LINK_LIBRARIES(${THIS_LIBNAME} ${OpenCV_LIBS} ${OpenMP_CXX_LIBRARIES})

INSTALL(TARGETS ${THIS_LIBNAME}
        RUNTIME DESTINATION bin COMPONENT main			# Install the dll file in bin directory
        LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT main
        ARCHIVE DESTINATION lib COMPONENT main)			# Install the dll.a file in lib directory


IF(FBOW_DEVINSTALL)

    INSTALL(FILES ${hdrs}
            DESTINATION include/fbow
            COMPONENT main)
ENDIF()


#set(sources
#    fbow.cpp
#    vocabulary_creator.cpp
#)

#set(headers
#    fbow.h
#   fbow_exports.h
#    vocabulary_creator.h
#)

#add_library(fbow ${sources} ${headers})

#set_target_properties(fbow PROPERTIES          # create *nix style library versions + symbolic links
#    DEFINE_SYMBOL FBOW_DSO_EXPORTS
#    VERSION ${PROJECT_VERSION}
#    SOVERSION ${PROJECT_SOVERSION}
#    CXX_VISIBILITY_PRESET hidden                # Make all the symbols hidden, so we have to explicitly define an API
#    CLEAN_DIRECT_OUTPUT 1                       # allow creating static and shared libs without conflicts
#    OUTPUT_NAME "fbow${PROJECT_DLLVERSION}"    # avoid conflicts between library and binary target names
#)

#include(GenerateExportHeader)
#generate_export_header(fbow)

#target_link_libraries(fbow PUBLIC opencv_core)
#target_link_libraries(fbow PRIVATE opencv_features2d)

#target_include_directories(fbow PUBLIC
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#  $<INSTALL_INTERFACE:include>
#)

#install(TARGETS fbow EXPORT fbowConfig
#    RUNTIME DESTINATION bin
#    LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
#    ARCHIVE DESTINATION lib
#)


#install(EXPORT arucoConfig DESTINATION "share/aruco/cmake")
