INCLUDE_DIRECTORIES(. ./utils/)

FILE(GLOB hdrs_base "*.h" )
FILE(GLOB srcs_base "*.cpp")

FILE(GLOB hdrs ${hdrs_base}  )
FILE(GLOB srcs  ${srcs_base} )

SET(THIS_LIBNAME fbow)

ADD_LIBRARY(${THIS_LIBNAME} ${srcs} ${hdrs})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} ) 

SET_TARGET_PROPERTIES(${THIS_LIBNAME} PROPERTIES 				# create *nix style library versions + symbolic links
        DEFINE_SYMBOL FBOW_DSO_EXPORTS
	VERSION ${PROJECT_VERSION}
	SOVERSION ${PROJECT_SOVERSION}
	CLEAN_DIRECT_OUTPUT 1 					# allow creating static and shared libs without conflicts
        OUTPUT_NAME "${THIS_LIBNAME}${PROJECT_DLLVERSION}" 	# avoid conflicts between library and binary target names
)

TARGET_LINK_LIBRARIES(${THIS_LIBNAME} ${REQUIRED_LIBRARIES} )

INSTALL(TARGETS ${THIS_LIBNAME}
	RUNTIME DESTINATION bin COMPONENT main			# Install the dll file in bin directory
	LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT main
        ARCHIVE DESTINATION lib COMPONENT main)			# Install the dll.a file in lib directory



INSTALL(FILES ${hdrs_base}
        DESTINATION include/${THIS_LIBNAME}
        COMPONENT main)

INSTALL(FILES ${utils_hdr}
        DESTINATION include/${THIS_LIBNAME}/utils
        COMPONENT main)
INSTALL(FILES ${optimizers_hdr}
        DESTINATION include/${THIS_LIBNAME}/optimizers
        COMPONENT main)

