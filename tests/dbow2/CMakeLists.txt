INCLUDE_DIRECTORIES(. ${OpenCV_INCLUDE_DIRS})

FILE(GLOB hdrs_base "*.h" )
FILE(GLOB srcs_base "*.cpp")


FILE(GLOB hdrs ${hdrs_base}  )
FILE(GLOB srcs  ${srcs_base} )


SET(LIBNAME dbow2)
ADD_LIBRARY(${LIBNAME} ${srcs} ${hdrs})
MESSAGE(STATUS "dbow2 :${LIBNAME}: ${srcs} ${hdrs}")
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} ) 

SET_TARGET_PROPERTIES(${LIBNAME} PROPERTIES 				# create *nix style library versions + symbolic links
	VERSION ${PROJECT_VERSION}
	SOVERSION ${PROJECT_SOVERSION}
	CLEAN_DIRECT_OUTPUT 1 					# allow creating static and shared libs without conflicts
        OUTPUT_NAME "${LIBNAME}${PROJECT_DLLVERSION}" 	# avoid conflicts between library and binary target names
)

#TARGET_LINK_LIBRARIES(${LIBNAME} ${REQUIRED_LIBRARIES} )

INSTALL(TARGETS ${LIBNAME}
	RUNTIME DESTINATION bin COMPONENT main			# Install the dll file in bin directory
	LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT main
        ARCHIVE DESTINATION lib COMPONENT main)			# Install the dll.a file in lib directory



INSTALL(FILES ${hdrs_base}
        DESTINATION include/${LIBNAME}
        COMPONENT main)


