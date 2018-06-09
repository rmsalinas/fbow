# ----------------------------------------------------------------------------
#   Basic Configuration
# ----------------------------------------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(fbow)
set(PROJECT_VERSION "0.0.1")
string(REGEX MATCHALL "[0-9]" PROJECT_VERSION_PARTS "${PROJECT_VERSION}")
list(GET PROJECT_VERSION_PARTS 0 PROJECT_VERSION_MAJOR)
list(GET PROJECT_VERSION_PARTS 1 PROJECT_VERSION_MINOR)
list(GET PROJECT_VERSION_PARTS 2 PROJECT_VERSION_PATCH)
set(PROJECT_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

#------------------------------------------------------
# Build type
#------------------------------------------------------

IF(NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE "Release" )
ENDIF()

#------------------------------------------------------
# Lib Names and Dirs
#------------------------------------------------------

if(WIN32)
    # Postfix of DLLs:
    SET(PROJECT_DLLVERSION "${PROJECT_VERSION_MAJOR}${PROJECT_VERSION_MINOR}${PROJECT_VERSION_PATCH}")
    SET(RUNTIME_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Directory for dlls and binaries")
    SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Directory for binaries")
    SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Directory for dlls")
else()
    # Postfix of so's:
    set(PROJECT_DLLVERSION)
    SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_INSTALL_PREFIX}/lib/cmake/ /usr/lib/cmake /home/mjmarin/libs/ORB_SLAM2/lib/)

endif()



#
OPTION(USE_OWN_EIGEN3	"Set to OFF to use a standard eigen3 version" ON)
OPTION(BUILD_UTILS	"Set to OFF to not build utils" ON)
OPTION(BUILD_SHARED_LIBS 	"Set to OFF to build static libraries" ON)
OPTION(USE_CONTRIB "Set on/off" OFF)

OPTION(USE_AVX "Set on/off" ON)
OPTION(USE_SSE3 "Set on/off" ON)

# ----------------------------------------------------------------------------
#   Find Dependencies
# ----------------------------------------------------------------------------
find_package(OpenCV  REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
MESSAGE("OPENCVINCDIR=${OpenCV_INCLUDE_DIRS}")

IF(USE_CONTRIB)
add_definitions(-DUSE_CONTRIB)
ENDIF()
if(NOT OpenCV_VERSION VERSION_LESS "3.0")
    ADD_DEFINITIONS(-DOPENCV_VERSION_3)
    SET(OPENCV_VERSION_3 ON)
ELSE()
    SET(OPENCV_VERSION_3 OFF)
ENDIF()

SET(REQUIRED_LIBRARIES ${REQUIRED_LIBRARIES} ${OpenCV_LIBS})

# ----------------------------------------------------------------------------
#   PROJECT CONFIGURATION
#   force some variables that could be defined in the command line to be written to cache
# ----------------------------------------------------------------------------
OPTION(INSTALL_DOC 	"Set to ON to build/install Documentation" OFF)
IF (INSTALL_DOC)
    FIND_PACKAGE(Doxygen REQUIRED)
    MESSAGE( STATUS "INSTALL_DOC:         ${INSTALL_DOC} ")
    INCLUDE("${PROJECT_SOURCE_DIR}/generateDoc.cmake")
    GENERATE_DOCUMENTATION(${PROJECT_SOURCE_DIR}/dox.in)
ENDIF()
 
# ----------------------------------------------------------------------------
#   Uninstall target, for "make uninstall"
# ----------------------------------------------------------------------------
CONFIGURE_FILE( "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)
ADD_CUSTOM_TARGET(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")

# ----------------------------------------------------------------------------
# create configuration file from .in file (If you use windows take care with paths)
# ----------------------------------------------------------------------------

CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/config.cmake.in" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
INSTALL(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" DESTINATION share/${PROJECT_NAME}/ )




# ----------------------------------------------------------------------------
#   Program Optimization and debug (Extracted from OpenCV)
# ----------------------------------------------------------------------------
set(WARNINGS_ARE_ERRORS 		OFF CACHE BOOL "Treat warnings as errors")
set(WHOLE_PROGRAM_OPTIMIZATION 	OFF CACHE BOOL "Flags for whole program optimization.")

set(EXTRA_C_FLAGS "")
set(EXTRA_C_FLAGS_RELEASE "")
set(EXTRA_C_FLAGS_DEBUG "")
set(EXTRA_EXE_LINKER_FLAGS "")
set(EXTRA_EXE_LINKER_FLAGS_RELEASE "")
set(EXTRA_EXE_LINKER_FLAGS_DEBUG "")

IF(CMAKE_COMPILER_IS_GNUCXX OR MINGW)
    set(ENABLE_PROFILING 		OFF CACHE BOOL "Enable profiling in the GCC compiler (Add flags: -g -pg)")
    set(USE_OMIT_FRAME_POINTER 	ON CACHE BOOL "Enable -fomit-frame-pointer for GCC")
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES arm*) # We can use only -O2 because the -O3 causes gcc crash
        set(USE_O2 ON CACHE BOOL "Enable -O2 for GCC")
        set(USE_FAST_MATH OFF CACHE BOOL "Enable -ffast-math for GCC")
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES powerpc*)
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC")
        set(USE_POWERPC ON CACHE BOOL "Enable PowerPC for GCC")
    endif ()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64* OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64*)
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC")
        set(USE_FAST_MATH OFF CACHE BOOL "Enable -ffast-math for GCC")
        set(USE_MMX ON CACHE BOOL "Enable MMX for GCC")
        set(USE_SSE ON CACHE BOOL "Enable SSE for GCC")
        set(USE_SSE2 ON CACHE BOOL "Enable SSE2 for GCC")
        set(USE_SSE3 ON CACHE BOOL "Enable SSE3 for GCC")
    endif()
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES i686* OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES x86)
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC")
        set(USE_FAST_MATH OFF CACHE BOOL "Enable -ffast-math for GCC")
        set(USE_MMX ON CACHE BOOL "Enable MMX for GCC")
        set(USE_SSE OFF CACHE BOOL "Enable SSE for GCC")
        set(USE_SSE2 OFF CACHE BOOL "Enable SSE2 for GCC")
        set(USE_SSE3 OFF CACHE BOOL "Enable SSE3 for GCC")
    endif ()
    set(USE_AVX OFF CACHE BOOL "Enable AVX for GCC")


    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall")

    if(WARNINGS_ARE_ERRORS)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Werror")
    endif()

    # The -Wno-long-long is required in 64bit systems when including sytem headers.
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES x86_64* OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES amd64*)
		set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wno-long-long")
    endif()

    # Whole program optimization
    if(WHOLE_PROGRAM_OPTIMIZATION)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -fwhole-program --combine")
    endif()

    # Other optimizations
    if(USE_OMIT_FRAME_POINTER)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -fomit-frame-pointer")
    endif()
    if(USE_O2)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O2")
    endif()
    if(USE_O3)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
    endif()
    if(USE_FAST_MATH)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -ffast-math")
    endif()
    if(USE_POWERPC)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mcpu=G3 -mtune=G5")
    endif()
    if(USE_MMX)
       set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -mmmx")
       add_definitions(-DUSE_MMX)
    endif()
    if(USE_SSE)
       set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -msse ")
       add_definitions(-DUSE_SSE)
    endif()
    if(USE_SSE2)
       set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -msse2")
       add_definitions(-DUSE_SSE2)
    endif()
    if(USE_SSE3 AND NOT MINGW) # SSE3 should be disabled under MingW because it generates compiler errors
       set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -msse3")
       add_definitions(-DUSE_SSE3)
    endif()

    if(USE_SSE4 AND NOT MINGW) # SSE3 should be disabled under MingW because it generates compiler errors
       set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -msse4.1")
       add_definitions(-DUSE_SSE4)
    endif()

    IF(USE_AVX) 
      set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -mavx")
     add_definitions(-DUSE_AVX)
    endif()

 if(ENABLE_PROFILING)
        set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -pg -g")
    else()
        if(NOT APPLE)
            set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -ffunction-sections")
        endif()
    endif()


    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES armv7l) # In ARM_COrtex8 with neon, enalble vectorized operations
        SET(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}  ")
    ENDIF()


    set(CMAKE_CXX_FLAGS_RELEASE "${VECTORIAL_INSTRUCTIONS} ${EXTRA_C_FLAGS_RELEASE}  -O3  -std=c++11 -Wall -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${VECTORIAL_INSTRUCTIONS}  -g3  -DDEBUG -D_DEBUG   -std=c++11 -Wall")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${VECTORIAL_INSTRUCTIONS} -O2 -g3  -DDEBUG -D_DEBUG  -std=c++11 -Wall")

    message(STATUS "vec=${VECTORIAL_INSTRUCTIONS}")
    MESSAGE( STATUS "-------------------------------------------------------------------------------" )
    message( STATUS "GNU COMPILER")
    MESSAGE( STATUS "-------------------------------------------------------------------------------" )




ELSE()  # MSVC

add_definitions(-DNOMINMAX)


ENDIF()#END OF COMPILER SPECIFIC OPTIONS
find_package(OpenMP  )
if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} ")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS} ")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS} ")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_EXE_LINKER_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_EXE_LINKER_FLAGS_RELEASE}")
SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${EXTRA_EXE_LINKER_FLAGS_DEBUG}")


#------------------------------------------------
# DIRS
#------------------------------------------------
ADD_SUBDIRECTORY(src)
IF (BUILD_UTILS)
  ADD_SUBDIRECTORY(utils)
  ADD_SUBDIRECTORY(tests)
ENDIF()


# ----------------------------------------------------------------------------
# display status message for important variables
# ----------------------------------------------------------------------------
message( STATUS )
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
message( STATUS "General configuration for ${PROJECT_NAME} ${PROJECT_VERSION}")
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
message( STATUS )
message("    Built as dynamic libs?:" ${BUILD_SHARED_LIBS})
message("    Compiler:"                   "${CMAKE_COMPILER}"   "${CMAKE_CXX_COMPILER}")

message( STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message( STATUS "C++ flags (Release):       ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message( STATUS "C++ flags (Debug):         ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message( STATUS "C++ flags (Relase+Debug):         ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

message( STATUS "CMAKE_CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
message( STATUS "CMAKE_BINARY_DIR:         ${CMAKE_BINARY_DIR}")

MESSAGE( STATUS )
MESSAGE( STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}" )
MESSAGE( STATUS "BUILD_SHARED_LIBS = ${BUILD_SHARED_LIBS}" )
MESSAGE( STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
MESSAGE( STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}" )
MESSAGE( STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}" )
MESSAGE( STATUS "BUILD_UTILS= ${BUILD_UTILS}" )
MESSAGE( STATUS "BUILD_TESTS= ${BUILD_TESTS}" )
MESSAGE( STATUS "OPENCV_DIR= ${OpenCV_DIR}" ) 
 


MESSAGE( STATUS )
MESSAGE( STATUS "OpenCV_LIB_DIR=${OpenCV_LIB_DIR}")
MESSAGE( STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}")

MESSAGE( STATUS )
MESSAGE( STATUS )
MESSAGE( STATUS "Change a value with: cmake -D<Variable>=<Value>" )
MESSAGE( STATUS )
