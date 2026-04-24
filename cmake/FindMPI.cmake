# FindMPI.cmake - Locate MPI library and headers
#
# This module defines the following variables:
#   MPI_FOUND          - True if the MPI library and headers are found
#   MPI_INCLUDE_DIRS   - The include directories for MPI
#   MPI_LIBRARIES      - The libraries to link against for MPI
#   MPI_VERSION        - The version string of MPI (if available)
#   MPI_CXX_FOUND      - True if the C++ bindings are found
#
# Required version: MPI 3.1 or higher
#
# Usage:
#   find_package(MPI 3.1)
#
# Supported implementations: OpenMPI, MPICH, Intel MPI

# Early return if target is already defined
if(TARGET MPI::MPI_CXX)
    return()
endif()

# Minimum MPI version requirement (for MPI_T_* functions)
set(MPI_MIN_VERSION "3.1")

# Define search paths in priority order
set(MPI_SEARCH_PATHS
    ${MPI_DIR}
    $ENV{MPI_DIR}
    /usr/lib/x86_64-linux-gnu
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /usr/lib
)

# Locate the MPI header file (mpi.h)
find_path(MPI_INCLUDE_DIR
    NAMES mpi.h
    HINTS ${MPI_DIR} $ENV{MPI_DIR}
    PATHS ${MPI_SEARCH_PATHS}
    PATH_SUFFIXES include include/mpi include/mpiport
    DOC "MPI include directory containing mpi.h"
)

# Locate the MPI C++ library (mpi_cxx or mpi for older implementations)
find_library(MPI_CXX_LIBRARY
    NAMES mpi_cxx mpi
    HINTS ${MPI_DIR} $ENV{MPI_DIR}
    PATHS ${MPI_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
    DOC "MPI C++ library (libmpi_cxx.so or libmpi.so)"
)

# Locate the base MPI library (always needed)
find_library(MPI_LIBRARY
    NAMES mpi
    HINTS ${MPI_DIR} $ENV{MPI_DIR}
    PATHS ${MPI_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
    DOC "MPI library (libmpi.so)"
)

# Combine libraries
if(MPI_CXX_LIBRARY AND MPI_LIBRARY)
    set(MPI_LIBRARIES ${MPI_CXX_LIBRARY} ${MPI_LIBRARY})
elseif(MPI_LIBRARY)
    set(MPI_LIBRARIES ${MPI_LIBRARY})
endif()

set(MPI_INCLUDE_DIRS ${MPI_INCLUDE_DIR})
set(MPI_CXX_FOUND ${MPI_CXX_LIBRARY} )

# Version detection from mpi.h
if(MPI_INCLUDE_DIR AND EXISTS "${MPI_INCLUDE_DIR}/mpi.h")
    # Try to extract version from mpi.h header
    file(STRINGS "${MPI_INCLUDE_DIR}/mpi.h" MPI_VERSION_DEF
        REGEX "^#define[ \t]+MPI_VERSION[ \t]+[0-9]+"
        LIMIT_COUNT 1
    )

    if(MPI_VERSION_DEF)
        string(REGEX REPLACE "^#define[ \t]+MPI_VERSION[ \t]+([0-9]+).*" "\\1"
            MPI_VERSION_MAJOR "${MPI_VERSION_DEF}")

        # Try to get subversion
        file(STRINGS "${MPI_INCLUDE_DIR}/mpi.h" MPI_SUBVERSION_DEF
            REGEX "^#define[ \t]+MPI_SUBVERSION[ \t]+[0-9]+"
            LIMIT_COUNT 1
        )

        if(MPI_SUBVERSION_DEF)
            string(REGEX REPLACE "^#define[ \t]+MPI_SUBVERSION[ \t]+([0-9]+).*" "\\1"
                MPI_VERSION_MINOR "${MPI_SUBVERSION_DEF}")
        else()
            set(MPI_VERSION_MINOR "0")
        endif()

        set(MPI_VERSION "${MPI_VERSION_MAJOR}.${MPI_VERSION_MINOR}")
    endif()
endif()

# Handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPI
    REQUIRED_VARS
        MPI_LIBRARY
        MPI_INCLUDE_DIRS
    VERSION_VAR
        MPI_VERSION
    FAIL_MESSAGE
        "MPI not found. Set MPI_DIR environment variable or install MPI 3.1+. "
        "See https://www.open-mpi.org or https://www.mpich.org for installation instructions."
)

# Create imported target MPI::MPI_CXX
if(MPI_FOUND AND NOT TARGET MPI::MPI_CXX)
    if(MPI_CXX_LIBRARY)
        add_library(MPI::MPI_CXX UNKNOWN IMPORTED)
        set_target_properties(MPI::MPI_CXX PROPERTIES
            IMPORTED_LOCATION "${MPI_CXX_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${MPI_INCLUDE_DIRS}"
        )

        # Add the base MPI library as a dependency
        if(MPI_LIBRARY AND NOT MPI_LIBRARY STREQUAL MPI_CXX_LIBRARY)
            set_target_properties(MPI::MPI_CXX PROPERTIES
                IMPORTED_LINK_INTERFACE_LIBRARIES "${MPI_LIBRARY}"
            )
        endif()
    else()
        # Fall back to just MPI library
        add_library(MPI::MPI_CXX UNKNOWN IMPORTED)
        set_target_properties(MPI::MPI_CXX PROPERTIES
            IMPORTED_LOCATION "${MPI_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${MPI_INCLUDE_DIRS}"
        )
    endif()

    # Set version target property
    if(MPI_VERSION)
        set_target_properties(MPI::MPI_CXX PROPERTIES
            INTERFACE_MPI_VERSION "${MPI_VERSION}"
        )
    endif()
endif()

# Mark variables as advanced
mark_as_advanced(MPI_INCLUDE_DIR)
mark_as_advanced(MPI_LIBRARY)
mark_as_advanced(MPI_CXX_LIBRARY)
