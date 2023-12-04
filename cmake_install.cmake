<<<<<<< HEAD
# Install script for directory: /Users/juliaisaac/Suture-Placement
=======
# Install script for directory: /Users/virajramakrishnan/Desktop/Uni/AUTOLab/Suture-Placement
>>>>>>> 6bb2e7a (add CGAL mesh functionality)

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
<<<<<<< HEAD
    set(CMAKE_INSTALL_CONFIG_NAME "")
=======
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
>>>>>>> 6bb2e7a (add CGAL mesh functionality)
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
<<<<<<< HEAD
  set(CMAKE_OBJDUMP "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/objdump")
=======
  set(CMAKE_OBJDUMP "/Library/Developer/CommandLineTools/usr/bin/objdump")
>>>>>>> 6bb2e7a (add CGAL mesh functionality)
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
<<<<<<< HEAD
file(WRITE "/Users/juliaisaac/Suture-Placement/${CMAKE_INSTALL_MANIFEST}"
=======
file(WRITE "/Users/virajramakrishnan/Desktop/Uni/AUTOLab/Suture-Placement/${CMAKE_INSTALL_MANIFEST}"
>>>>>>> 6bb2e7a (add CGAL mesh functionality)
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
